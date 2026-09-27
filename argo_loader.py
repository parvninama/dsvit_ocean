"""
argo_loader.py - Single canonical data loader for independent in-situ Argo observations.

Handles:
  1. Discovery, verification, and loading of Argo NetCDF float profiles
  2. Surface 0.25-deg neighborhood extraction and caching (SurfaceDataLoader)
  3. Spatial and temporal collocation between Argo observations and surface fields
  4. GPU tensor pre-conversion for high-speed batch evaluation
"""
from __future__ import annotations
import os
import glob
import math
import numpy as np
import pandas as pd
import xarray as xr
import torch
from typing import List, Dict, Any, Optional, Tuple

DEFAULT_TARGET_DEPTHS = np.array(
    [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0],
    dtype=np.float32
)

SURFACE_VARS = ['sst', 'sss', 'sla', 'u_current', 'v_current', 'u_wind', 'v_wind']


# ==============================================================================
# 1. ARGO NETCDF PROFILE LOADING & VERIFICATION
# ==============================================================================

def verify_argo_depth_coordinate(depth_values: np.ndarray, expected_depths: np.ndarray = DEFAULT_TARGET_DEPTHS, filename: str = "") -> bool:
    """Asserts that the depth coordinate of the Argo NetCDF file matches the expected 15 target depths."""
    depth_values = np.asarray(depth_values, dtype=np.float32)
    if len(depth_values) != len(expected_depths):
        raise ValueError(
            f"Depth dimension mismatch in {filename}: found {len(depth_values)} levels, expected {len(expected_depths)}."
        )
    if not np.allclose(depth_values, expected_depths, atol=1e-2):
        raise ValueError(
            f"Depth coordinate values mismatch in {filename}:\n"
            f"  Found:    {depth_values.tolist()}\n"
            f"  Expected: {expected_depths.tolist()}"
        )
    return True


def load_argo_profiles(
    argo_dir: str,
    max_profiles: Optional[int] = None,
    expected_depths: np.ndarray = DEFAULT_TARGET_DEPTHS
) -> List[Dict[str, Any]]:
    """Discovers all monthly argo_test_*.nc files, checks coordinates, and parses profile records."""
    pattern = os.path.join(argo_dir, "argo_test_*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No Argo test files matching 'argo_test_*.nc' found in {argo_dir}")

    profiles = []
    total_files = len(files)
    print(f"[argo_loader] Found {total_files} Argo NetCDF files in {argo_dir}")

    for file_idx, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        try:
            ds = xr.open_dataset(fpath)
        except Exception as e:
            print(f"[argo_loader] Warning: Could not open {fname}: {e}")
            continue

        if 'depth' in ds.coords:
            verify_argo_depth_coordinate(ds['depth'].values, expected_depths=expected_depths, filename=fname)

        n_profs = ds.sizes.get('profile', 0)
        times = pd.to_datetime(ds['time'].values)
        lats = ds['latitude'].values.astype(float)
        lons = ds['longitude'].values.astype(float)
        temps = ds['temperature'].values.astype(np.float32) # [N, 15]
        dmasks = ds['depth_mask'].values.astype(np.uint8)   # [N, 15]
        
        platforms = [str(p) for p in ds['platform_number'].values] if 'platform_number' in ds else ['UNKNOWN'] * n_profs
        cycles = [int(c) for c in ds['cycle_number'].values] if 'cycle_number' in ds else [0] * n_profs
        directions = [str(d) for d in ds['direction'].values] if 'direction' in ds else ['A'] * n_profs
        data_modes = [str(m) for m in ds['data_mode'].values] if 'data_mode' in ds else ['D'] * n_profs

        ds.close()

        for i in range(n_profs):
            p_time = times[i]
            p_lat = lats[i]
            p_lon = lons[i]
            plat = platforms[i]
            cyc = cycles[i]
            prof_id = f"{plat}_{cyc}_{p_time.strftime('%Y%m%d%H%M')}"

            profiles.append({
                "profile_id": prof_id,
                "platform_number": plat,
                "cycle_number": cyc,
                "direction": directions[i],
                "data_mode": data_modes[i],
                "time": p_time,
                "latitude": p_lat,
                "longitude": p_lon,
                "temperature": temps[i],
                "depth_mask": dmasks[i],
                "file_source": fname
            })

            if max_profiles is not None and len(profiles) >= max_profiles:
                print(f"[argo_loader] Reached maximum profile limit ({max_profiles})")
                return profiles

    print(f"[argo_loader] Successfully loaded {len(profiles)} Argo profiles from {total_files} files.")
    return profiles


# ==============================================================================
# 2. SURFACE DATA LOADER FOR COLLOCATION
# ==============================================================================

class SurfaceDataLoader:
    """Manages access, spatial extraction, masking, and normalization of 0.25-deg surface data for collocation."""
    def __init__(self, regrid_dir: str, norm_stats: Dict[str, Any]):
        self.regrid_dir = regrid_dir
        self.norm_mean = np.asarray(norm_stats['mean'], dtype=np.float32)  # [7]
        self.norm_std  = np.asarray(norm_stats['std'],  dtype=np.float32)  # [7]
        self.norm_std  = np.where(self.norm_std < 1e-8, 1.0, self.norm_std)

        self.file_index = []
        files = sorted(glob.glob(os.path.join(regrid_dir, "combined_*.nc")))
        if not files:
            raise FileNotFoundError(f"No regridded combined files found in {regrid_dir}")

        for fpath in files:
            fname = os.path.basename(fpath)
            parts = fname.replace("combined_", "").replace(".nc", "").split("_")
            if len(parts) == 2:
                d_start = pd.to_datetime(parts[0], format="%Y%m%d")
                d_end   = pd.to_datetime(parts[1], format="%Y%m%d")
                self.file_index.append((d_start, d_end, fpath))

        self._cached_path: Optional[str] = None
        self._cached_ds: Optional[xr.Dataset] = None

        first_ds = xr.open_dataset(files[0])
        self.grid_lats = first_ds['latitude'].values.astype(np.float32)
        self.grid_lons = first_ds['longitude'].values.astype(np.float32)
        first_ds.close()

    def _find_file_for_date(self, target_date: pd.Timestamp) -> Optional[str]:
        for d_start, d_end, fpath in self.file_index:
            if d_start <= target_date <= d_end:
                return fpath
        return None

    def _get_dataset(self, target_date: pd.Timestamp) -> Optional[xr.Dataset]:
        fpath = self._find_file_for_date(target_date)
        if fpath is None:
            return None
        if self._cached_path != fpath:
            if self._cached_ds is not None:
                self._cached_ds.close()
            self._cached_path = fpath
            self._cached_ds = xr.open_dataset(fpath)
        return self._cached_ds

    def extract_patch(
        self,
        target_lat: float,
        target_lon: float,
        target_time: pd.Timestamp,
        patch_size: int = 11
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, float, float]]:
        ds = self._get_dataset(target_time)
        if ds is None:
            return None

        time_vals = pd.to_datetime(ds['time'].values)
        time_diffs = np.abs(time_vals - target_time)
        t_idx = int(np.argmin(time_diffs))
        time_diff_hours = float(time_diffs[t_idx].total_seconds() / 3600.0)

        lat_diffs = np.abs(self.grid_lats - target_lat)
        lon_diffs = np.abs(self.grid_lons - target_lon)
        lat_idx = int(np.argmin(lat_diffs))
        lon_idx = int(np.argmin(lon_diffs))

        center_lat = float(self.grid_lats[lat_idx])
        center_lon = float(self.grid_lons[lon_idx])

        # Haversine distance
        dphi = math.radians(center_lat - target_lat)
        dlam = math.radians(center_lon - target_lon)
        a = math.sin(dphi/2.0)**2 + math.cos(math.radians(target_lat)) * math.cos(math.radians(center_lat)) * math.sin(dlam/2.0)**2
        spatial_dist_km = float(6371.0088 * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)))

        half = patch_size // 2
        y_min = lat_idx - half
        y_max = lat_idx + half + 1
        x_min = lon_idx - half
        x_max = lon_idx + half + 1

        if y_min < 0 or y_max > len(self.grid_lats) or x_min < 0 or x_max > len(self.grid_lons):
            return None

        patch_channels = []
        for var_name in SURFACE_VARS:
            da = ds[var_name].isel(time=t_idx, latitude=slice(y_min, y_max), longitude=slice(x_min, x_max))
            if "depth" in da.dims:
                da = da.isel(depth=0)
            patch_channels.append(da.values.astype(np.float32))

        patch_raw = np.stack(patch_channels, axis=0)  # [7, 11, 11]

        valid_masks = [~np.isnan(patch_raw[c]) for c in range(7)]
        surface_mask = np.logical_and.reduce(valid_masks).astype(np.float32)

        if surface_mask[half, half] == 0.0:
            return None

        patch_norm = (patch_raw - self.norm_mean[:, None, None]) / self.norm_std[:, None, None]
        patch_norm = np.where(surface_mask[None, :, :] > 0.5, patch_norm, 0.0)

        return patch_norm, surface_mask, center_lat, center_lon, time_diff_hours

    def close(self):
        if self._cached_ds is not None:
            self._cached_ds.close()
            self._cached_ds = None
            self._cached_path = None

    def __del__(self):
        self.close()


# ==============================================================================
# 3. SPATIAL & TEMPORAL COLLOCATION
# ==============================================================================

class RejectionTracker:
    def __init__(self):
        self.total_profiles = 0
        self.matched_profiles = 0
        self.rejections: Dict[str, List[str]] = {
            "missing_surface_data": [],
            "invalid_location": [],
            "excessive_spatial_distance": [],
            "excessive_temporal_difference": [],
            "no_valid_depths": [],
            "all_surface_cells_invalid": []
        }

    def record_acceptance(self):
        self.total_profiles += 1
        self.matched_profiles += 1

    def record_rejection(self, reason: str, profile_id: str):
        self.total_profiles += 1
        if reason not in self.rejections:
            self.rejections[reason] = []
        self.rejections[reason].append(profile_id)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_profiles": self.total_profiles,
            "matched_profiles": self.matched_profiles,
            "rejections": {k: len(v) for k, v in self.rejections.items()}
        }

    def print_summary(self):
        print(f"\n[RejectionTracker] Summary: {self.matched_profiles}/{self.total_profiles} profiles matched successfully.")
        for k, v in self.rejections.items():
            if len(v) > 0:
                print(f"  - {k}: {len(v)} rejected")


def collocate_profile(
    profile: Dict[str, Any],
    surface_loader: SurfaceDataLoader,
    max_spatial_distance_km: float = 30.0,
    max_temporal_difference_hours: float = 36.0,
    patch_size: int = 11,
    min_valid_depths: int = 1,
    tracker: Optional[RejectionTracker] = None,
) -> Optional[Dict[str, Any]]:
    prof_id = profile["profile_id"]
    lat = float(profile["latitude"])
    lon = float(profile["longitude"])
    obs_time = profile["time"]
    dmask = profile["depth_mask"]

    if np.sum(dmask) < min_valid_depths:
        if tracker: tracker.record_rejection("no_valid_depths", prof_id)
        return None

    res = surface_loader.extract_patch(lat, lon, obs_time, patch_size=patch_size)
    if res is None:
        if tracker: tracker.record_rejection("missing_surface_data", prof_id)
        return None

    patch_data, surface_mask, center_lat, center_lon, time_diff_hours = res
    if time_diff_hours > max_temporal_difference_hours:
        if tracker: tracker.record_rejection("excessive_temporal_difference", prof_id)
        return None

    if tracker: tracker.record_acceptance()
    return {
        "profile_id": prof_id,
        "platform_number": profile.get("platform_number", "UNKNOWN"),
        "cycle_number": profile.get("cycle_number", "UNKNOWN"),
        "direction": profile.get("direction", "A"),
        "data_mode": profile.get("data_mode", "R"),
        "file_source": profile.get("file_source", ""),
        "time": obs_time,
        "latitude": lat,
        "longitude": lon,
        "grid_lat": center_lat,
        "grid_lon": center_lon,
        "surface_data": patch_data,
        "surface_mask": surface_mask,
        "argo_temperature": profile["temperature"],
        "depth_mask": profile["depth_mask"],
        "time_diff_hours": time_diff_hours,
    }


def collocate_and_cache_argo(
    profiles: List[Dict[str, Any]],
    regrid_dir: str,
    norm_stats: Dict[str, Any],
    max_spatial_distance_km: float = 30.0,
    max_temporal_difference_hours: float = 36.0
) -> List[Dict[str, Any]]:
    """Collocates a list of Argo profiles with surface grid cells once and caches in memory."""
    loader = SurfaceDataLoader(regrid_dir, norm_stats)
    tracker = RejectionTracker()
    matched = []
    for p in profiles:
        rec = collocate_profile(
            p, loader,
            max_spatial_distance_km=max_spatial_distance_km,
            max_temporal_difference_hours=max_temporal_difference_hours,
            tracker=tracker
        )
        if rec is not None:
            matched.append(rec)
    loader.close()
    return matched


def prepare_argo_tensors(records: List[Dict[str, Any]], device: torch.device) -> Dict[str, torch.Tensor]:
    """Pre-converts cached records into GPU tensors for ultra-fast intra-batch validation."""
    surf_arr = np.stack([r["surface_data"] for r in records])
    lat_arr = np.array([r["latitude"] for r in records], dtype=np.float32)
    lon_arr = np.array([r["longitude"] for r in records], dtype=np.float32)
    smask_arr = np.stack([r["surface_mask"] for r in records])

    # Seasonal time
    seas_list = []
    for r in records:
        t = pd.to_datetime(r["time"])
        doy = t.timetuple().tm_yday
        frac = 2.0 * np.pi * (doy / 365.25)
        seas_list.append([float(np.sin(frac)), float(np.cos(frac))])
    seas_arr = np.array(seas_list, dtype=np.float32)

    target_arr = np.stack([r["argo_temperature"] for r in records])
    dmask_arr = np.stack([r["depth_mask"] for r in records])
    target_clean = np.where(dmask_arr > 0.5, target_arr, 0.0).astype(np.float32)

    return {
        "surface_data": torch.from_numpy(surf_arr).float().to(device),
        "latitude": torch.from_numpy(lat_arr).float().to(device),
        "longitude": torch.from_numpy(lon_arr).float().to(device),
        "seasonal_time": torch.from_numpy(seas_arr).float().to(device),
        "surface_mask": torch.from_numpy(smask_arr).float().to(device),
        "argo_temperature": torch.from_numpy(target_clean).float().to(device),
        "depth_mask": torch.from_numpy(dmask_arr).float().to(device),
    }