import os
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List

TARGET_DEPTHS_DEFAULT = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

class HarmonicClimatology:
    """
    Seasonal Harmonic Climatology Model:
      T_clim(lat, lon, depth, seasonal_time) = 
          a0 + a1*sin(phase) + b1*cos(phase) + a2*sin(2*phase) + b2*cos(2*phase)
    where phase = 2 * pi * seasonal_time, with seasonal_time = (day_of_year - 1) / 365.25 in [0, 1).

    Fitted strictly using training partition data only.
    Saved to disk and frozen across validation, GLORYS test, and ARGO test.
    """
    def __init__(
        self,
        lat_min: float = 5.0,
        lat_max: float = 30.0,
        lon_min: float = 45.0,
        lon_max: float = 105.0,
        resolution: float = 0.25,
        target_depths: Optional[List[float]] = None,
    ):
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.resolution = resolution
        self.target_depths = target_depths or TARGET_DEPTHS_DEFAULT
        self.n_depths = len(self.target_depths)

        # Coordinate axes
        self.lat_coords = np.arange(lat_min, lat_max + resolution * 0.5, resolution, dtype=np.float32)
        self.lon_coords = np.arange(lon_min, lon_max + resolution * 0.5, resolution, dtype=np.float32)
        self.n_lat = len(self.lat_coords)
        self.n_lon = len(self.lon_coords)

        # Coefficient grid: [n_lat, n_lon, n_depths, 5]
        # 5 harmonic terms: [1.0, sin(phase), cos(phase), sin(2*phase), cos(2*phase)]
        self.coeffs = np.zeros((self.n_lat, self.n_lon, self.n_depths, 5), dtype=np.float32)
        # Regional fallback profile per depth
        self.global_fallback = np.zeros((self.n_depths, 5), dtype=np.float32)
        self.is_fitted = False

        # PyTorch cached buffers for high-throughput batch evaluation
        self._torch_coeffs: Optional[torch.Tensor] = None
        self._torch_fallback: Optional[torch.Tensor] = None

    def _lat_to_idx(self, lat: np.ndarray) -> np.ndarray:
        idx = np.round((lat - self.lat_min) / self.resolution).astype(np.int32)
        return np.clip(idx, 0, self.n_lat - 1)

    def _lon_to_idx(self, lon: np.ndarray) -> np.ndarray:
        idx = np.round((lon - self.lon_min) / self.resolution).astype(np.int32)
        return np.clip(idx, 0, self.n_lon - 1)

    def fit_from_samples(
        self,
        lats: np.ndarray,            # [N]
        lons: np.ndarray,            # [N]
        seasonal_times: np.ndarray,  # [N]
        temps: np.ndarray,           # [N, 15]
        depth_masks: np.ndarray,     # [N, 15]
        ridge_lambda: float = 1e-3,
    ):
        """
        Fit harmonic coefficients using Ridge regression per spatial grid cell and depth level.
        """
        print(f"  Fitting harmonic climatology from {len(lats):,} training samples...")
        N = len(lats)
        phase = 2.0 * np.pi * seasonal_times
        X = np.stack([
            np.ones(N, dtype=np.float32),
            np.sin(phase),
            np.cos(phase),
            np.sin(2.0 * phase),
            np.cos(2.0 * phase),
        ], axis=1)  # [N, 5]

        # 1. Compute global fallback per depth across all valid training observations
        for d in range(self.n_depths):
            dm = depth_masks[:, d].astype(bool)
            if dm.sum() >= 5:
                Xd = X[dm]
                yd = temps[dm, d]
                reg = ridge_lambda * np.eye(5)
                w = np.linalg.solve(Xd.T @ Xd + reg, Xd.T @ yd)
                self.global_fallback[d] = w
            elif dm.sum() > 0:
                self.global_fallback[d, 0] = np.mean(temps[dm, d])

        # 2. Accumulate normal equations per (lat_idx, lon_idx, depth)
        lat_idxs = self._lat_to_idx(lats)
        lon_idxs = self._lon_to_idx(lons)

        # Dictionary of (lat_i, lon_i) -> list of sample indices
        cell_map: Dict[Tuple[int, int], List[int]] = {}
        for i in range(N):
            key = (lat_idxs[i], lon_idxs[i])
            cell_map.setdefault(key, []).append(i)

        fitted_cells = 0
        reg = ridge_lambda * np.eye(5)

        for (li, lj), s_idxs in cell_map.items():
            s_arr = np.array(s_idxs, dtype=np.int32)
            Xs = X[s_arr]
            for d in range(self.n_depths):
                dms = depth_masks[s_arr, d].astype(bool)
                if dms.sum() >= 8:  # Enough seasonal observations
                    Xsd = Xs[dms]
                    ysd = temps[s_arr[dms], d]
                    w = np.linalg.solve(Xsd.T @ Xsd + reg, Xsd.T @ ysd)
                    self.coeffs[li, lj, d] = w
                elif dms.sum() > 0:
                    self.coeffs[li, lj, d, 0] = np.mean(temps[s_arr[dms], d])
                    # inherit seasonal harmonics from fallback
                    self.coeffs[li, lj, d, 1:] = self.global_fallback[d, 1:]
                else:
                    self.coeffs[li, lj, d] = self.global_fallback[d]
            fitted_cells += 1

        # Fill untouched cells with global fallback
        for li in range(self.n_lat):
            for lj in range(self.n_lon):
                if (li, lj) not in cell_map:
                    self.coeffs[li, lj] = self.global_fallback

        self.is_fitted = True
        print(f"  Climatology fitted over {fitted_cells:,} active cells ({self.n_lat}x{self.n_lon} grid).")

    def save(self, filepath: str):
        """Save climatology coefficients and metadata."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        # Use npz for portable, fast loading without NetCDF dependency issues
        np.savez_compressed(
            filepath,
            coeffs=self.coeffs,
            global_fallback=self.global_fallback,
            lat_min=self.lat_min,
            lat_max=self.lat_max,
            lon_min=self.lon_min,
            lon_max=self.lon_max,
            resolution=self.resolution,
            target_depths=np.array(self.target_depths, dtype=np.float32),
        )
        print(f"  Harmonic climatology saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "HarmonicClimatology":
        """Load frozen climatology from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Climatology file not found at: {filepath}")
        data = np.load(filepath)
        obj = cls(
            lat_min=float(data["lat_min"]),
            lat_max=float(data["lat_max"]),
            lon_min=float(data["lon_min"]),
            lon_max=float(data["lon_max"]),
            resolution=float(data["resolution"]),
            target_depths=data["target_depths"].tolist(),
        )
        obj.coeffs = data["coeffs"].astype(np.float32)
        obj.global_fallback = data["global_fallback"].astype(np.float32)
        obj.is_fitted = True
        return obj

    def lookup_torch(
        self,
        latitude: torch.Tensor,       # [B]
        longitude: torch.Tensor,      # [B]
        seasonal_time: torch.Tensor,  # [B]
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Fast GPU/CPU vectorized lookup of climatological profiles [B, 15].
        """
        if device is None:
            device = latitude.device

        if self._torch_coeffs is None or self._torch_coeffs.device != device:
            self._torch_coeffs = torch.from_numpy(self.coeffs).to(device)

        B = latitude.shape[0]
        # Compute grid indices
        lat_idx = torch.round((latitude - self.lat_min) / self.resolution).long().clamp(0, self.n_lat - 1)
        lon_idx = torch.round((longitude - self.lon_min) / self.resolution).long().clamp(0, self.n_lon - 1)

        # Gather coefficients: [B, 15, 5]
        coeffs_b = self._torch_coeffs[lat_idx, lon_idx]  # [B, 15, 5]

        # Compute harmonic basis: [B, 5]
        phase = 2.0 * math.pi * seasonal_time  # [B]
        basis = torch.stack([
            torch.ones_like(phase),
            torch.sin(phase),
            torch.cos(phase),
            torch.sin(2.0 * phase),
            torch.cos(2.0 * phase),
        ], dim=-1)  # [B, 5]

        # Inner product: sum over harmonic dimension -> [B, 15]
        t_clim = (coeffs_b * basis.unsqueeze(1)).sum(dim=-1)
        return t_clim
