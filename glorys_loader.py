"""
glorys_loader.py - Single canonical data loader for GLORYS ocean observations.

Handles:
  - Discovery and verification of HDF5 monthly samples (samples_*.h5)
  - Chronological 4-year / 2-year partitioning (Train / Val / Test)
  - Normalization statistics calculation, caching, and persistence
  - Surface variables (SST, SSS, SLA, Currents, Winds) + 15 Target Depth Temperatures
  - Harmonic Fourier climatology integration for anomaly calculation
  - High-performance PyTorch DataLoader construction with worker memory pinning
"""
from __future__ import annotations
import os
import sys
import glob
import datetime
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _discover_files(data_dir: str) -> List[str]:
    pattern = os.path.join(data_dir, "samples_*.h5")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No HDF5 files matching samples_*.h5 found in: {data_dir}\n"
            "Check data_dir path in your configuration."
        )
    valid = []
    for p in paths:
        try:
            with h5py.File(p, "r") as f:
                if "surface_data" in f and f["surface_data"].shape[0] > 0:
                    valid.append(p)
        except Exception:
            continue
    if not valid:
        raise RuntimeError("All discovered HDF5 files are empty or unreadable.")
    return valid


def _file_counts(paths: List[str]) -> List[int]:
    counts = []
    for p in paths:
        with h5py.File(p, "r") as f:
            counts.append(int(f["surface_data"].shape[0]))
    return counts


def _build_index(paths: List[str], counts: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    total = sum(counts)
    fi = np.empty(total, dtype=np.int32)
    li = np.empty(total, dtype=np.int32)
    offset = 0
    for file_idx, n in enumerate(counts):
        fi[offset : offset + n] = file_idx
        li[offset : offset + n] = np.arange(n, dtype=np.int32)
        offset += n
    return fi, li


def _format_date(date_ordinal: int) -> str:
    d = datetime.date(1970, 1, 1) + datetime.timedelta(days=int(date_ordinal))
    return d.isoformat()


def _get_sample_date(paths: List[str], file_idx: int, local_idx: int) -> str:
    with h5py.File(paths[file_idx], "r") as f:
        date_ord = int(f["date"][local_idx])
    return _format_date(date_ord)


def _dayofyear_sincos(date_ordinal: int) -> Tuple[float, float]:
    d = datetime.date(1970, 1, 1) + datetime.timedelta(days=int(date_ordinal))
    doy = d.timetuple().tm_yday
    frac = 2.0 * np.pi * (doy / 365.25)
    return float(np.sin(frac)), float(np.cos(frac))


class GlorysDataset(Dataset):
    """
    Standard PyTorch Dataset for GLORYS ocean reanalysis data.
    Provides surface predictors and 15-depth vertical temperature targets.
    """
    def __init__(
        self,
        paths: List[str],
        file_indices: np.ndarray,
        local_indices: np.ndarray,
        norm_stats: Optional[Dict[str, Any]] = None,
        climatology: Optional[Any] = None,
        max_samples: Optional[int] = None,
    ):
        self.paths = paths
        self.file_indices = file_indices
        self.local_indices = local_indices
        self.climatology = climatology
        self._handles: Dict[int, h5py.File] = {}

        if max_samples is not None and max_samples < len(self.file_indices):
            step = len(self.file_indices) / max_samples
            sel = [int(i * step) for i in range(max_samples)]
            self.file_indices = self.file_indices[sel]
            self.local_indices = self.local_indices[sel]

        self.length = len(self.file_indices)

        if norm_stats is not None:
            self.mean = np.asarray(norm_stats["mean"], dtype=np.float32)
            self.std = np.asarray(norm_stats["std"], dtype=np.float32)
            self.std = np.where(self.std < 1e-8, 1.0, self.std)
        else:
            self.mean = None
            self.std = None

    def _get_handle(self, fi: int) -> h5py.File:
        if fi not in self._handles:
            self._handles[fi] = h5py.File(self.paths[fi], "r")
        return self._handles[fi]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fi = int(self.file_indices[idx])
        li = int(self.local_indices[idx])
        f = self._get_handle(fi)

        surf = np.array(f["surface_data"][li], dtype=np.float32)  # [7, 11, 11]
        temp = np.array(f["temperature"][li], dtype=np.float32)   # [15]
        lat = float(f["latitude"][li] if "latitude" in f else f["lat"][li])
        lon = float(f["longitude"][li] if "longitude" in f else f["lon"][li])
        date_ord = int(f["date_ordinal"][li] if "date_ordinal" in f else f["date"][li])

        if "depth_mask" in f:
            dmask = np.array(f["depth_mask"][li], dtype=np.float32)
        else:
            dmask = np.ones_like(temp)

        if "surface_mask" in f:
            smask = np.array(f["surface_mask"][li], dtype=np.float32)
        else:
            smask = np.ones((11, 11), dtype=np.float32)

        sin_t, cos_t = _dayofyear_sincos(date_ord)
        seasonal_time = np.array([sin_t, cos_t], dtype=np.float32)

        # Normalize surface predictors
        if self.mean is not None and self.std is not None:
            surf = (surf - self.mean[:, None, None]) / self.std[:, None, None]
            surf = np.where(smask[None, :, :] > 0.5, surf, 0.0)

        # Harmonic climatology (if used)
        if self.climatology is not None:
            clim_temp = self.climatology.get_profile(lat, lon, date_ord)
            clim_temp = np.where(dmask > 0.5, clim_temp, 0.0).astype(np.float32)
            anom_temp = np.where(dmask > 0.5, temp - clim_temp, 0.0).astype(np.float32)
        else:
            clim_temp = np.zeros_like(temp)
            anom_temp = np.zeros_like(temp)

        return {
            "surface_data": torch.from_numpy(surf).float(),
            "temperature": torch.from_numpy(temp).float(),
            "anomaly_temperature": torch.from_numpy(anom_temp).float(),
            "climatology_temperature": torch.from_numpy(clim_temp).float(),
            "latitude": torch.tensor(lat, dtype=torch.float32),
            "longitude": torch.tensor(lon, dtype=torch.float32),
            "seasonal_time": torch.from_numpy(seasonal_time).float(),
            "surface_mask": torch.from_numpy(smask).float(),
            "depth_mask": torch.from_numpy(dmask).float(),
            "date_ordinal": torch.tensor(date_ord, dtype=torch.int64),
        }

    def close(self):
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self):
        self.close()


# Backward-compatible alias
Ocean4YearDataset = GlorysDataset


def compute_normalization_stats(
    data_dir: str,
    n_sample_files: int = 12,
    samples_per_file: int = 5000,
    seed: int = 42
) -> Dict[str, List[float]]:
    """Computes mean and std across surface variables for normalization."""
    paths = _discover_files(data_dir)
    rng = np.random.default_rng(seed)
    chosen_paths = rng.choice(paths, size=min(n_sample_files, len(paths)), replace=False)

    all_data = []
    for p in chosen_paths:
        with h5py.File(p, "r") as f:
            n_avail = f["surface_data"].shape[0]
            if n_avail == 0:
                continue
            k = min(samples_per_file, n_avail)
            sub_indices = rng.choice(n_avail, size=k, replace=False)
            sub_indices.sort()
            batch = f["surface_data"][sub_indices]   # [k, 7, 11, 11]
            if "surface_mask" in f:
                smask = f["surface_mask"][sub_indices] # [k, 11, 11]
                ocean_pixels = batch.transpose(0, 2, 3, 1)[smask > 0.5]      # [N, 7]
            else:
                ocean_pixels = batch.transpose(0, 2, 3, 1).reshape(-1, 7)
            if len(ocean_pixels) > 0:
                all_data.append(ocean_pixels)

    if not all_data:
        raise RuntimeError("No valid ocean data found to compute normalization statistics.")

    stacked = np.concatenate(all_data, axis=0)
    means = np.mean(stacked, axis=0).tolist()
    stds = np.std(stacked, axis=0).tolist()
    stds = [s if s > 1e-6 else 1.0 for s in stds]
    return {"mean": means, "std": stds}


def get_glorys_dataloaders(
    cfg: Any,
    norm_stats: Optional[Dict[str, Any]] = None,
    climatology: Optional[Any] = None,
    device: Optional[torch.device] = None,
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Discovers files, computes normalization stats, partitions data chronologically,
    and returns (train_loader, val_loader, test_loader, norm_stats).
    """
    paths = _discover_files(cfg.data_dir)
    counts = _file_counts(paths)
    fi, li = _build_index(paths, counts)
    total_samples = len(fi)

    if norm_stats is None:
        stats_cache = os.path.join(cfg.checkpoints_dir, "norm_stats.json")
        if os.path.exists(stats_cache):
            import json
            with open(stats_cache, "r") as fp:
                norm_stats = json.load(fp)
        else:
            norm_stats = compute_normalization_stats(cfg.data_dir)

    train_end = int(total_samples * cfg.train_frac)
    val_end = int(total_samples * (cfg.train_frac + cfg.val_frac))

    train_ds = GlorysDataset(paths, fi[:train_end], li[:train_end], norm_stats, climatology, max_samples)
    val_ds = GlorysDataset(paths, fi[train_end:val_end], li[train_end:val_end], norm_stats, climatology, max_samples)
    test_ds = GlorysDataset(paths, fi[val_end:], li[val_end:], norm_stats, climatology, max_samples)

    num_workers = getattr(cfg, "num_workers", 4)
    pin_memory = getattr(cfg, "pin_memory", False)
    batch_size = getattr(cfg, "batch_size", 32)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, norm_stats


# Backward-compatible alias
get_4yr_dataloaders = get_glorys_dataloaders
get_upgraded_dataloaders = get_glorys_dataloaders
