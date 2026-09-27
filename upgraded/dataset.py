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

from upgraded.climatology import HarmonicClimatology

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
        d = int(f["date_ordinal"][local_idx])
        return _format_date(d)


class UpgradedOceanDataset(Dataset):
    """
    Lazy HDF5 Dataset for single-day surface observations + seasonal time + subsurface profiles.
    Computes normalized seasonal_time on-the-fly from date_ordinal:
      seasonal_time = (day_of_year - 1) / 365.25 in [0, 1)
    """
    def __init__(
        self,
        paths: List[str],
        file_indices: Optional[np.ndarray] = None,
        local_indices: Optional[np.ndarray] = None,
        norm_stats: Optional[Dict[str, np.ndarray]] = None,
        indices: Optional[np.ndarray] = None,
        climatology: Optional[HarmonicClimatology] = None,
    ):
        super().__init__()
        self.paths = paths
        if file_indices is None or local_indices is None:
            counts = _file_counts(paths)
            fi_all, li_all = _build_index(paths, counts)
            if indices is not None:
                self._fi = fi_all[indices]
                self._li = li_all[indices]
            else:
                self._fi = fi_all
                self._li = li_all
        else:
            self._fi = file_indices
            self._li = local_indices

        self._len = len(self._fi)
        self.norm_stats = norm_stats
        self.climatology = climatology
        self._handles: Optional[Dict[int, h5py.File]] = None

    def __len__(self) -> int:
        return self._len

    def _get_handle(self, fi: int) -> h5py.File:
        if self._handles is None:
            self._handles = {}
        if fi not in self._handles:
            self._handles[fi] = h5py.File(self.paths[fi], "r")
        return self._handles[fi]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx = int(self._fi[idx])
        local_idx = int(self._li[idx])
        f = self._get_handle(file_idx)

        surface_data = f["surface_data"][local_idx].astype(np.float32)  # [7, 11, 11]
        temperature = f["temperature"][local_idx].astype(np.float32)    # [15]
        latitude = float(f["latitude"][local_idx])
        longitude = float(f["longitude"][local_idx])
        surface_mask = f["surface_mask"][local_idx].astype(np.float32)  # [11, 11]
        depth_mask = f["depth_mask"][local_idx].astype(np.float32)      # [15]
        date_ord = int(f["date_ordinal"][local_idx])

        # Compute seasonal time: (day_of_year - 1) / 365.25
        dt = datetime.date(1970, 1, 1) + datetime.timedelta(days=date_ord)
        doy = dt.timetuple().tm_yday
        seasonal_time = float((doy - 1.0) / 365.25)

        # Normalize surface channels (only over ocean cells)
        if self.norm_stats is not None:
            mean = self.norm_stats["mean"][:, None, None]
            std = self.norm_stats["std"][:, None, None]
            std = np.where(std < 1e-8, 1.0, std)
            surface_data = (surface_data - mean) / std
            surface_data *= surface_mask[None]

        sample = {
            "surface_data": torch.from_numpy(surface_data),
            "temperature": torch.from_numpy(temperature),
            "latitude": torch.tensor(latitude, dtype=torch.float32),
            "longitude": torch.tensor(longitude, dtype=torch.float32),
            "seasonal_time": torch.tensor(seasonal_time, dtype=torch.float32),
            "surface_mask": torch.from_numpy(surface_mask),
            "depth_mask": torch.from_numpy(depth_mask),
            "date": torch.tensor(date_ord, dtype=torch.int32),
        }
        return sample

    def __del__(self):
        if hasattr(self, "_handles") and self._handles is not None:
            for h in self._handles.values():
                try:
                    h.close()
                except Exception:
                    pass
            self._handles = None


def compute_training_norm_stats(
    paths: List[str],
    file_indices: np.ndarray,
    local_indices: np.ndarray,
    n_surface_vars: int = 7,
    subsample: int = 50000,
) -> Dict[str, np.ndarray]:
    total = len(file_indices)
    n_sub = min(subsample, total)
    rng = np.random.default_rng(42)
    sub = rng.choice(total, size=n_sub, replace=False)
    print(f"  Computing normalization stats from {n_sub:,} sampled training patches...")

    groups: Dict[int, List[int]] = {}
    for k in sub:
        fi = int(file_indices[k])
        li = int(local_indices[k])
        groups.setdefault(fi, []).append(li)

    n_arr = np.zeros(n_surface_vars, dtype=np.float64)
    sum_arr = np.zeros(n_surface_vars, dtype=np.float64)
    sum2 = np.zeros(n_surface_vars, dtype=np.float64)

    for fi in sorted(groups):
        with h5py.File(paths[fi], "r") as f:
            for li in groups[fi]:
                sd = f["surface_data"][li].astype(np.float64)
                sm = f["surface_mask"][li].astype(bool)
                for ch in range(n_surface_vars):
                    vals = sd[ch][sm]
                    if len(vals) == 0:
                        continue
                    n_arr[ch] += len(vals)
                    sum_arr[ch] += vals.sum()
                    sum2[ch] += (vals ** 2).sum()

    mean = sum_arr / np.maximum(n_arr, 1)
    var = sum2 / np.maximum(n_arr, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-8))
    names = ["SST", "SSS", "SLA", "U_curr", "V_curr", "U_wind", "V_wind"]
    print("  Channel normalization parameters (ocean cells only):")
    for ch in range(n_surface_vars):
        print(f"    {names[ch]:10s}: mean={mean[ch]:10.4f}  std={std[ch]:8.4f}")

    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def get_upgraded_dataloaders(
    cfg: Any,
    norm_stats: Optional[Dict[str, np.ndarray]] = None,
    climatology: Optional[HarmonicClimatology] = None,
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, np.ndarray]]:
    """
    Constructs train, validation, and untouched GLORYS test DataLoaders
    using the strict chronological partition protocol:
      - 90% development: 81% training, 9% validation
      - 10% untouched GLORYS test
    """
    paths = _discover_files(cfg.data_dir)
    counts = _file_counts(paths)
    fi_all, li_all = _build_index(paths, counts)
    n_total = len(fi_all)

    # 90% dev / 10% test split
    n_dev = int(round(n_total * 0.90))
    n_test = n_total - n_dev

    # Within dev: 81% train, 9% validation
    n_train = int(round(n_dev * 0.90))
    n_val = n_dev - n_train

    tr_fi, tr_li = fi_all[:n_train], li_all[:n_train]
    va_fi, va_li = fi_all[n_train:n_dev], li_all[n_train:n_dev]
    te_fi, te_li = fi_all[n_dev:], li_all[n_dev:]

    d_train_start = _get_sample_date(paths, tr_fi[0], tr_li[0])
    d_train_end = _get_sample_date(paths, tr_fi[-1], tr_li[-1])
    d_val_start = _get_sample_date(paths, va_fi[0], va_li[0])
    d_val_end = _get_sample_date(paths, va_fi[-1], va_li[-1])
    d_test_start = _get_sample_date(paths, te_fi[0], te_li[0])
    d_test_end = _get_sample_date(paths, te_fi[-1], te_li[-1])

    print("\n" + "=" * 70)
    print("  CHRONOLOGICAL DATASET PARTITIONING PROTOCOL")
    print("=" * 70)
    print(f"  Total Valid Samples : {n_total:,}")
    print(f"  Training (81%)      : {n_train:,}  [{d_train_start} to {d_train_end}]")
    print(f"  Validation (9%)     : {n_val:,}  [{d_val_start} to {d_val_end}]")
    print(f"  GLORYS Test (10%)   : {n_test:,}  [{d_test_start} to {d_test_end}]  (UNTOUCHED)")
    print("=" * 70 + "\n")

    # Optional cap for debug/sanity runs
    effective_max = getattr(cfg, "max_samples", None)
    if max_samples is not None:
        effective_max = max_samples
    if effective_max is not None and 0 < effective_max < n_train:
        print(f"  [DEBUG LIMIT] Capping training to {effective_max:,} samples.")
        tr_fi = tr_fi[:effective_max]
        tr_li = tr_li[:effective_max]
        val_cap = max(64, int(effective_max * 0.11))
        va_fi = va_fi[:val_cap]
        va_li = va_li[:val_cap]
        te_fi = te_fi[:val_cap]
        te_li = te_li[:val_cap]

    # Compute normalization statistics strictly from training partition
    if norm_stats is None:
        norm_stats = compute_training_norm_stats(paths, tr_fi, tr_li, getattr(cfg, "n_surface_vars", 7))

    train_ds = UpgradedOceanDataset(paths, tr_fi, tr_li, norm_stats=norm_stats, climatology=climatology)
    val_ds = UpgradedOceanDataset(paths, va_fi, va_li, norm_stats=norm_stats, climatology=climatology)
    test_ds = UpgradedOceanDataset(paths, te_fi, te_li, norm_stats=norm_stats, climatology=climatology)

    batch_size = getattr(cfg, "batch_size", 32)
    num_workers = getattr(cfg, "num_workers", 4)
    pin_memory = getattr(cfg, "pin_memory", False)
    drop_last = len(train_ds) >= batch_size

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )

    return train_loader, val_loader, test_loader, norm_stats


def get_4yr_dataloaders(cfg, device=None, norm_stats=None, climatology=None, max_samples: Optional[int] = None):
    """
    Returns (train_loader, val_loader, test_loader, norm_stats) for training and evaluation.

    * If 5+ calendar years are available: trains on 4 years (90% train, 10% val) and tests on 5th year.
    * If <5 years are available: partitions chronologically 80% dev (72% train, 8% val) and 20% test.
    * Uses all available CPU cores for DataLoader workers.
    """
    import os
    from torch.utils.data import DataLoader

    paths = _discover_files(cfg.data_dir)
    counts = _file_counts(paths)
    fi_all, li_all = _build_index(paths, counts)
    n_total = len(fi_all)

    # Determine calendar years rapidly per file
    years_list = []
    for p in paths:
        with h5py.File(p, "r") as f:
            n_samples = f["surface_data"].shape[0]
            if n_samples > 0:
                first_ord = int(f["date_ordinal"][0])
                yr = (datetime.date(1970, 1, 1) + datetime.timedelta(days=first_ord)).year
                years_list.append(np.full(n_samples, yr, dtype=np.int32))
    years = np.concatenate(years_list) if years_list else np.empty(0, dtype=np.int32)
    unique_years = np.sort(np.unique(years))

    if len(unique_years) >= 5:
        train_years = unique_years[:4]
        test_years = unique_years[4:]
        dev_idx = np.where(np.isin(years, train_years))[0]
        test_idx = np.where(np.isin(years, test_years))[0]
        split_desc = f"4-Year Dev ({unique_years[0]}-{unique_years[3]}) vs 5th-Year Test ({unique_years[4]})"
    else:
        # Chronological 80% dev / 20% test
        n_dev = int(round(n_total * 0.80))
        dev_idx = np.arange(n_dev)
        test_idx = np.arange(n_dev, n_total)
        split_desc = f"Chronological 80/20 Dev/Test ({n_dev:,} dev / {n_total - n_dev:,} test samples)"

    n_dev = len(dev_idx)
    n_train = int(round(n_dev * 0.90))
    train_idx = dev_idx[:n_train]
    val_idx = dev_idx[n_train:]

    # Optional cap for rapid debugging / testing
    effective_max = max_samples if max_samples is not None else getattr(cfg, "max_samples", None)
    if effective_max is not None and 0 < effective_max < len(train_idx):
        print(f"  [DEBUG LIMIT] Capping training to {effective_max:,} samples.")
        train_idx = train_idx[:effective_max]
        val_cap = max(64, int(effective_max * 0.11))
        val_idx = val_idx[:val_cap]
        test_idx = test_idx[:val_cap]

    d_tr_s = _get_sample_date(paths, fi_all[train_idx[0]], li_all[train_idx[0]])
    d_tr_e = _get_sample_date(paths, fi_all[train_idx[-1]], li_all[train_idx[-1]])
    d_va_s = _get_sample_date(paths, fi_all[val_idx[0]], li_all[val_idx[0]])
    d_va_e = _get_sample_date(paths, fi_all[val_idx[-1]], li_all[val_idx[-1]])
    d_te_s = _get_sample_date(paths, fi_all[test_idx[0]], li_all[test_idx[0]])
    d_te_e = _get_sample_date(paths, fi_all[test_idx[-1]], li_all[test_idx[-1]])

    print("\n" + "=" * 70)
    print("  DATASET PARTITIONING PROTOCOL")
    print(f"  Split Mode        : {split_desc}")
    print(f"  Total Samples     : {n_total:,}")
    print(f"  Training Split    : {len(train_idx):,}  [{d_tr_s} to {d_tr_e}]")
    print(f"  Validation Split  : {len(val_idx):,}  [{d_va_s} to {d_va_e}]")
    print(f"  Held-out Test     : {len(test_idx):,}  [{d_te_s} to {d_te_e}]")
    print("=" * 70 + "\n")


    # Normalization statistics
    if norm_stats is None:
        ns_path = os.path.join(getattr(cfg, "checkpoints_dir", "checkpoints"), "norm_stats.json")
        if os.path.exists(ns_path):
            import json
            with open(ns_path, "r") as f:
                data = json.load(f)
            norm_stats = {
                "mean": np.array(data["mean"], dtype=np.float32),
                "std": np.array(data["std"], dtype=np.float32),
            }
        else:
            norm_stats = compute_training_norm_stats(
                paths, fi_all[train_idx], li_all[train_idx], getattr(cfg, "n_surface_vars", 7)
            )

    train_ds = UpgradedOceanDataset(paths, fi_all[train_idx], li_all[train_idx], norm_stats=norm_stats, climatology=None)
    val_ds = UpgradedOceanDataset(paths, fi_all[val_idx], li_all[val_idx], norm_stats=norm_stats, climatology=None)
    test_ds = UpgradedOceanDataset(paths, fi_all[test_idx], li_all[test_idx], norm_stats=norm_stats, climatology=None)

    batch_size = getattr(cfg, "batch_size", 32)
    cpu_cores = os.cpu_count() or 4
    num_workers = min(cpu_cores, 8)
    pin_memory = getattr(cfg, "pin_memory", False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=(len(train_ds) >= batch_size)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=pin_memory, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=pin_memory, drop_last=False
    )

    return train_loader, val_loader, test_loader, norm_stats


