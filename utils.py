import os
import random
import json
import csv
import time
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto"):
    if device_str in ("dml", "directml"):
        try:
            import torch_directml
            if torch_directml.is_available():
                dev = torch_directml.device()
                name = torch_directml.device_name(0)
                print(f"  GPU Backend: DirectML ({name.strip()})")
                return dev
        except ImportError:
            pass

    if device_str == "auto":
        if torch.cuda.is_available():
            print(f"  GPU Backend: CUDA ({torch.cuda.get_device_name(0)})")
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  GPU Backend: Apple Silicon (MPS)")
            return torch.device("mps")
        try:
            import torch_directml
            if torch_directml.is_available():
                dev = torch_directml.device()
                name = torch_directml.device_name(0)
                print(f"  GPU Backend: DirectML ({name.strip()})")
                return dev
        except ImportError:
            pass
        return torch.device("cpu")

    return torch.device(device_str)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_complexity_report(model: torch.nn.Module):
    """
    Computes and prints a detailed breakdown of model parameters across all 8 submodules.
    """
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_train_params = total_params - trainable_params

    def _module_params(mod: Optional[torch.nn.Module]) -> Tuple[int, int]:
        if mod is None:
            return 0, 0
        tot = sum(p.numel() for p in mod.parameters())
        trn = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        return tot, trn

    components = []

    # 1. Spatial Encoder (CNN)
    cnn_tot, cnn_trn = _module_params(getattr(model, "cnn_encoder", None))
    components.append(("1. Spatial Encoder (CNN)", cnn_tot, cnn_trn))

    # 2. Positional, Geographic & Seasonal Embeddings
    pos_tot, pos_trn = _module_params(getattr(model, "pos_enc", None))
    geo_tot, geo_trn = _module_params(getattr(model, "geo_enc", None))
    seas_tot, seas_trn = _module_params(getattr(model, "seasonal_enc", None))
    components.append(("2. Positional, Geographic & Seasonal Embeddings", pos_tot + geo_tot + seas_tot, pos_trn + geo_trn + seas_trn))

    # 3. Vision Transformer Encoder (Spatial ViT + LayerNorm)
    sp_vit_tot, sp_vit_trn = _module_params(getattr(model, "spatial_transformer", None))
    sp_nm_tot, sp_nm_trn   = _module_params(getattr(model, "spatial_norm", None))
    components.append(("3. Vision Transformer Encoder", sp_vit_tot + sp_nm_tot, sp_vit_trn + sp_nm_trn))

    # 4. Cross-Attention / Center Token Representation
    components.append(("4. Cross-Attention Module (Center Token)", 0, 0))

    # 5. Depth Query Embeddings
    dep_tot, dep_trn = _module_params(getattr(model, "depth_encoder", None))
    components.append(("5. Depth Query Embeddings", dep_tot, dep_trn))

    # 6. Vertical Attention / Transformer
    vt_tot, vt_trn = _module_params(getattr(model, "vertical_transformer", None))
    vt_nm_tot, vt_nm_trn = _module_params(getattr(model, "vertical_norm", None))
    components.append(("6. Vertical Attention / Transformer", vt_tot + vt_nm_tot, vt_trn + vt_nm_trn))

    # 7. Temperature Prediction Head (Direct or Anomaly)
    th_tot, th_trn = _module_params(getattr(model, "temp_head", None))
    ah_tot, ah_trn = _module_params(getattr(model, "anomaly_head", None))
    components.append(("7. Temperature / Anomaly Head", th_tot + ah_tot, th_trn + ah_trn))

    # 8. Uncertainty Head (Direct or Correlated Diag+LowRank)
    unc_tot, unc_trn = _module_params(getattr(model, "unc_head", None))
    diag_tot, diag_trn = _module_params(getattr(model, "diag_var_head", None))
    lr_tot, lr_trn = _module_params(getattr(model, "low_rank_head", None))
    components.append(("8. Uncertainty Head", unc_tot + diag_tot + lr_tot, unc_trn + diag_trn + lr_trn))

    # Surface Reconstruction Decoder (if present)
    dec_tot, dec_trn = _module_params(getattr(model, "cnn_decoder", None))
    if dec_tot > 0:
        components.append(("   Surface Reconstruction Decoder", dec_tot, dec_trn))

    print("\n" + "="*75)
    print("  MODEL ARCHITECTURE & COMPLEXITY REPORT")
    print("="*75)
    print(f"  {'Submodule Component':<42} {'Parameters':>12} {'Trainable':>12} {'% Total':>8}")
    print("  " + "-"*72)
    for name, tot, trn in components:
        pct = (tot / max(total_params, 1)) * 100.0
        print(f"  {name:<42} {tot:>12,d} {trn:>12,d} {pct:>7.2f}%")
    print("  " + "-"*72)
    print(f"  {'TOTAL MODEL PARAMETERS':<42} {total_params:>12,d} {trainable_params:>12,d} {100.0:>7.2f}%")
    print(f"  {'NON-TRAINABLE PARAMETERS':<42} {non_train_params:>12,d}")
    print("="*75 + "\n")


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    cfg,
    norm_stats: Dict,
    is_best: bool = False,
    best_val_rmse: float = float("inf"),
    global_step: int = 0,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_metric": best_val_rmse,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
        "norm_stats": {k: v.tolist() for k, v in norm_stats.items()} if norm_stats is not None else None,
        "is_best": is_best,
    }
    torch.save(state, path)
    print(f"  Checkpoint saved: {path} (epoch={epoch}, best_val_rmse={best_val_rmse:.4f})")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device=None):
    if device is None:
        device = torch.device("cpu")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and state.get("optimizer_state_dict"):
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict"):
        scheduler.load_state_dict(state["scheduler_state_dict"])
    epoch = state.get("epoch", 0)
    
    # Robustly resolve best_val_rmse from all supported checkpoint schemas
    best_val_rmse = state.get("best_validation_metric")
    if best_val_rmse is None or best_val_rmse == float("inf"):
        best_val_rmse = state.get("best_val_rmse")
    if best_val_rmse is None or best_val_rmse == float("inf"):
        val_m = state.get("val_metrics", {})
        if isinstance(val_m, dict) and "rmse_c" in val_m:
            best_val_rmse = val_m["rmse_c"]
    if best_val_rmse is None or best_val_rmse == float("inf"):
        m = state.get("metrics", {})
        if isinstance(m, dict):
            best_val_rmse = m.get("overall_abs_rmse", m.get("rmse", float("inf")))
    if best_val_rmse is None:
        best_val_rmse = float("inf")

    global_step = state.get("global_step", 0)
    metrics = state.get("val_metrics", state.get("metrics", {}))
    norm_stats_raw = state.get("norm_stats", {})
    norm_stats = {k: np.array(v, dtype=np.float32) for k, v in norm_stats_raw.items()} if norm_stats_raw else {}
    print(f"  Loaded checkpoint from epoch {epoch} (best_val_rmse={best_val_rmse:.4f}): {path}")
    return epoch, best_val_rmse, global_step, metrics, norm_stats


def save_norm_stats(norm_stats: Dict, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    data = {k: v.tolist() for k, v in norm_stats.items()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Norm stats saved: {path}")


def load_norm_stats(path: str) -> Dict:
    with open(path, "r") as f:
        data = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in data.items()}


class CSVLogger:
    """Append one row per epoch to a CSV log file with resume support."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self._initialized = os.path.exists(path) and os.path.getsize(path) > 0
        self._fieldnames = None
        if self._initialized:
            with open(self.path, "r") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    self._fieldnames = header

    def log(self, row: Dict):
        if not self._initialized:
            self._fieldnames = list(row.keys())
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()
            self._initialized = True
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            writer.writerow(row)


class EarlyStopping:
    """Stop training when validation metric does not improve."""

    def __init__(self, patience: int = 10, mode: str = "min", min_delta: float = 1e-4):
        self.patience  = patience
        self.mode      = mode
        self.min_delta = min_delta
        self.best      = float("inf") if mode == "min" else float("-inf")
        self.counter   = 0
        self.best_epoch = 0

    def __call__(self, value: float, epoch: int) -> bool:
        improved = (
            (self.mode == "min" and value < self.best - self.min_delta) or
            (self.mode == "max" and value > self.best + self.min_delta)
        )
        if improved:
            self.best  = value
            self.counter = 0
            self.best_epoch = epoch
            return False   # do NOT stop
        else:
            self.counter += 1
            return self.counter >= self.patience   # stop

    def status(self) -> str:
        return f"patience {self.counter}/{self.patience}  best={self.best:.4f} @ epoch {self.best_epoch}"


class Timer:
    def __init__(self):
        self._start = time.time()

    def elapsed(self) -> float:
        return time.time() - self._start

    def reset(self):
        self._start = time.time()

    def elapsed_str(self) -> str:
        s = self.elapsed()
        return f"{int(s//60)}m {s%60:.1f}s"
