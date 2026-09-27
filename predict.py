"""
predict.py - Standalone inference script for Ocean Subsurface Temperature Reconstruction.

Usage:
  python predict.py
  python predict.py --checkpoint saved_models/overnight_champion/model.pt
"""
from __future__ import annotations
import os
import sys
import argparse
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from upgraded.model import UpgradedOceanReconstructionModel

TARGET_DEPTHS_M = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


def load_model(checkpoint_path: str = "saved_models/overnight_champion/model.pt", device: torch.device = None):
    """Loads model weights, architecture config, and normalization statistics from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    norm_stats = ckpt.get("norm_stats", None)

    model = UpgradedOceanReconstructionModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.set_training_mode("direct")
    model.eval()

    return model, cfg, norm_stats, device


def predict_temperature(
    model: torch.nn.Module,
    surface_data: np.ndarray | torch.Tensor,
    latitude: float | torch.Tensor,
    longitude: float | torch.Tensor,
    day_of_year: int | float = 180,
    norm_stats: dict = None,
    device: torch.device = torch.device("cpu"),
):
    """
    Reconstructs 15-depth ocean temperature profile from 11x11 surface patch.

    Args:
        surface_data: [7, 11, 11] raw or normalized physical surface features
                      (SST, SSS, SLA, U_curr, V_curr, U_wind, V_wind)
        latitude: scalar float (degrees N)
        longitude: scalar float (degrees E)
        day_of_year: integer day of year (1-365)
        norm_stats: dict with 'mean' and 'std' for the 7 channels (if normalizing on the fly)
    Returns:
        temperatures: [15] predicted temperature values in °C
        uncertainties: [15] 1-sigma calibrated uncertainty in °C
    """
    model.eval()

    if isinstance(surface_data, np.ndarray):
        if norm_stats is not None and "mean" in norm_stats:
            mean = np.array(norm_stats["mean"], dtype=np.float32)[:, None, None]
            std = np.array(norm_stats["std"], dtype=np.float32)[:, None, None]
            surface_data = (surface_data - mean) / np.maximum(std, 1e-6)
        surf_t = torch.from_numpy(surface_data).float().unsqueeze(0).to(device)
    else:
        surf_t = surface_data.to(device)
        if surf_t.dim() == 3:
            surf_t = surf_t.unsqueeze(0)

    lat_t = torch.tensor([float(latitude)], dtype=torch.float32, device=device)
    lon_t = torch.tensor([float(longitude)], dtype=torch.float32, device=device)

    # Seasonal time in [0, 1)
    frac = float(day_of_year) / 365.25
    seas_t = torch.tensor([frac], dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model(surf_t, lat_t, lon_t, seas_t)
        pred_temp = out.absolute_temp.squeeze(0).cpu().numpy()
        uncertainty = torch.sqrt(out.diag_variance).squeeze(0).cpu().numpy()

    return pred_temp, uncertainty


def main():
    parser = argparse.ArgumentParser(description="Ocean Subsurface Temperature Inference")
    parser.add_argument("--checkpoint", type=str, default="saved_models/overnight_champion/model.pt")
    parser.add_argument("--lat", type=float, default=15.0, help="Latitude (°N)")
    parser.add_argument("--lon", type=float, default=85.0, help="Longitude (°E)")
    parser.add_argument("--day", type=int, default=180, help="Day of year (1-365)")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  OCEAN SUBSURFACE TEMPERATURE RECONSTRUCTION (INFERENCE)")
    print("=" * 65)

    model, cfg, norm_stats, device = load_model(args.checkpoint)
    print(f"Device: {device} | Loaded: {args.checkpoint}")
    print(f"Location: {args.lat:.2f}°N, {args.lon:.2f}°E | Day of Year: {args.day}")

    # Generate sample 11x11 patch (7 channels: SST, SSS, SLA, U_curr, V_curr, U_wind, V_wind)
    sample_patch = np.zeros((7, 11, 11), dtype=np.float32)
    sample_patch[0] = 301.5  # ~28.35 °C (SST in Kelvin)
    sample_patch[1] = 32.5   # Salinity (PSU)
    sample_patch[2] = 0.05   # Sea level anomaly (m)

    temps, uncs = predict_temperature(
        model, sample_patch, args.lat, args.lon, day_of_year=args.day, norm_stats=norm_stats, device=device
    )

    print("\n" + "-" * 65)
    print(f"{'Depth':>8} | {'Predicted Temp':>16} | {'1-Sigma Uncertainty':>20}")
    print("-" * 65)
    for d, t, u in zip(TARGET_DEPTHS_M, temps, uncs):
        print(f"{d:>7}m | {t:>14.2f} °C | {u:>18.2f} °C")
    print("-" * 65 + "\n")


if __name__ == "__main__":
    main()
