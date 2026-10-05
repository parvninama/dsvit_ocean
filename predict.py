"""
predict.py - Standalone inference script for Ocean Subsurface Temperature Reconstruction.

Usage:
  python predict.py
  python predict.py --checkpoint saved_models/best_model/model.pt
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


def load_model(checkpoint_path: str = "saved_models/best_model/model.pt", device: torch.device = None):
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


def plot_profile(depths: list, temps: np.ndarray, uncs: np.ndarray, lat: float, lon: float, day: int, output_path: str = "predicted_profile.png"):
    """Plots and saves the reconstructed ocean vertical temperature curve with uncertainty shading."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 8), dpi=120)
    ax.plot(temps, depths, "o-", color="#0055aa", linewidth=2.2, markersize=5.5, label=r"Predicted $T(z)$")
    ax.fill_betweenx(depths, temps - uncs, temps + uncs, color="#0055aa", alpha=0.22, label=r"Calibrated $1\sigma$ Uncertainty")

    # Vertical axis: depth decreases from up (1000m) to down (0m)
    ax.set_ylim(-20, max(depths) + 50)
    ax.set_xlabel("Temperature (°C)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Depth (m)", fontsize=11, fontweight="bold")
    ax.set_title(f"Reconstructed Ocean Thermal Profile\nLocation: {lat:.2f}°N, {lon:.2f}°E | Day of Year: {day}", fontsize=11, pad=12)

    # Highlight thermocline bottleneck layer
    ax.axhspan(50, 150, color="#ff9900", alpha=0.15, label="Thermocline Layer (50–150m)")

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [Plot] Plotted temperature profile curve saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Ocean Subsurface Temperature Inference")
    parser.add_argument("--checkpoint", type=str, default="saved_models/best_model/model.pt")
    parser.add_argument("--lat", type=float, default=15.0, help="Latitude (°N)")
    parser.add_argument("--lon", type=float, default=85.0, help="Longitude (°E)")
    parser.add_argument("--day", type=int, default=180, help="Day of year (1-365)")
    parser.add_argument("--plot", type=str, default="predicted_profile.png", help="Filename to save the plotted temperature curve (default: predicted_profile.png)")
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

    if args.plot and args.plot.lower() not in ("none", "false", "0", ""):
        plot_profile(TARGET_DEPTHS_M, temps, uncs, args.lat, args.lon, args.day, output_path=args.plot)


if __name__ == "__main__":
    main()
