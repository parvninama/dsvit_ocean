"""
evaluate_glorys_upgraded.py - Full evaluation on untouched 10% GLORYS test partition.
Evaluates both anomaly and absolute temperature reconstruction, uncertainty calibration,
and performs failure and geographic analysis.
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
from typing import Dict

from upgraded.config import load_upgraded_config as load_config
from upgraded.model import UpgradedOceanReconstructionModel
from upgraded.dataset import get_upgraded_dataloaders
from upgraded.climatology import HarmonicClimatology
from upgraded.metrics import compute_upgraded_metrics, format_upgraded_metrics_table, DEPTH_LABELS
from upgraded.visualization import generate_all_upgraded_plots
from utils import get_device, load_checkpoint, load_norm_stats

@torch.no_grad()
def evaluate_glorys_test(
    model: UpgradedOceanReconstructionModel,
    climatology: HarmonicClimatology,
    loader,
    device: torch.device,
    results_dir: str,
    max_samples: int = None,
):
    model.eval()
    print("\n" + "="*70)
    print("  EVALUATING ON UNTOUCHED 10% HELD-OUT GLORYS TEST SET")
    print("="*70)

    all_abs_pred, all_anom_pred = [], []
    all_abs_tgt, all_anom_tgt = [], []
    all_clim, all_sigma, all_dmask = [], [], []
    all_recon, all_surf, all_smask = [], [], []
    all_lats, all_lons, all_dates = [], [], []

    n_samples = 0
    for batch_idx, batch in enumerate(loader):
        surf = batch["surface_data"].to(device)
        temp = batch["temperature"].to(device)
        lat = batch["latitude"].to(device)
        lon = batch["longitude"].to(device)
        seas = batch["seasonal_time"].to(device)
        smask = batch["surface_mask"].to(device)
        dmask = batch["depth_mask"].to(device)
        B = surf.shape[0]
        n_samples += B

        clim = climatology.lookup_torch(lat, lon, seas, device=device) if climatology is not None else None
        out = model(surf, lat, lon, seas, smask, clim)

        p_abs = out.absolute_temp.cpu().numpy()
        p_anom = out.anomaly_mean.cpu().numpy() if out.anomaly_mean is not None else p_abs
        t_abs = temp.cpu().numpy()
        c_np = clim.cpu().numpy() if clim is not None else np.zeros_like(t_abs)
        t_anom = t_abs - c_np
        d_var = out.diag_variance.cpu().numpy()
        l_fac = out.low_rank_factor.cpu().numpy()
        p_sigma = np.sqrt(d_var + np.sum(l_fac ** 2, axis=-1))

        all_abs_pred.append(p_abs)
        all_anom_pred.append(p_anom)
        all_abs_tgt.append(t_abs)
        all_anom_tgt.append(t_anom)
        all_clim.append(c_np)
        all_sigma.append(p_sigma)
        all_dmask.append(dmask.cpu().numpy())
        all_lats.append(lat.cpu().numpy())
        all_lons.append(lon.cpu().numpy())
        date_tensor = batch.get("date_ordinal", batch.get("date", None))
        all_dates.append(date_tensor.numpy() if date_tensor is not None else np.zeros(B, dtype=np.int32))

        if out.recon is not None:
            all_recon.append(out.recon.cpu().numpy())
            all_surf.append(surf.cpu().numpy())
            all_smask.append(smask.cpu().numpy())

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
            print(f"  Processed {n_samples:,} test samples...", flush=True)

        if max_samples is not None and n_samples >= max_samples:
            break

    pred_abs_np = np.concatenate(all_abs_pred)
    pred_anom_np = np.concatenate(all_anom_pred)
    target_abs_np = np.concatenate(all_abs_tgt)
    target_anom_np = np.concatenate(all_anom_tgt)
    clim_np = np.concatenate(all_clim)
    pred_sigma_np = np.concatenate(all_sigma)
    dmask_np = np.concatenate(all_dmask)
    lats_np = np.concatenate(all_lats)
    lons_np = np.concatenate(all_lons)
    dates_np = np.concatenate(all_dates)
    recon_np = np.concatenate(all_recon) if all_recon else None
    surf_np = np.concatenate(all_surf) if all_surf else None
    smask_np = np.concatenate(all_smask) if all_smask else None

    metrics = compute_upgraded_metrics(
        pred_abs=pred_abs_np,
        pred_anom=pred_anom_np,
        target_abs=target_abs_np,
        target_anom=target_anom_np,
        depth_mask=dmask_np,
        pred_sigma=pred_sigma_np,
        recon=recon_np,
        orig_surf=surf_np,
        surf_mask=smask_np,
    )

    print("\n" + "="*70)
    print("  GLORYS TEST EVALUATION METRICS")
    print("="*70)
    print(format_upgraded_metrics_table(metrics, prefix="  "))

    # Failure Analysis: Top 5 Highest Error Profiles
    profile_rmses = []
    for i in range(len(pred_abs_np)):
        m = dmask_np[i].astype(bool)
        if m.sum() > 0:
            profile_rmses.append(np.sqrt(np.mean((pred_abs_np[i, m] - target_abs_np[i, m]) ** 2)))
        else:
            profile_rmses.append(0.0)
    profile_rmses = np.array(profile_rmses)
    top_failures = np.argsort(profile_rmses)[::-1][:5]

    print("\n  === Failure Analysis: Top 5 Highest Error Profiles ===")
    for rank, idx in enumerate(top_failures, 1):
        m = dmask_np[idx].astype(bool)
        rmse_i = profile_rmses[idx]
        mae_i = np.mean(np.abs(pred_abs_np[idx, m] - target_abs_np[idx, m]))
        sig_i = np.mean(pred_sigma_np[idx, m])
        print(f"  #{rank} Index {idx:5d} | Lat={lats_np[idx]:.2f}°N Lon={lons_np[idx]:.2f}°E DateOrd={dates_np[idx]} | "
              f"ValidDepths={m.sum()}/15 | RMSE={rmse_i:.3f}°C | MAE={mae_i:.3f}°C | MeanSigma={sig_i:.3f}°C")

    # Save metrics JSON & plots
    test_res_dir = os.path.join(results_dir, "glorys_test")
    os.makedirs(test_res_dir, exist_ok=True)
    with open(os.path.join(test_res_dir, "glorys_test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    generate_all_upgraded_plots(
        pred_abs=pred_abs_np,
        target_abs=target_abs_np,
        climatology=clim_np,
        depth_mask=dmask_np,
        pred_sigma=pred_sigma_np,
        metrics=metrics,
        history=None,
        results_dir=test_res_dir,
    )
    print(f"\n  Diagnostic reports and plots saved to: {test_res_dir}")
    print("="*70 + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Upgraded Model on GLORYS Test Set")
    parser.add_argument("--checkpoint", type=str, default="saved_models/overnight_champion/model.pt")
    parser.add_argument("--config", type=str, default="saved_models/overnight_champion/config.yaml")
    parser.add_argument("--data", "--data-dir", dest="data", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    ckpt_raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "config" in ckpt_raw:
        cfg = ckpt_raw["config"]
    elif os.path.exists(args.config):
        cfg = load_config(args.config)
    else:
        cfg = load_config("saved_models/overnight_champion/config.yaml")

    if args.data:
        sub_samples = os.path.join(args.data, "samples_monthly")
        if os.path.exists(sub_samples):
            cfg.data_dir = sub_samples
        else:
            cfg.data_dir = args.data

    device_str = args.device if args.device != "auto" else getattr(cfg, "device", "auto")
    device = get_device(device_str)

    model = UpgradedOceanReconstructionModel(cfg).to(device)
    model.load_state_dict(ckpt_raw["model_state_dict"])
    epoch = ckpt_raw.get("epoch", 0)
    best_metric = ckpt_raw.get("best_metric", 0.0)
    norm_stats = ckpt_raw.get("norm_stats", None)
    training_mode = ckpt_raw.get("training_mode", "direct")
    model.set_training_mode(training_mode)
    model.eval()
    print(f"Loaded checkpoint from epoch {epoch} (best_val_rmse={best_metric:.4f}, mode={training_mode})")

    climatology = None
    if training_mode == "anomaly":
        clim_path = getattr(cfg, "climatology_path", "checkpoints/climatology.npz")
        if os.path.exists(clim_path):
            climatology = HarmonicClimatology.load(clim_path)

    _, _, test_loader, _ = get_upgraded_dataloaders(cfg, norm_stats=norm_stats, climatology=climatology)
    evaluate_glorys_test(model, climatology, test_loader, device, cfg.results_dir, max_samples=args.max_samples)

if __name__ == "__main__":
    main()
