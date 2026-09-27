"""
evaluate_argo_upgraded.py - Independent In-Situ Argo Float Evaluation for Upgraded Ocean Model.

Scientific protocol:
  - Strictly independent evaluation: Argo data was NEVER used for training, normalization, or tuning.
  - Each Argo profile is collocated with the corresponding 0.25° daily surface patch.
  - Normalized surface data + geographic (lat/lon) + seasonal time passed to frozen model.
  - Reconstructs 15 absolute temperature predictions and correlated uncertainty.
  - Comprehensive statistical metrics, calibration analysis, depth error profiling, and report generation.
"""
from __future__ import annotations
import os
import sys
import argparse
import time
import json
import datetime
import numpy as np
import pandas as pd
import torch
import yaml
from typing import Dict, List, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from upgraded.model import UpgradedOceanReconstructionModel
from upgraded.climatology import HarmonicClimatology
from upgraded.config import UpgradedConfig, load_upgraded_config
from argo_loader import (
    load_argo_profiles, DEFAULT_TARGET_DEPTHS, SurfaceDataLoader, collocate_profile, RejectionTracker
)
from argo_metrics import compute_argo_evaluation_metrics, format_argo_metrics_table, export_argo_metrics, TARGET_DEPTHS_M, DEPTH_LABELS
from diagnostics import rank_profile_failures, compare_with_glorys, generate_evaluation_report_md
from visualization import generate_all_argo_plots
from utils import get_device, load_checkpoint


def parse_argo_time_to_seasonal_time(time_str: str) -> float:
    """Parses Argo ISO/datetime string into normalized seasonal time [0, 1)."""
    try:
        dt = pd.to_datetime(time_str)
        doy = dt.dayofyear
        return float((doy - 1.0) / 365.25)
    except Exception:
        return 0.5


def run_argo_evaluation_upgraded(
    config_path: str,
    checkpoint_path: str,
    device_str: str = "auto",
    max_profiles: Optional[int] = None,
    output_dir: Optional[str] = "results/argo_test",
    argo_dir: Optional[str] = None,
):
    print("\n" + "="*70)
    print("  INDEPENDENT ARGO EVALUATION (UPGRADED INCEPTION-VIT MODEL)")
    print("="*70)

    ckpt_raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" in ckpt_raw:
        cfg = ckpt_raw["config"]
    elif os.path.exists(config_path):
        cfg = load_upgraded_config(config_path)
    else:
        cfg = load_upgraded_config("saved_models/overnight_champion/config.yaml")

    if hasattr(cfg, "to_dict"):
        cfg_dict = cfg.to_dict()
    elif isinstance(cfg, dict):
        cfg_dict = cfg
    else:
        cfg_dict = getattr(cfg, "__dict__", {})

    device = get_device(device_str if device_str != "auto" else getattr(cfg, "device", "auto"))
    print(f"Device: {device}")

    model = UpgradedOceanReconstructionModel(cfg).to(device)
    model.load_state_dict(ckpt_raw["model_state_dict"])
    epoch = ckpt_raw.get("epoch", 0)
    best_val = ckpt_raw.get("best_metric", 0.0)
    norm_stats = ckpt_raw.get("norm_stats", None)
    training_mode = ckpt_raw.get("training_mode", "direct")
    model.set_training_mode(training_mode)
    model.eval()

    # Load Climatology only if needed (for anomaly mode)
    climatology = None
    if training_mode == "anomaly":
        clim_path = cfg_dict.get("climatology_path", "checkpoints/climatology.npz")
        climatology = HarmonicClimatology.load(clim_path)
        print(f"Loaded frozen climatology from: {clim_path}")
    else:
        print("Model is in 'direct' temperature mode: Fourier climatology path disabled.")

    # 2. Discover and Load Argo Profiles
    if argo_dir is None:
        argo_dir = cfg_dict.get("argo_dir", "argo_test_bob_2years")
    regrid_dir = cfg_dict.get("regrid_dir", "/Users/parvninama/Programs/SIH/dsvit_ocean/bob_ocean_dataset_2years/regridded_monthly")
    
    t0 = time.time()
    raw_profiles = load_argo_profiles(argo_dir, max_profiles=max_profiles)
    print(f"[evaluate_argo] Loaded {len(raw_profiles):,} raw Argo profiles in {time.time()-t0:.1f}s")

    # 3. Collocation
    surface_loader = SurfaceDataLoader(regrid_dir, norm_stats)
    tracker = RejectionTracker()

    print("[evaluate_argo] Collocating profiles with 0.25° surface observations...")
    matched_records = []
    t_colloc = time.time()
    for idx, p in enumerate(raw_profiles):
        rec = collocate_profile(
            p, surface_loader,
            max_spatial_distance_km=cfg_dict.get("max_spatial_dist_km", 30.0),
            max_temporal_difference_hours=cfg_dict.get("max_temporal_diff_hours", 36.0),
            min_valid_depths=1,
            tracker=tracker
        )
        if rec is not None:
            matched_records.append(rec)
        if (idx + 1) % 500 == 0 or (idx + 1) == len(raw_profiles):
            print(f"  Processed {idx+1}/{len(raw_profiles)} profiles ({len(matched_records)} matched)...")

    surface_loader.close()
    print(f"[evaluate_argo] Collocation complete ({len(matched_records):,} matched) in {time.time()-t_colloc:.1f}s")
    tracker.print_summary()

    if not matched_records:
        print("[evaluate_argo] ERROR: No profiles matched.")
        return

    # 4. Inference Batching
    print(f"\n[evaluate_argo] Running upgraded model inference on {len(matched_records):,} matched profiles...")
    t_inf = time.time()
    all_pred_mu = []
    all_pred_sigma = []
    batch_size = 64
    N = len(matched_records)

    with torch.no_grad():
        for b_start in range(0, N, batch_size):
            b_end = min(N, b_start + batch_size)
            b_records = matched_records[b_start:b_end]

            surf_b = torch.from_numpy(np.stack([r["surface_data"] for r in b_records])).to(device)
            lat_b = torch.tensor([r["latitude"] for r in b_records], dtype=torch.float32).to(device)
            lon_b = torch.tensor([r["longitude"] for r in b_records], dtype=torch.float32).to(device)
            smask_b = torch.from_numpy(np.stack([r["surface_mask"] for r in b_records])).to(device)
            
            seas_vals = [parse_argo_time_to_seasonal_time(str(r["time"])) for r in b_records]
            seas_b = torch.tensor(seas_vals, dtype=torch.float32).to(device)

            if training_mode == "anomaly" and climatology is not None:
                clim_b = climatology.lookup_torch(lat_b, lon_b, seas_b, device=device)
            else:
                clim_b = None
            out = model(surf_b, lat_b, lon_b, seas_b, smask_b, clim_b)

            p_abs = out.absolute_temp.cpu().numpy()
            d_var = out.diag_variance.cpu().numpy()
            l_fac = out.low_rank_factor.cpu().numpy()
            p_sigma = np.sqrt(d_var + np.sum(l_fac ** 2, axis=-1))

            all_pred_mu.append(p_abs)
            all_pred_sigma.append(p_sigma)

    pred_mu = np.concatenate(all_pred_mu, axis=0)
    pred_sigma = np.concatenate(all_pred_sigma, axis=0)

    # Attach predictions back to records
    for idx, r in enumerate(matched_records):
        r["predicted_temperature"] = pred_mu[idx]
        r["predicted_uncertainty"] = pred_sigma[idx]

    argo_targets = np.stack([r["argo_temperature"] for r in matched_records])
    depth_masks = np.stack([r["depth_mask"] for r in matched_records])

    # 5. Compute Metrics
    metrics = compute_argo_evaluation_metrics(pred_mu, pred_sigma, argo_targets, depth_masks)
    print("\n" + "="*70)
    print("  UPGRADED MODEL VS. INDEPENDENT ARGO IN-SITU OBSERVATIONS")
    print("="*70)
    print(format_argo_metrics_table(metrics, prefix="  "))

    # 6. Export Results & Diagnostic Plots
    os.makedirs(output_dir, exist_ok=True)
    export_argo_metrics(metrics, output_dir)

    failure_records = rank_profile_failures(matched_records, top_k=20)
    with open(os.path.join(output_dir, "failure_profiles.json"), "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_failures = []
        for fr in failure_records:
            sf = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in fr.items()}
            serializable_failures.append(sf)
        json.dump(serializable_failures, f, indent=2)

    glorys_comp = compare_with_glorys(matched_records)
    with open(os.path.join(output_dir, "glorys_comparison.json"), "w") as f:
        json.dump(glorys_comp, f, indent=2)

    cfg.depth_map_levels = [0, 75, 100, 300, 500]
    cfg.num_profile_plots = 6
    cfg.max_spatial_distance_km = cfg_dict.get("max_spatial_dist_km", 30.0)
    cfg.max_temporal_difference_hours = cfg_dict.get("max_temporal_diff_hours", 36.0)

    plots_dir = os.path.join(output_dir, "plots")
    generate_all_argo_plots(metrics, matched_records, failure_records, plots_dir, cfg)

    rejection_summary = tracker.get_summary()
    report_path = os.path.join(output_dir, "argo_evaluation_report.md")
    generate_evaluation_report_md(
        metrics=metrics,
        rejection_summary=rejection_summary,
        top_failures=failure_records,
        glorys_comp=glorys_comp,
        output_path=report_path,
        cfg=cfg,
    )

    print(f"\n  Argo evaluation report generated: {report_path}")
    print(f"  Diagnostic plots saved to: {plots_dir}")
    print("="*70 + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Upgraded Ocean Model Against Argo Floats")
    parser.add_argument("--checkpoint", type=str, default="saved_models/overnight_champion/model.pt")
    parser.add_argument("--config", type=str, default="saved_models/overnight_champion/config.yaml")
    parser.add_argument("--data", "--argo-dir", dest="data", type=str, default=None)
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results/argo_test")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_argo_evaluation_upgraded(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device_str=args.device,
        max_profiles=args.max_profiles,
        output_dir=args.output_dir,
        argo_dir=args.data,
    )

if __name__ == "__main__":
    main()
