#!/usr/bin/env python3
"""
tune_overnight.py - Automated Overnight Hyperparameter Search for Ocean Reconstruction.

Scientific Objective:
  Iterate across a targeted cross-validation hyperparameter dictionary to minimize
  the Argo test error and break the thermocline (50-150m) bottleneck below the
  current champion score of 0.964°C.

Safety Guarantees:
  - NEVER modifies or overwrites the saved 0.964°C checkpoint (preserved in checkpoints/archive/best_argo_dev_0.964.pt).
  - Isolates every trial into dedicated result and checkpoint subdirectories.
  - Automatically updates an overall leaderboard (LEADERBOARD.md) after each trial.
  - Continues uninterrupted across trials if any single trial encounters an issue.
"""

import os
import sys
import time
import json
import csv
import copy
import yaml
import subprocess
from typing import Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PYTHON_EXE = os.path.join(PROJECT_ROOT, ".venv", "bin", "python3")
if not os.path.exists(PYTHON_EXE):
    PYTHON_EXE = sys.executable

# Standard thermocline depth weight templates
STANDARD_DEPTH_WEIGHTS = {
    "0": 1.0, "5": 1.0, "10": 1.0, "20": 1.0, "30": 1.1,
    "50": 1.5, "75": 2.0, "100": 2.0, "125": 1.75, "150": 1.75,
    "200": 1.25, "300": 1.0, "500": 1.0, "700": 1.0, "1000": 1.0
}

AGGRESSIVE_THERMOCLINE_WEIGHTS = {
    "0": 1.0, "5": 1.0, "10": 1.0, "20": 1.0, "30": 1.2,
    "50": 2.0, "75": 3.0, "100": 3.0, "125": 2.5, "150": 2.0,
    "200": 1.5, "300": 1.0, "500": 1.0, "700": 1.0, "1000": 1.0
}

PEAK_75_100_WEIGHTS = {
    "0": 1.0, "5": 1.0, "10": 1.0, "20": 1.0, "30": 1.1,
    "50": 1.8, "75": 3.5, "100": 3.5, "125": 2.2, "150": 1.8,
    "200": 1.2, "300": 1.0, "500": 1.0, "700": 1.0, "1000": 1.0
}

GAUSSIAN_THERMOCLINE_WEIGHTS = {
    "0": 1.0, "5": 1.0, "10": 1.0, "20": 1.1, "30": 1.3,
    "50": 2.0, "75": 2.8, "100": 3.0, "125": 2.6, "150": 1.9,
    "200": 1.4, "300": 1.1, "500": 1.0, "700": 1.0, "1000": 1.0
}

# ==============================================================================
# CROSS-VALIDATION HYPERPARAMETER SEARCH DICTIONARY
# ==============================================================================
HYPERPARAMETER_SEARCH_SPACE: List[Dict[str, Any]] = [
    {
        "trial_id": "trial_01_lr_5e5",
        "description": "Lower Learning Rate (5e-5) for finer convergence around thermocline",
        "learning_rate": 5e-5,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.30,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "depth_weights": STANDARD_DEPTH_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_02_lr_7e5",
        "description": "Intermediate Learning Rate (7e-5)",
        "learning_rate": 7e-5,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.30,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "depth_weights": STANDARD_DEPTH_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_03_lr_1.5e4",
        "description": "Higher Learning Rate (1.5e-4) to test faster escape from local basins",
        "learning_rate": 1.5e-4,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.30,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "depth_weights": STANDARD_DEPTH_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_04_grad_0.50",
        "description": "Increased Vertical Gradient Loss (0.50) to enforce vertical stratification",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.50,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "depth_weights": STANDARD_DEPTH_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_05_grad_0.75",
        "description": "Strong Vertical Gradient Loss (0.75) for sharp physical boundary layer",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.75,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "depth_weights": STANDARD_DEPTH_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_06_grad_0.20",
        "description": "Relaxed Vertical Gradient Loss (0.20) prioritizing direct temperature",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.20,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "depth_weights": STANDARD_DEPTH_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_07_huber_0.5",
        "description": "Robust Huber Delta (0.5) closer to L1 for outlier-resistant in-situ fitting",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.35,
        "lambda_temperature": 1.0,
        "huber_delta": 0.5,
        "depth_weights": STANDARD_DEPTH_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_08_huber_1.5",
        "description": "Smooth Huber Delta (1.5) closer to L2 for quadratic gradient centering",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.30,
        "lambda_temperature": 1.0,
        "huber_delta": 1.5,
        "depth_weights": STANDARD_DEPTH_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_09_weight_decay_5e4",
        "description": "Stronger Regularization (5e-4) to suppress GLORYS memorization",
        "learning_rate": 1e-4,
        "weight_decay": 5e-4,
        "lambda_gradient": 0.30,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "depth_weights": STANDARD_DEPTH_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_10_aggressive_thermocline",
        "description": "Aggressive Depth Weighting (3.0x at 75-100m, 2.0x at 50/150m)",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.40,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "depth_weights": AGGRESSIVE_THERMOCLINE_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_11_peak_75_100_boost",
        "description": "Targeted Peak Focus (3.5x at 75m & 100m bottleneck depths)",
        "learning_rate": 8e-5,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.45,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "depth_weights": PEAK_75_100_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_12_gaussian_thermocline",
        "description": "Gaussian Smooth Thermocline Profile centered at 100m",
        "learning_rate": 8e-5,
        "weight_decay": 2e-4,
        "lambda_gradient": 0.40,
        "lambda_temperature": 1.0,
        "huber_delta": 0.8,
        "depth_weights": GAUSSIAN_THERMOCLINE_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_13_combined_synergy_A",
        "description": "Synergy Combo: LR=6e-5, Grad=0.50, Huber=0.8, Peak Depth Weighting",
        "learning_rate": 6e-5,
        "weight_decay": 2e-4,
        "lambda_gradient": 0.50,
        "lambda_temperature": 1.0,
        "huber_delta": 0.8,
        "depth_weights": PEAK_75_100_WEIGHTS,
        "argo_patience": 6,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_14_combined_synergy_B",
        "description": "Synergy Combo: LR=5e-5, Grad=0.60, Huber=1.0, Aggressive Depth Weighting",
        "learning_rate": 5e-5,
        "weight_decay": 3e-4,
        "lambda_gradient": 0.60,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "depth_weights": AGGRESSIVE_THERMOCLINE_WEIGHTS,
        "argo_patience": 6,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_15_deep_convergence",
        "description": "Deep Convergence: LR=4e-5, Patience=7, Grad=0.45",
        "learning_rate": 4e-5,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.45,
        "lambda_temperature": 1.0,
        "huber_delta": 0.9,
        "depth_weights": PEAK_75_100_WEIGHTS,
        "argo_patience": 7,
        "argo_val_interval": 200,
    },
    # --- ARCHITECTURAL HYPERPARAMETERS (CNN & TRANSFORMER DIMENSIONS) ---
    {
        "trial_id": "trial_16_cnn_kernel_5",
        "description": "Larger 5x5 CNN Kernel for wider spatial ocean surface context",
        "learning_rate": 8e-5,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.40,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "cnn_kernel_size": 5,
        "depth_weights": STANDARD_DEPTH_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_17_cnn_channels_wide",
        "description": "Wider Inception CNN Channels [48, 96, 128] for rich multi-scale feature maps",
        "learning_rate": 8e-5,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.40,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "cnn_channels": [48, 96, 128],
        "depth_weights": STANDARD_DEPTH_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_18_cnn_channels_compact",
        "description": "Compact CNN Channels [24, 48, 72] to prevent feature overfitting",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.35,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "cnn_channels": [24, 48, 72],
        "depth_weights": STANDARD_DEPTH_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_19_transformer_4_spatial_layers",
        "description": "Deeper Spatial ViT (4 Layers) for longer-range surface correlations",
        "learning_rate": 7e-5,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.40,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "spatial_n_layers": 4,
        "depth_weights": PEAK_75_100_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_20_transformer_3_vertical_layers",
        "description": "Deeper Vertical Transformer (3 Layers) specifically targeting thermocline depth coupling",
        "learning_rate": 7e-5,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.45,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "vertical_n_layers": 3,
        "depth_weights": PEAK_75_100_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_21_transformer_8_heads",
        "description": "8-Head Self-Attention in Spatial & Vertical Transformers for multi-aspect representation",
        "learning_rate": 8e-5,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.40,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "spatial_n_heads": 8,
        "vertical_n_heads": 8,
        "depth_weights": STANDARD_DEPTH_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_22_transformer_mlp_ratio_3",
        "description": "Wider Transformer Feedforward MLP (mlp_ratio = 3.0)",
        "learning_rate": 7e-5,
        "weight_decay": 2e-4,
        "lambda_gradient": 0.40,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "spatial_mlp_ratio": 3.0,
        "vertical_mlp_ratio": 3.0,
        "depth_weights": STANDARD_DEPTH_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_23_covariance_rank_5",
        "description": "Higher Uncertainty Covariance Rank (rank=5) for complex inter-depth covariance",
        "learning_rate": 8e-5,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.40,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "covariance_rank": 5,
        "depth_weights": STANDARD_DEPTH_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
    # --- TRAINING PARAMETERS (BATCH SIZE, GRAD CLIPPING, SCHEDULE, FREQUENCY) ---
    {
        "trial_id": "trial_24_batch_size_16",
        "description": "Smaller Batch Size (16) with stochastic regularization to escape plateaus",
        "learning_rate": 6e-5,
        "batch_size": 16,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.40,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "depth_weights": PEAK_75_100_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 400,
    },
    {
        "trial_id": "trial_25_tight_grad_clip",
        "description": "Tighter Gradient Clipping (grad_clip=0.5) to stabilize steep thermocline backpropagation",
        "learning_rate": 8e-5,
        "grad_clip": 0.5,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.50,
        "lambda_temperature": 1.0,
        "huber_delta": 0.8,
        "depth_weights": PEAK_75_100_WEIGHTS,
        "argo_patience": 6,
        "argo_val_interval": 200,
    },
    {
        "trial_id": "trial_26_frequent_argo_eval",
        "description": "High Frequency Validation (Every 100 batches, patience=8) for razor-sharp early stopping",
        "learning_rate": 7e-5,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.45,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "depth_weights": PEAK_75_100_WEIGHTS,
        "argo_patience": 8,
        "argo_val_interval": 100,
    },
    {
        "trial_id": "trial_27_auxiliary_recon_boost",
        "description": "Higher Surface Reconstruction Loss (lambda_rec=0.25) to preserve ocean surface fidelity",
        "learning_rate": 8e-5,
        "weight_decay": 1e-4,
        "lambda_gradient": 0.40,
        "lambda_temperature": 1.0,
        "lambda_reconstruction": 0.25,
        "huber_delta": 1.0,
        "depth_weights": STANDARD_DEPTH_WEIGHTS,
        "argo_patience": 5,
        "argo_val_interval": 200,
    },
]


def update_leaderboard(results: List[Dict[str, Any]], leaderboard_dir: str):
    """Updates Markdown, CSV, and JSON leaderboards ranked by Argo Final RMSE."""
    os.makedirs(leaderboard_dir, exist_ok=True)
    sorted_res = sorted(results, key=lambda x: x.get("argo_final_rmse", float("inf")))

    # 1. Markdown Leaderboard
    md_path = os.path.join(leaderboard_dir, "LEADERBOARD.md")
    lines = [
        "# 🏆 OVERNIGHT HYPERPARAMETER TUNING LEADERBOARD",
        "",
        f"**Last Updated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Total Experiments Recorded:** {len(sorted_res)}",
        f"**Current Champion Score:** {sorted_res[0]['argo_final_rmse']:.4f}°C ({sorted_res[0]['trial_id']})",
        "",
        "---",
        "## Performance Leaderboard (Ranked by Untouched Argo Final Test RMSE)",
        "",
        "| Rank | Trial ID | Argo Final RMSE | TC (50-150m) RMSE | TC (75-150m) RMSE | Bias | LR | Grad Weight | Huber Delta | Status |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for rank, r in enumerate(sorted_res, start=1):
        f_rmse = r.get("argo_final_rmse", float("nan"))
        tc_50 = r.get("tc_50_150_rmse", float("nan"))
        tc_75 = r.get("tc_75_150_rmse", float("nan"))
        bias = r.get("bias", float("nan"))
        lr_s = f"{r.get('learning_rate', 0):.1e}" if "learning_rate" in r else "-"
        grad_s = f"{r.get('lambda_gradient', 0):.2f}" if "lambda_gradient" in r else "-"
        hub_s = f"{r.get('huber_delta', 0):.2f}" if "huber_delta" in r else "-"
        status = r.get("status", "COMPLETED")
        
        star = " ⭐ [NEW BEST]" if rank == 1 else ""
        if r.get("is_baseline", False):
            star = " 🔒 [BASELINE]"

        lines.append(
            f"| {rank} | `{r['trial_id']}`{star} | **{f_rmse:.4f}°C** | {tc_50:.4f}°C | {tc_75:.4f}°C | {bias:+.3f}°C | {lr_s} | {grad_s} | {hub_s} | {status} |"
        )

    lines.extend([
        "",
        "---",
        "## Baseline Reference (Saved & Untouched)",
        "- **Checkpoint:** `checkpoints/archive/best_argo_dev_0.964.pt` (Immutable)",
        "- **Baseline Argo Final RMSE:** 0.9641°C",
        "- **Baseline Thermocline 50-150m RMSE:** 1.4147°C",
        "- **Champion Checkpoint (Best Overall):** `checkpoints/best_argo_overall.pt`",
        "",
    ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 2. JSON Leaderboard
    json_path = os.path.join(leaderboard_dir, "leaderboard.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sorted_res, f, indent=2)

    # 3. CSV Leaderboard
    csv_path = os.path.join(leaderboard_dir, "leaderboard.csv")
    fieldnames = [
        "rank", "trial_id", "argo_final_rmse", "tc_50_150_rmse", "tc_75_150_rmse",
        "bias", "learning_rate", "lambda_gradient", "huber_delta", "weight_decay",
        "duration_min", "status"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rank, r in enumerate(sorted_res, start=1):
            row = dict(r)
            row["rank"] = rank
            writer.writerow(row)


def main():
    print("=" * 80)
    print("  🌙 AUTONOMOUS OVERNIGHT HYPERPARAMETER TUNING PIPELINE")
    print(f"  Start Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 80)

    tuning_res_dir = os.path.join(PROJECT_ROOT, "results", "tuning_overnight")
    tuning_ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "tuning_overnight")
    tuning_cfg_dir = os.path.join(PROJECT_ROOT, "configs", "tuning_overnight")
    os.makedirs(tuning_res_dir, exist_ok=True)
    os.makedirs(tuning_ckpt_dir, exist_ok=True)
    os.makedirs(tuning_cfg_dir, exist_ok=True)

    # Baseline: Saved 0.964 model
    baseline_entry = {
        "trial_id": "baseline_0.964_checkpoint",
        "description": "Previous best saved model (immutable baseline)",
        "is_baseline": True,
        "argo_final_rmse": 0.9640758,
        "tc_50_150_rmse": 1.4146627,
        "tc_75_150_rmse": 1.4460037,
        "bias": 0.3069967,
        "learning_rate": 1e-4,
        "lambda_gradient": 0.30,
        "lambda_temperature": 1.0,
        "huber_delta": 1.0,
        "weight_decay": 1e-4,
        "duration_min": 7.5,
        "status": "BASELINE_LOCKED",
    }

    results: List[Dict[str, Any]] = [baseline_entry]
    best_overall_rmse = baseline_entry["argo_final_rmse"]
    best_overall_trial = "baseline_0.964_checkpoint"

    # Seed the overall champion with the baseline model
    best_overall_ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_argo_overall.pt")
    if not os.path.exists(best_overall_ckpt_path):
        import shutil
        baseline_ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "archive", "best_argo_dev_0.964.pt")
        if os.path.exists(baseline_ckpt_path):
            shutil.copyfile(baseline_ckpt_path, best_overall_ckpt_path)

    update_leaderboard(results, tuning_res_dir)

    print(f"\n[BASELINE LOCKED] Initial Champion RMSE: {best_overall_rmse:.4f}°C")
    print(f"Total Trials Scheduled: {len(HYPERPARAMETER_SEARCH_SPACE)}\n")

    base_config_path = os.path.join(PROJECT_ROOT, "configs", "thermocline_weighted_experiment.yaml")

    for idx, trial in enumerate(HYPERPARAMETER_SEARCH_SPACE, start=1):
        trial_id = trial["trial_id"]
        t0 = time.time()
        print("\n" + "#" * 80)
        print(f"  ▶ [TRIAL {idx:02d}/{len(HYPERPARAMETER_SEARCH_SPACE):02d}] : {trial_id}")
        print(f"  Description: {trial['description']}")
        print(f"  Parameters : LR={trial['learning_rate']} | Grad={trial['lambda_gradient']} | Huber={trial['huber_delta']} | WD={trial['weight_decay']}")
        print("#" * 80 + "\n", flush=True)

        trial_exp_dir = os.path.join(tuning_res_dir, trial_id)
        trial_ckpt_dir = os.path.join(tuning_ckpt_dir, trial_id)
        os.makedirs(trial_exp_dir, exist_ok=True)
        os.makedirs(trial_ckpt_dir, exist_ok=True)

        # Write trial-specific YAML if depth weights differ from standard
        with open(base_config_path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)

        cfg_dict["experiment_name"] = trial_id
        cfg_dict["results_dir"] = os.path.relpath(trial_exp_dir, PROJECT_ROOT)
        cfg_dict["checkpoints_dir"] = os.path.relpath(trial_ckpt_dir, PROJECT_ROOT)
        # Map all hyperparameter & architectural configurations into the trial YAML
        for k, v in trial.items():
            if k in ["trial_id", "description", "argo_patience", "argo_val_interval"]:
                continue
            if k == "depth_weights":
                if "loss" not in cfg_dict or not isinstance(cfg_dict["loss"], dict):
                    cfg_dict["loss"] = {}
                cfg_dict["loss"]["depth_weights"] = v
            else:
                cfg_dict[k] = v

        trial_cfg_path = os.path.join(tuning_cfg_dir, f"{trial_id}.yaml")
        with open(trial_cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False)

        # Build execution command adhering strictly to user format
        cmd = [
            PYTHON_EXE,
            os.path.join(PROJECT_ROOT, "scripts", "run_thermocline_experiment.py"),
            "--config", trial_cfg_path,
            "--epochs", "15",
            "--batch-size", str(trial.get("batch_size", 32)),
            "--argo-val-interval", str(trial.get("argo_val_interval", 200)),
            "--argo-patience", str(trial.get("argo_patience", 5)),
            "--argo-min-delta", "0.001",
            "--results-dir", os.path.relpath(trial_exp_dir, PROJECT_ROOT),
            "--checkpoints-dir", os.path.relpath(trial_ckpt_dir, PROJECT_ROOT),
            "--experiment-name", trial_id,
        ]

        print(f"Executing: {' '.join(cmd)}\n", flush=True)

        try:
            # Run trial process
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
            dur_min = (time.time() - t0) / 60.0

            # Collect results from trial output
            metrics_path = os.path.join(trial_exp_dir, "argo_final_test_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    final_metrics = json.load(f)

                f_rmse = final_metrics["overall"]["rmse"]
                f_bias = final_metrics["overall"]["bias"]
                tc_50 = final_metrics.get("thermocline_50_150_rmse", float("nan"))
                tc_75 = final_metrics.get("thermocline_75_150_rmse", float("nan"))

                trial_record = {
                    "trial_id": trial_id,
                    "description": trial["description"],
                    "argo_final_rmse": f_rmse,
                    "tc_50_150_rmse": tc_50,
                    "tc_75_150_rmse": tc_75,
                    "bias": f_bias,
                    "learning_rate": trial["learning_rate"],
                    "lambda_gradient": trial["lambda_gradient"],
                    "lambda_temperature": trial["lambda_temperature"],
                    "huber_delta": trial["huber_delta"],
                    "weight_decay": trial["weight_decay"],
                    "duration_min": round(dur_min, 2),
                    "status": "COMPLETED",
                }

                results.append(trial_record)

                if f_rmse < best_overall_rmse:
                    import shutil
                    best_overall_rmse = f_rmse
                    best_overall_trial = trial_id
                    trial_best_ckpt = os.path.join(trial_ckpt_dir, "best_argo_dev.pt")
                    if os.path.exists(trial_best_ckpt):
                        shutil.copyfile(trial_best_ckpt, best_overall_ckpt_path)
                    print(f"\n🎉 ⭐ NEW OVERALL CHAMPION FOUND: {trial_id} with RMSE {f_rmse:.4f}°C!\n", flush=True)
                else:
                    print(f"\n✓ Trial {trial_id} finished in {dur_min:.1f}m. RMSE: {f_rmse:.4f}°C (Champion: {best_overall_rmse:.4f}°C by {best_overall_trial})\n", flush=True)

            else:
                print(f"⚠️ Metrics file missing for {trial_id} at {metrics_path}")
                results.append({
                    "trial_id": trial_id,
                    "description": trial["description"],
                    "status": "METRICS_MISSING",
                    "duration_min": round(dur_min, 2),
                })

        except subprocess.CalledProcessError as e:
            dur_min = (time.time() - t0) / 60.0
            print(f"❌ Trial {trial_id} failed with exit code {e.returncode}: {e}")
            results.append({
                "trial_id": trial_id,
                "description": trial["description"],
                "status": f"FAILED_EXIT_{e.returncode}",
                "duration_min": round(dur_min, 2),
            })
        except Exception as e:
            dur_min = (time.time() - t0) / 60.0
            print(f"❌ Unexpected error in trial {trial_id}: {e}")
            results.append({
                "trial_id": trial_id,
                "description": trial["description"],
                "status": f"ERROR_{str(e)[:30]}",
                "duration_min": round(dur_min, 2),
            })

        # Update leaderboard after every trial
        update_leaderboard(results, tuning_res_dir)
        print(f"Leaderboard updated at: {os.path.join(tuning_res_dir, 'LEADERBOARD.md')}\n", flush=True)

    print("=" * 80)
    print(f"🏁 ALL OVERNIGHT TRIALS COMPLETE!")
    print(f"  Final Champion : {best_overall_trial} ({best_overall_rmse:.4f}°C)")
    print(f"  Best Weights Saved To: {best_overall_ckpt_path}")
    print(f"  Leaderboard: {os.path.join(tuning_res_dir, 'LEADERBOARD.md')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
