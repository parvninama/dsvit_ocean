"""
argo_metrics.py - Detailed statistical evaluation against independent Argo profile observations.

Computes:
  - Overall MAE, RMSE, bias, Pearson correlation, total valid observation count
  - Per-depth MAE, RMSE, bias, Pearson correlation, observation count
  - Uncertainty metrics: mean sigma, sigma-error correlation, 1-sigma / 2-sigma empirical coverage
  - Shallow vs. deep water stratification based on valid vertical levels
  - CSV and JSON export functions
"""
from __future__ import annotations
import csv
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

TARGET_DEPTHS_M = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
DEPTH_LABELS = [f"{d}m" for d in TARGET_DEPTHS_M]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    std_a = np.std(a)
    std_b = np.std(b)
    if std_a < 1e-9 or std_b < 1e-9:
        return float("nan")
    c = float(np.corrcoef(a, b)[0, 1])
    return c if not np.isnan(c) else float("nan")


def compute_argo_evaluation_metrics(
    pred_mu: np.ndarray,          # [N, 15] predicted means (degC)
    pred_sigma: np.ndarray,       # [N, 15] predicted uncertainties (degC)
    argo_target: np.ndarray,      # [N, 15] observed Argo temperatures (degC)
    depth_mask: np.ndarray,       # [N, 15] 1=valid Argo observation, 0=invalid/missing
    shallow_thresholds: tuple = (8, 12)
) -> Dict[str, Any]:
    """
    Computes rigorous independent evaluation metrics comparing model predictions to Argo ground truth.
    Respects depth_mask strictly; invalid or NaN depths never contribute to any metric.
    """
    N, D = pred_mu.shape
    assert D == 15, f"Expected 15 depths, got {D}"

    # Overall valid mask: depth_mask must be 1 AND target must not be NaN
    valid_mask = (depth_mask == 1) & (~np.isnan(argo_target)) & (~np.isnan(pred_mu))

    pred_v = pred_mu[valid_mask]
    target_v = argo_target[valid_mask]
    sigma_v = pred_sigma[valid_mask]
    err_v = pred_v - target_v
    abs_err_v = np.abs(err_v)

    n_total_valid = int(valid_mask.sum())

    # Overall metrics
    overall = {
        "mae": float(np.mean(abs_err_v)) if n_total_valid > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(err_v**2))) if n_total_valid > 0 else float("nan"),
        "bias": float(np.mean(err_v)) if n_total_valid > 0 else float("nan"),
        "correlation": _safe_corr(pred_v, target_v),
        "n_valid": n_total_valid,
        "mean_sigma": float(np.mean(sigma_v)) if n_total_valid > 0 else float("nan"),
        "coverage_1sigma": float(np.mean(abs_err_v <= sigma_v)) if n_total_valid > 0 else float("nan"),
        "coverage_2sigma": float(np.mean(abs_err_v <= 2.0 * sigma_v)) if n_total_valid > 0 else float("nan"),
        "sigma_error_corr": _safe_corr(sigma_v, abs_err_v),
        "gaussian_nll": float(np.mean(np.log(np.maximum(sigma_v, 1e-4)) + 0.5 * (err_v / np.maximum(sigma_v, 1e-4))**2)) if n_total_valid > 0 else float("nan")
    }

    # Per-depth metrics
    depth_records = []
    for d_idx, depth_m in enumerate(TARGET_DEPTHS_M):
        d_mask = valid_mask[:, d_idx]
        n_d = int(d_mask.sum())

        if n_d < 1:
            depth_records.append({
                "depth_m": depth_m,
                "label": DEPTH_LABELS[d_idx],
                "n_valid": 0,
                "rmse": float("nan"),
                "mae": float("nan"),
                "bias": float("nan"),
                "correlation": float("nan"),
                "mean_sigma": float("nan"),
                "mean_abs_err": float("nan"),
                "coverage_1sigma": float("nan"),
                "coverage_2sigma": float("nan"),
                "sigma_error_corr": float("nan")
            })
            continue

        p_d = pred_mu[:, d_idx][d_mask]
        t_d = argo_target[:, d_idx][d_mask]
        s_d = pred_sigma[:, d_idx][d_mask]
        e_d = p_d - t_d
        ae_d = np.abs(e_d)

        depth_records.append({
            "depth_m": depth_m,
            "label": DEPTH_LABELS[d_idx],
            "n_valid": n_d,
            "rmse": float(np.sqrt(np.mean(e_d**2))),
            "mae": float(np.mean(ae_d)),
            "bias": float(np.mean(e_d)),
            "correlation": _safe_corr(p_d, t_d),
            "mean_sigma": float(np.mean(s_d)),
            "mean_abs_err": float(np.mean(ae_d)),
            "coverage_1sigma": float(np.mean(ae_d <= s_d)),
            "coverage_2sigma": float(np.mean(ae_d <= 2.0 * s_d)),
            "sigma_error_corr": _safe_corr(s_d, ae_d)
        })

    # Shallow-water / Bathymetric stratification
    valid_depths_per_profile = valid_mask.sum(axis=1)  # [N]
    th_low, th_high = shallow_thresholds

    few_mask = valid_depths_per_profile <= th_low
    med_mask = (valid_depths_per_profile > th_low) & (valid_depths_per_profile <= th_high)
    many_mask = valid_depths_per_profile > th_high

    def calc_group_stats(sel_mask):
        if sel_mask.sum() == 0:
            return {"n_profiles": 0, "n_valid_depths": 0, "rmse": float("nan"), "mae": float("nan"), "bias": float("nan")}
        m_sel = valid_mask[sel_mask]
        p_sel = pred_mu[sel_mask][m_sel]
        t_sel = argo_target[sel_mask][m_sel]
        if len(p_sel) == 0:
            return {"n_profiles": int(sel_mask.sum()), "n_valid_depths": 0, "rmse": float("nan"), "mae": float("nan"), "bias": float("nan")}
        err = p_sel - t_sel
        return {
            "n_profiles": int(sel_mask.sum()),
            "n_valid_depths": len(p_sel),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err))
        }

    water_depth_analysis = {
        f"shallow_coastal (valid_levels<={th_low})": calc_group_stats(few_mask),
        f"intermediate_depth (valid_levels {th_low+1}-{th_high})": calc_group_stats(med_mask),
        f"deep_ocean (valid_levels>{th_high})": calc_group_stats(many_mask)
    }

    return {
        "overall": overall,
        "depth_wise": depth_records,
        "shallow_water_analysis": water_depth_analysis
    }


def format_argo_metrics_table(metrics: Dict[str, Any], prefix: str = "  ") -> str:
    """
    Returns a clean formatted text table of depth-wise and overall metrics.
    """
    ov = metrics["overall"]
    lines = []
    lines.append(f"{prefix}OVERALL METRICS (N_obs={ov['n_valid']:,}):")
    lines.append(f"{prefix}  MAE: {ov['mae']:.3f} C | RMSE: {ov['rmse']:.3f} C | Bias: {ov['bias']:+.3f} C | Corr: {ov['correlation']:.3f}")
    lines.append(f"{prefix}  Mean Sigma: {ov['mean_sigma']:.3f} C | 1-Sigma Cov: {ov['coverage_1sigma']*100:.1f}% | 2-Sigma Cov: {ov['coverage_2sigma']*100:.1f}% | NLL: {ov['gaussian_nll']:.3f}")
    lines.append("")
    lines.append(f"{prefix}{'Depth':>6s} | {'RMSE (C)':>9s} | {'MAE (C)':>9s} | {'Bias (C)':>9s} | {'Corr':>6s} | {'Sigma':>7s} | {'Cov 1s':>6s} | {'Cov 2s':>6s} | {'N_valid':>7s}")
    lines.append(f"{prefix}" + "-"*83)

    for rec in metrics["depth_wise"]:
        d_lbl = rec["label"]
        n_v = rec["n_valid"]
        if n_v == 0:
            lines.append(f"{prefix}{d_lbl:>6s} | {'nan':>9s} | {'nan':>9s} | {'nan':>9s} | {'nan':>6s} | {'nan':>7s} | {'nan':>6s} | {'nan':>6s} | {0:>7d}")
            continue

        c1 = f"{rec['coverage_1sigma']*100:.1f}%" if not np.isnan(rec['coverage_1sigma']) else "nan"
        c2 = f"{rec['coverage_2sigma']*100:.1f}%" if not np.isnan(rec['coverage_2sigma']) else "nan"
        cr = f"{rec['correlation']:.3f}" if not np.isnan(rec['correlation']) else "nan"

        lines.append(
            f"{prefix}{d_lbl:>6s} | {rec['rmse']:9.3f} | {rec['mae']:9.3f} | {rec['bias']:+9.3f} | "
            f"{cr:>6s} | {rec['mean_sigma']:7.3f} | {c1:>6s} | {c2:>6s} | {n_v:7d}"
        )

    lines.append("")
    lines.append(f"{prefix}SHALLOW VS. DEEP WATER STRATIFICATION:")
    for grp_name, s in metrics["shallow_water_analysis"].items():
        lines.append(
            f"{prefix}  {grp_name:40s}: Profiles={s['n_profiles']:4d} | RMSE={s['rmse']:.3f} C | MAE={s['mae']:.3f} C | Bias={s['bias']:+.3f} C"
        )

    return "\n".join(lines)


def export_argo_metrics(metrics: Dict[str, Any], output_dir: str):
    """
    Saves argo_test_metrics.csv and argo_test_metrics.json.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(output_dir, "argo_test_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save CSV
    csv_path = os.path.join(output_dir, "argo_test_metrics.csv")
    fields = ["depth_m", "label", "rmse", "mae", "bias", "correlation", "mean_sigma", "coverage_1sigma", "coverage_2sigma", "sigma_error_corr", "n_valid"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rec in metrics["depth_wise"]:
            writer.writerow(rec)

    print(f"[argo_metrics] Exported metrics to {json_path} and {csv_path}")