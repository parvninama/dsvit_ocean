"""
visualization.py - Publication-quality diagnostic plotting for independent Argo evaluation.

Generates:
  - rmse_vs_depth.png
  - mae_vs_depth.png
  - bias_vs_depth.png
  - correlation_vs_depth.png
  - uncertainty_vs_depth.png
  - uncertainty_vs_absolute_error.png
  - coverage_vs_depth.png
  - geographic_profile_rmse.png
  - depth_error_maps.png
  - example_profiles.png
  - failure_profiles.png
  - shallow_vs_deep_performance.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

TARGET_DEPTHS_M = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
DEPTH_LABELS = [f"{d}m" for d in TARGET_DEPTHS_M]


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    # Also save with exact argo_ prefix requested by protocol
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    if not basename.startswith("argo_"):
        if basename == "geographic_profile_rmse.png":
            argo_name = "argo_geographic_error_map.png"
        elif basename == "uncertainty_vs_absolute_error.png":
            argo_name = "argo_uncertainty_vs_error.png"
        else:
            argo_name = "argo_" + basename
        argo_path = os.path.join(dirname, argo_name)
        fig.savefig(argo_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_depth(metrics: Dict[str, Any], output_dir: str):
    """
    Plots RMSE, MAE, Bias, and Pearson Correlation vs Depth (inverted depth axis).
    """
    recs = metrics["depth_wise"]
    depths = [r["depth_m"] for r in recs if r["n_valid"] > 0]
    rmses  = [r["rmse"] for r in recs if r["n_valid"] > 0]
    maes   = [r["mae"] for r in recs if r["n_valid"] > 0]
    biases = [r["bias"] for r in recs if r["n_valid"] > 0]
    corrs  = [r["correlation"] for r in recs if r["n_valid"] > 0]

    # 1. RMSE vs Depth
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(rmses, depths, "o-", color="crimson", lw=2, ms=6)
    ax.set_xlabel("RMSE (°C)", fontsize=11)
    ax.set_ylabel("Depth (m)", fontsize=11)
    ax.set_title("Argo Test: RMSE vs Depth", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.5)
    _save(fig, os.path.join(output_dir, "rmse_vs_depth.png"))

    # 2. MAE vs Depth
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(maes, depths, "s-", color="darkorange", lw=2, ms=6)
    ax.set_xlabel("MAE (°C)", fontsize=11)
    ax.set_ylabel("Depth (m)", fontsize=11)
    ax.set_title("Argo Test: MAE vs Depth", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.5)
    _save(fig, os.path.join(output_dir, "mae_vs_depth.png"))

    # 3. Bias vs Depth
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(biases, depths, "^-", color="royalblue", lw=2, ms=6)
    ax.axvline(0, color="black", linestyle="--", lw=1.2, alpha=0.7)
    ax.set_xlabel("Mean Bias (°C)", fontsize=11)
    ax.set_ylabel("Depth (m)", fontsize=11)
    ax.set_title("Argo Test: Bias vs Depth", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.5)
    _save(fig, os.path.join(output_dir, "bias_vs_depth.png"))

    # 4. Correlation vs Depth
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(corrs, depths, "d-", color="forestgreen", lw=2, ms=6)
    ax.set_xlabel("Pearson Correlation (r)", fontsize=11)
    ax.set_ylabel("Depth (m)", fontsize=11)
    ax.set_title("Argo Test: Correlation vs Depth", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.5)
    _save(fig, os.path.join(output_dir, "correlation_vs_depth.png"))


def plot_uncertainty_diagnostics(metrics: Dict[str, Any], records: List[Dict[str, Any]], output_dir: str):
    """
    Plots:
      - uncertainty_vs_depth.png: Mean predicted sigma vs mean absolute error across depths
      - coverage_vs_depth.png: 1-sigma and 2-sigma empirical coverage vs theoretical bounds
      - uncertainty_vs_absolute_error.png: Calibration scatter/binned curve
    """
    recs = metrics["depth_wise"]
    depths = [r["depth_m"] for r in recs if r["n_valid"] > 0]
    sigmas = [r["mean_sigma"] for r in recs if r["n_valid"] > 0]
    abs_errs = [r["mean_abs_err"] for r in recs if r["n_valid"] > 0]
    cov1 = [r["coverage_1sigma"] * 100 for r in recs if r["n_valid"] > 0]
    cov2 = [r["coverage_2sigma"] * 100 for r in recs if r["n_valid"] > 0]

    # Uncertainty vs Depth
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.plot(sigmas, depths, "o-", color="purple", lw=2, label="Mean Predicted Uncertainty (σ)")
    ax.plot(abs_errs, depths, "s--", color="gray", lw=2, label="Mean Absolute Error (|err|)")
    ax.set_xlabel("Temperature (°C)", fontsize=11)
    ax.set_ylabel("Depth (m)", fontsize=11)
    ax.set_title("Uncertainty Calibration vs Depth", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    _save(fig, os.path.join(output_dir, "uncertainty_vs_depth.png"))

    # Coverage vs Depth
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(depths))
    width = 0.35
    ax.bar(x - width/2, cov1, width, label="1-Sigma Coverage (Actual)", color="cornflowerblue", alpha=0.85)
    ax.bar(x + width/2, cov2, width, label="2-Sigma Coverage (Actual)", color="lightsteelblue", alpha=0.85)
    ax.axhline(68.3, color="blue", linestyle="--", lw=1.5, label="Theoretical 1-Sigma (68.3%)")
    ax.axhline(95.4, color="darkblue", linestyle=":", lw=1.5, label="Theoretical 2-Sigma (95.4%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}m" for d in depths], rotation=45)
    ax.set_ylabel("Empirical Coverage (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("Argo Test: Empirical Uncertainty Coverage by Depth", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    _save(fig, os.path.join(output_dir, "coverage_vs_depth.png"))

    # Uncertainty vs Absolute Error calibration scatter / binned curve
    all_sigmas, all_errors = [], []
    for r in records:
        valid = (r["depth_mask"] == 1) & (~np.isnan(r["argo_temperature"]))
        all_sigmas.extend(r["predicted_uncertainty"][valid])
        all_errors.extend(np.abs(r["predicted_temperature"][valid] - r["argo_temperature"][valid]))

    if all_sigmas:
        all_sigmas = np.array(all_sigmas)
        all_errors = np.array(all_errors)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(all_sigmas, all_errors, alpha=0.2, s=12, color="teal", rasterized=True, label="Individual Valid Levels")

        # Binned mean
        bins = np.linspace(all_sigmas.min(), np.percentile(all_sigmas, 99), 12)
        bin_idx = np.digitize(all_sigmas, bins)
        bin_sig, bin_err = [], []
        for b in range(1, len(bins)):
            sel = bin_idx == b
            if sel.sum() > 5:
                bin_sig.append(float(np.mean(all_sigmas[sel])))
                bin_err.append(float(np.mean(all_errors[sel])))

        if bin_sig:
            ax.plot(bin_sig, bin_err, "r-o", lw=2.5, ms=7, label="Binned Mean Error")
            max_val = max(max(bin_sig), max(bin_err))
            ax.plot([0, max_val], [0, max_val], "k--", lw=1.2, label="Perfect Calibration (y=x)")

        ax.set_xlabel("Predicted Uncertainty σ (°C)", fontsize=11)
        ax.set_ylabel("Observed Absolute Error |T_pred - T_argo| (°C)", fontsize=11)
        ax.set_title("Predicted Uncertainty vs Actual Absolute Error", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        _save(fig, os.path.join(output_dir, "uncertainty_vs_absolute_error.png"))


def plot_geographic_errors(records: List[Dict[str, Any]], output_dir: str, depth_levels: List[int] = [100, 300, 500]):
    """
    Plots geographic error maps over the Bay of Bengal region:
      - Overall profile RMSE
      - Depth-specific error maps (e.g. 100m, 300m, 500m)
    """
    lats, lons, rmses = [], [], []
    for r in records:
        valid = (r["depth_mask"] == 1) & (~np.isnan(r["argo_temperature"]))
        if valid.sum() > 0:
            diff = r["predicted_temperature"][valid] - r["argo_temperature"][valid]
            rmses.append(float(np.sqrt(np.mean(diff**2))))
            lats.append(r["latitude"])
            lons.append(r["longitude"])

    if not lats:
        return

    # Profile RMSE map
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(lons, lats, c=rmses, cmap="YlOrRd", s=45, edgecolor="black", linewidth=0.5, alpha=0.85)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Profile RMSE (°C)", fontsize=11)
    ax.set_xlim(80, 100)
    ax.set_ylim(5, 23)
    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.set_title(f"Argo Test: Geographic Distribution of Profile RMSE (N={len(lats)})", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    _save(fig, os.path.join(output_dir, "geographic_profile_rmse.png"))

    # Depth-specific maps
    fig, axes = plt.subplots(1, len(depth_levels), figsize=(6 * len(depth_levels), 6))
    if len(depth_levels) == 1:
        axes = [axes]

    for ax, depth_target in zip(axes, depth_levels):
        if depth_target in TARGET_DEPTHS_M:
            d_idx = TARGET_DEPTHS_M.index(depth_target)
            d_lats, d_lons, d_errs = [], [], []
            for r in records:
                if r["depth_mask"][d_idx] == 1 and not np.isnan(r["argo_temperature"][d_idx]):
                    e = abs(r["predicted_temperature"][d_idx] - r["argo_temperature"][d_idx])
                    d_errs.append(e)
                    d_lats.append(r["latitude"])
                    d_lons.append(r["longitude"])

            if d_lats:
                sc_d = ax.scatter(d_lons, d_lats, c=d_errs, cmap="plasma", s=40, edgecolor="k", lw=0.4, alpha=0.85)
                cbar_d = plt.colorbar(sc_d, ax=ax, shrink=0.7)
                cbar_d.set_label(f"|Error| at {depth_target}m (°C)", fontsize=10)

            ax.set_xlim(80, 100)
            ax.set_ylim(5, 23)
            ax.set_xlabel("Longitude (°E)", fontsize=10)
            ax.set_ylabel("Latitude (°N)", fontsize=10)
            ax.set_title(f"Absolute Error at {depth_target}m (N={len(d_lats)})", fontsize=11, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "depth_error_maps.png"))


def plot_vertical_profile_examples(records: List[Dict[str, Any]], output_dir: str, num_profiles: int = 12):
    """
    Plots representative vertical profiles showing Argo observations vs. model predictions
    with uncertainty bands (inverted depth axis, surface at top).
    Selects from low error, median error, and high error regimes.
    """
    scored = []
    for r in records:
        valid = (r["depth_mask"] == 1) & (~np.isnan(r["argo_temperature"]))
        if valid.sum() >= 6:
            err = r["predicted_temperature"][valid] - r["argo_temperature"][valid]
            rmse = float(np.sqrt(np.mean(err**2)))
            scored.append((rmse, r))

    if not scored:
        return

    scored.sort(key=lambda x: x[0])
    N = len(scored)

    # Select samples from across performance distribution: low (best), median, high (challenging)
    k_each = max(1, num_profiles // 3)
    low_samples = [x[1] for x in scored[:k_each]]
    med_start = max(0, N//2 - k_each//2)
    med_samples = [x[1] for x in scored[med_start : med_start + k_each]]
    high_samples = [x[1] for x in scored[-k_each:]]

    selected = [("Low Error", low_samples), ("Median Error", med_samples), ("High Error", high_samples)]

    fig, axes = plt.subplots(3, k_each, figsize=(4.5 * k_each, 14), sharey=True)
    if k_each == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    for row_idx, (group_name, prof_list) in enumerate(selected):
        for col_idx, prof in enumerate(prof_list):
            ax = axes[row_idx, col_idx]
            dmask = prof["depth_mask"]
            valid = (dmask == 1) & (~np.isnan(prof["argo_temperature"]))

            d_plot = np.array(TARGET_DEPTHS_M)[valid]
            t_argo = prof["argo_temperature"][valid]
            t_pred = prof["predicted_temperature"][valid]
            t_sig  = prof["predicted_uncertainty"][valid]

            # Inverted depth: surface at top (0m), bottom at 1000m
            ax.plot(t_argo, d_plot, "k-o", ms=4, lw=1.8, label="Argo Obs")
            ax.plot(t_pred, d_plot, "r-s", ms=4, lw=1.8, label="Model μ")
            ax.fill_betweenx(d_plot, t_pred - t_sig, t_pred + t_sig, color="red", alpha=0.25, label="±1σ Band")

            ax.invert_yaxis()
            ax.set_xlabel("Temp (°C)", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel("Depth (m)", fontsize=10)

            diff = t_pred - t_argo
            p_rmse = float(np.sqrt(np.mean(diff**2)))
            date_s = str(prof["time"])[:10]
            ax.set_title(f"{group_name}\n{date_s} ({prof['latitude']:.1f}°N, {prof['longitude']:.1f}°E)\nRMSE={p_rmse:.2f}°C", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.5)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, loc="lower left")

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "example_profiles.png"))


def plot_failure_profiles(top_failures: List[Dict[str, Any]], output_dir: str):
    """
    Plots the top worst-performing profiles with error diagnostics.
    """
    if not top_failures:
        return

    n_show = min(6, len(top_failures))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=True)
    axes = axes.ravel()

    for idx in range(n_show):
        ax = axes[idx]
        prof = top_failures[idx]
        valid = (prof["depth_mask"] == 1) & (~np.isnan(prof["argo_temperature"]))

        d_plot = np.array(TARGET_DEPTHS_M)[valid]
        t_argo = prof["argo_temperature"][valid]
        t_pred = prof["predicted_temperature"][valid]
        t_sig  = prof["predicted_uncertainty"][valid]

        ax.plot(t_argo, d_plot, "k-o", ms=4, lw=2, label="Argo Ground Truth")
        ax.plot(t_pred, d_plot, "r-s", ms=4, lw=2, label="Model Prediction")
        ax.fill_betweenx(d_plot, t_pred - t_sig, t_pred + t_sig, color="red", alpha=0.3, label="±1σ Uncertainty")

        ax.invert_yaxis()
        ax.set_xlabel("Temperature (°C)", fontsize=10)
        if idx % 3 == 0:
            ax.set_ylabel("Depth (m)", fontsize=10)

        ax.set_title(
            f"Rank #{idx+1}: {prof['profile_id']}\n"
            f"RMSE={prof['profile_rmse']:.2f}°C | Dist={prof['spatial_distance_km']:.1f}km\n"
            f"Lat={prof['latitude']:.2f}, Lon={prof['longitude']:.2f}",
            fontsize=9
        )
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend(fontsize=8, loc="lower left")

    # Hide unused subplots
    for idx in range(n_show, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "failure_profiles.png"))


def generate_all_argo_plots(
    metrics: Dict[str, Any],
    records: List[Dict[str, Any]],
    top_failures: List[Dict[str, Any]],
    output_dir: str,
    cfg: Any
):
    """
    Orchestrates generation of all required diagnostic plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_metric_vs_depth(metrics, output_dir)
    plot_uncertainty_diagnostics(metrics, records, output_dir)
    plot_geographic_errors(records, output_dir, depth_levels=cfg.depth_map_levels)
    plot_vertical_profile_examples(records, output_dir, num_profiles=cfg.num_profile_plots)
    plot_failure_profiles(top_failures, output_dir)
    print(f"[visualization] All diagnostic plots saved to {output_dir}")