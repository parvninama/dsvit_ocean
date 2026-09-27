import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Optional, List

DEPTH_LABELS = ["0m","5m","10m","20m","30m","50m","75m","100m",
                "125m","150m","200m","300m","500m","700m","1000m"]
TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

def plot_loss_curves(history: Dict, save_dir: str):
    """Plots training and validation loss curves (total and individual components)."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = history.get("epoch", [])
    if len(epochs) == 0:
        return

    # Total loss
    plt.figure(figsize=(7, 5))
    if "train_total_loss" in history:
        plt.plot(epochs, history["train_total_loss"], "b-o", label="Train Total")
    if "val_total_loss" in history:
        plt.plot(epochs, history["val_total_loss"], "r--s", label="Val Total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Total Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_vs_validation_loss.png"), dpi=200)
    plt.close()

    # Component plots
    components = [
        ("anomaly_loss.png", "Anomaly Loss", "train_anomaly_loss", "val_anomaly_loss"),
        ("reconstruction_loss.png", "Surface Reconstruction Loss", "train_rec_loss", "val_rec_loss"),
        ("gradient_loss.png", "Vertical Gradient Loss", "train_grad_loss", "val_grad_loss"),
        ("uncertainty_loss.png", "Uncertainty NLL Loss", "train_unc_loss", "val_unc_loss"),
    ]
    for fname, title, tr_key, va_key in components:
        plt.figure(figsize=(7, 5))
        has_curve = False
        if tr_key in history:
            plt.plot(epochs, history[tr_key], "b-o", label=f"Train {title}")
            has_curve = True
        if va_key in history:
            plt.plot(epochs, history[va_key], "r--s", label=f"Val {title}")
            has_curve = True
        if has_curve:
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, fname), dpi=200)
        plt.close()


def plot_depth_profiles(metrics: Dict, save_dir: str):
    """Plots metric curves vs physical depth."""
    os.makedirs(save_dir, exist_ok=True)
    depths = np.array(TARGET_DEPTHS)

    # 1. Absolute RMSE vs depth
    plt.figure(figsize=(6, 7))
    plt.plot(metrics["depth_rmse"], depths, "r-o", linewidth=2)
    plt.gca().invert_yaxis()
    plt.xlabel("RMSE (°C)")
    plt.ylabel("Depth (m)")
    plt.title("Absolute Temperature RMSE vs Depth")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rmse_vs_depth.png"), dpi=200)
    plt.savefig(os.path.join(save_dir, "absolute_rmse_vs_depth.png"), dpi=200)
    plt.close()

    # 2. Anomaly RMSE vs depth
    if "depth_anom_rmse" in metrics:
        plt.figure(figsize=(6, 7))
        plt.plot(metrics["depth_anom_rmse"], depths, "b-s", linewidth=2)
        plt.gca().invert_yaxis()
        plt.xlabel("Anomaly RMSE (°C)")
        plt.ylabel("Depth (m)")
        plt.title("Temperature Anomaly RMSE vs Depth")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "anomaly_rmse_vs_depth.png"), dpi=200)
        plt.close()

    # 3. MAE vs depth
    plt.figure(figsize=(6, 7))
    plt.plot(metrics["depth_mae"], depths, "g-^", linewidth=2)
    plt.gca().invert_yaxis()
    plt.xlabel("MAE (°C)")
    plt.ylabel("Depth (m)")
    plt.title("Mean Absolute Error vs Depth")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mae_vs_depth.png"), dpi=200)
    plt.close()

    # 4. Bias vs depth
    plt.figure(figsize=(6, 7))
    plt.plot(metrics["depth_bias"], depths, "m-d", linewidth=2)
    plt.axvline(0, color="k", linestyle="--", alpha=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel("Bias (°C)")
    plt.ylabel("Depth (m)")
    plt.title("Mean Temperature Bias vs Depth")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bias_vs_depth.png"), dpi=200)
    plt.close()

    # 5. Correlation vs depth
    plt.figure(figsize=(6, 7))
    plt.plot(metrics["depth_corr"], depths, "c-v", linewidth=2)
    plt.gca().invert_yaxis()
    plt.xlabel("Pearson Correlation (r)")
    plt.ylabel("Depth (m)")
    plt.title("Correlation vs Depth")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "correlation_vs_depth.png"), dpi=200)
    plt.close()

    # 6. Uncertainty (sigma) vs depth
    if "depth_unc_mean" in metrics and not np.isnan(metrics["depth_unc_mean"][0]):
        plt.figure(figsize=(6, 7))
        plt.plot(metrics["depth_unc_mean"], depths, "purple", marker="o", linewidth=2, label="Mean Predicted Sigma")
        if "depth_mae" in metrics:
            plt.plot(metrics["depth_mae"], depths, "orange", linestyle="--", marker="s", label="Observed MAE")
        plt.gca().invert_yaxis()
        plt.xlabel("Uncertainty Sigma (°C)")
        plt.ylabel("Depth (m)")
        plt.title("Predicted Uncertainty vs Depth")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "uncertainty_vs_depth.png"), dpi=200)
        plt.close()

    # 7. Coverage vs depth
    if "depth_cov_1sigma" in metrics and not np.isnan(metrics["depth_cov_1sigma"][0]):
        plt.figure(figsize=(6, 7))
        plt.plot(np.array(metrics["depth_cov_1sigma"]) * 100, depths, "b-o", label="1-Sigma (68.3% target)")
        plt.plot(np.array(metrics["depth_cov_2sigma"]) * 100, depths, "g-s", label="2-Sigma (95.4% target)")
        plt.axvline(68.3, color="b", linestyle=":", alpha=0.6)
        plt.axvline(95.4, color="g", linestyle=":", alpha=0.6)
        plt.gca().invert_yaxis()
        plt.xlabel("Empirical Coverage (%)")
        plt.ylabel("Depth (m)")
        plt.title("Empirical Confidence Interval Coverage vs Depth")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "coverage_vs_depth.png"), dpi=200)
        plt.close()


def plot_sample_profiles(
    pred_abs: np.ndarray,
    target_abs: np.ndarray,
    climatology: np.ndarray,
    depth_mask: np.ndarray,
    pred_sigma: Optional[np.ndarray],
    save_path: str,
    n_samples: int = 6,
    is_failure: bool = False,
):
    """Plots multi-panel vertical temperature profiles (Target vs Pred vs Climatology + 1-sigma)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    depths = np.array(TARGET_DEPTHS)

    # If failure plot, select highest RMSE profiles
    N = len(pred_abs)
    if is_failure:
        profile_rmses = []
        for i in range(N):
            m = depth_mask[i].astype(bool)
            if m.sum() > 0:
                profile_rmses.append(np.sqrt(np.mean((pred_abs[i, m] - target_abs[i, m]) ** 2)))
            else:
                profile_rmses.append(0.0)
        indices = np.argsort(profile_rmses)[::-1][:n_samples]
    else:
        indices = np.linspace(0, N - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, sample_idx in enumerate(indices):
        ax = axes[idx]
        m = depth_mask[sample_idx].astype(bool)
        d_val = depths[m]
        p_val = pred_abs[sample_idx, m]
        t_val = target_abs[sample_idx, m]
        c_val = climatology[sample_idx, m]

        ax.plot(t_val, d_val, "k-o", label="Target (Truth)", linewidth=2)
        ax.plot(p_val, d_val, "r-s", label="Predicted Profile", linewidth=2)
        ax.plot(c_val, d_val, "g--", label="Climatology", linewidth=1.5, alpha=0.8)

        if pred_sigma is not None:
            s_val = pred_sigma[sample_idx, m]
            ax.fill_betweenx(d_val, p_val - s_val, p_val + s_val, color="red", alpha=0.2, label="±1σ Uncertainty")

        ax.invert_yaxis()
        rmse = np.sqrt(np.mean((p_val - t_val) ** 2)) if len(p_val) > 0 else 0.0
        ax.set_title(f"Sample #{sample_idx} (RMSE: {rmse:.2f}°C)")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Depth (m)")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def generate_all_upgraded_plots(
    pred_abs: np.ndarray,
    target_abs: np.ndarray,
    climatology: np.ndarray,
    depth_mask: np.ndarray,
    pred_sigma: Optional[np.ndarray],
    metrics: Dict,
    history: Optional[Dict],
    results_dir: str,
):
    """Master generation of all required diagnostic plots."""
    os.makedirs(results_dir, exist_ok=True)
    if history:
        plot_loss_curves(history, results_dir)
    plot_depth_profiles(metrics, results_dir)

    # Uncertainty vs absolute error scatter
    if pred_sigma is not None:
        mask = depth_mask.astype(bool)
        all_sig = pred_sigma[mask]
        all_err = np.abs(pred_abs[mask] - target_abs[mask])
        sub = np.random.choice(len(all_sig), size=min(5000, len(all_sig)), replace=False)

        plt.figure(figsize=(7, 6))
        plt.scatter(all_sig[sub], all_err[sub], alpha=0.3, c="teal", s=15)
        lim = max(np.max(all_sig[sub]), np.max(all_err[sub]))
        plt.plot([0, lim], [0, lim], "r--", label="1:1 Reference Line")
        plt.xlabel("Predicted Uncertainty Sigma (°C)")
        plt.ylabel("Observed Absolute Error (°C)")
        plt.title("Uncertainty Calibration: Sigma vs Absolute Error")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "uncertainty_vs_absolute_error.png"), dpi=200)
        plt.close()

    # Temperature profile visualizations
    plot_sample_profiles(
        pred_abs, target_abs, climatology, depth_mask, pred_sigma,
        save_path=os.path.join(results_dir, "temperature_profiles.png"),
        n_samples=6, is_failure=False,
    )
    plot_sample_profiles(
        pred_abs, target_abs, climatology, depth_mask, pred_sigma,
        save_path=os.path.join(results_dir, "failure_profiles.png"),
        n_samples=6, is_failure=True,
    )
