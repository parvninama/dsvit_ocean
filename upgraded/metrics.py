import numpy as np
from typing import Dict, List, Optional

DEPTH_LABELS = ["0m","5m","10m","20m","30m","50m","75m","100m",
                "125m","150m","200m","300m","500m","700m","1000m"]
SURF_VARS = ["SST", "SSS", "SLA", "U_curr", "V_curr", "U_wind", "V_wind"]

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    std_a = np.std(a)
    std_b = np.std(b)
    if std_a < 1e-10 or std_b < 1e-10:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_upgraded_metrics(
    pred_abs: np.ndarray,                            # [N, 15] predicted absolute temp
    pred_anom: Optional[np.ndarray] = None,          # [N, 15] predicted anomaly (None in direct mode)
    target_abs: np.ndarray = None,                   # [N, 15] ground truth temp
    target_anom: Optional[np.ndarray] = None,        # [N, 15] ground truth anomaly (None in direct mode)
    depth_mask: np.ndarray = None,                   # [N, 15] binary mask
    pred_sigma: Optional[np.ndarray] = None,         # [N, 15] std dev from (D + diag(L L^T))^0.5
    recon: Optional[np.ndarray] = None,              # [N, 7, 11, 11]
    orig_surf: Optional[np.ndarray] = None,          # [N, 7, 11, 11]
    surf_mask: Optional[np.ndarray] = None,          # [N, 11, 11]
) -> Dict:
    metrics = {}
    N, D = pred_abs.shape
    mask = depth_mask.astype(bool)

    # 1. Overall Absolute Temperature Metrics
    p_abs_v = pred_abs[mask]
    t_abs_v = target_abs[mask]
    err_abs = p_abs_v - t_abs_v

    metrics["overall_abs_mae"]  = float(np.mean(np.abs(err_abs))) if len(err_abs) > 0 else float("nan")
    metrics["overall_abs_rmse"] = float(np.sqrt(np.mean(err_abs ** 2))) if len(err_abs) > 0 else float("nan")
    metrics["overall_abs_bias"] = float(np.mean(err_abs)) if len(err_abs) > 0 else float("nan")
    metrics["overall_abs_corr"] = _safe_corr(p_abs_v, t_abs_v)
    metrics["n_valid_total"]    = int(mask.sum())

    # For backward-compatible reporting keys:
    metrics["overall_mae"]  = metrics["overall_abs_mae"]
    metrics["overall_rmse"] = metrics["overall_abs_rmse"]
    metrics["overall_bias"] = metrics["overall_abs_bias"]
    metrics["overall_corr"] = metrics["overall_abs_corr"]

    # 2. Overall Anomaly Metrics (Active only if anomaly targets are provided)
    has_anom = pred_anom is not None and target_anom is not None
    if has_anom:
        p_anom_v = pred_anom[mask]
        t_anom_v = target_anom[mask]
        err_anom = p_anom_v - t_anom_v
        metrics["overall_anom_mae"]  = float(np.mean(np.abs(err_anom))) if len(err_anom) > 0 else float("nan")
        metrics["overall_anom_rmse"] = float(np.sqrt(np.mean(err_anom ** 2))) if len(err_anom) > 0 else float("nan")
        metrics["overall_anom_bias"] = float(np.mean(err_anom)) if len(err_anom) > 0 else float("nan")
        metrics["overall_anom_corr"] = _safe_corr(p_anom_v, t_anom_v)
    else:
        metrics["overall_anom_mae"]  = float("nan")
        metrics["overall_anom_rmse"] = float("nan")
        metrics["overall_anom_bias"] = float("nan")
        metrics["overall_anom_corr"] = float("nan")

    # 3. Depth-Wise Metrics
    dw_abs_mae, dw_abs_rmse, dw_abs_bias, dw_abs_corr = [], [], [], []
    dw_anom_mae, dw_anom_rmse, dw_anom_bias, dw_anom_corr = [], [], [], []
    dw_count, dw_unc_mean, coverage_1s, coverage_2s = [], [], [], []

    for d in range(D):
        dm = mask[:, d]
        n = int(dm.sum())
        dw_count.append(n)
        if n == 0:
            for lst in [dw_abs_mae, dw_abs_rmse, dw_abs_bias, dw_abs_corr,
                        dw_anom_mae, dw_anom_rmse, dw_anom_bias, dw_anom_corr,
                        dw_unc_mean, coverage_1s, coverage_2s]:
                lst.append(float("nan"))
            continue

        # Absolute
        pa = pred_abs[:, d][dm]
        ta = target_abs[:, d][dm]
        ea = pa - ta
        dw_abs_mae.append(float(np.mean(np.abs(ea))))
        dw_abs_rmse.append(float(np.sqrt(np.mean(ea ** 2))))
        dw_abs_bias.append(float(np.mean(ea)))
        dw_abs_corr.append(_safe_corr(pa, ta))

        # Anomaly
        if has_anom:
            pn = pred_anom[:, d][dm]
            tn = target_anom[:, d][dm]
            en = pn - tn
            dw_anom_mae.append(float(np.mean(np.abs(en))))
            dw_anom_rmse.append(float(np.sqrt(np.mean(en ** 2))))
            dw_anom_bias.append(float(np.mean(en)))
            dw_anom_corr.append(_safe_corr(pn, tn))
        else:
            dw_anom_mae.append(float("nan"))
            dw_anom_rmse.append(float("nan"))
            dw_anom_bias.append(float("nan"))
            dw_anom_corr.append(float("nan"))

        # Uncertainty calibration
        if pred_sigma is not None:
            sig = pred_sigma[:, d][dm]
            dw_unc_mean.append(float(np.mean(sig)))
            coverage_1s.append(float(np.mean(np.abs(ea) <= sig)))
            coverage_2s.append(float(np.mean(np.abs(ea) <= 2.0 * sig)))
        else:
            dw_unc_mean.append(float("nan"))
            coverage_1s.append(float("nan"))
            coverage_2s.append(float("nan"))

    metrics["depth_mae"]        = dw_abs_mae
    metrics["depth_rmse"]       = dw_abs_rmse
    metrics["depth_bias"]       = dw_abs_bias
    metrics["depth_corr"]       = dw_abs_corr
    metrics["depth_count"]      = dw_count
    metrics["depth_anom_rmse"]  = dw_anom_rmse
    metrics["depth_anom_mae"]   = dw_anom_mae
    metrics["depth_unc_mean"]   = dw_unc_mean
    metrics["depth_cov_1sigma"] = coverage_1s
    metrics["depth_cov_2sigma"] = coverage_2s

    # 4. Uncertainty Calibration (Overall)
    if pred_sigma is not None:
        sig_v = pred_sigma[mask]
        abs_err = np.abs(err_abs)
        metrics["unc_nll"] = float(np.mean(np.log(sig_v + 1e-6) + 0.5 * (err_abs / (sig_v + 1e-6)) ** 2))
        metrics["cov_1sigma"] = float(np.mean(abs_err <= sig_v))
        metrics["cov_2sigma"] = float(np.mean(abs_err <= 2.0 * sig_v))
        metrics["mean_sigma"] = float(np.mean(sig_v))
        metrics["unc_err_corr"] = _safe_corr(sig_v, abs_err)
    else:
        metrics["unc_nll"] = float("nan")
        metrics["cov_1sigma"] = float("nan")
        metrics["cov_2sigma"] = float("nan")
        metrics["mean_sigma"] = float("nan")
        metrics["unc_err_corr"] = float("nan")

    # 5. Reconstruction Metrics (per surface channel)
    if recon is not None and orig_surf is not None:
        rec_mae, rec_rmse = [], []
        for ch in range(recon.shape[1]):
            if surf_mask is not None:
                m = surf_mask.astype(bool)
                r = recon[:, ch][m]
                o = orig_surf[:, ch][m]
            else:
                r = recon[:, ch].ravel()
                o = orig_surf[:, ch].ravel()
            if len(r) == 0:
                rec_mae.append(float("nan"))
                rec_rmse.append(float("nan"))
                continue
            rec_mae.append(float(np.mean(np.abs(r - o))))
            rec_rmse.append(float(np.sqrt(np.mean((r - o) ** 2))))
        metrics["rec_mae"]  = rec_mae
        metrics["rec_rmse"] = rec_rmse

    return metrics


def format_upgraded_metrics_table(metrics: Dict, prefix: str = "") -> str:
    lines = []
    lines.append(f"{prefix}Overall Absolute: RMSE={metrics.get('overall_abs_rmse', float('nan')):6.3f}C  "
                 f"MAE={metrics.get('overall_abs_mae', float('nan')):6.3f}C  "
                 f"Bias={metrics.get('overall_abs_bias', float('nan')):+6.3f}C  "
                 f"Corr={metrics.get('overall_abs_corr', float('nan')):5.3f}")
    lines.append(f"{prefix}Overall Anomaly : RMSE={metrics.get('overall_anom_rmse', float('nan')):6.3f}C  "
                 f"MAE={metrics.get('overall_anom_mae', float('nan')):6.3f}C  "
                 f"Corr={metrics.get('overall_anom_corr', float('nan')):5.3f}  "
                 f"N={metrics.get('n_valid_total', 0):,}")
    lines.append(f"{prefix}Uncertainty Cal : MeanSigma={metrics.get('mean_sigma', float('nan')):6.3f}C  "
                 f"1-Sigma Cov={metrics.get('cov_1sigma', float('nan'))*100:5.1f}%  "
                 f"2-Sigma Cov={metrics.get('cov_2sigma', float('nan'))*100:5.1f}%  "
                 f"Sigma-Err Corr={metrics.get('unc_err_corr', float('nan')):5.3f}")
    lines.append(f"{prefix}{'Depth':6s}  {'AbsRMSE':7s}  {'AnomRMSE':8s}  {'AbsMAE':6s}  {'Bias':6s}  {'Corr':5s}  {'Sigma':6s}  {'Cov1s':5s}  {'N':>6s}")
    for d, lbl in enumerate(DEPTH_LABELS):
        rmse = metrics["depth_rmse"][d]
        anom_rmse = metrics["depth_anom_rmse"][d]
        mae = metrics["depth_mae"][d]
        bias = metrics["depth_bias"][d]
        corr = metrics["depth_corr"][d]
        sigma = metrics["depth_unc_mean"][d]
        cov1 = metrics["depth_cov_1sigma"][d]
        n = metrics["depth_count"][d]
        def fmt(v, sign=""):
            if v is None or v != v or np.isnan(v):
                return "   nan"
            return f"{v:{sign}6.3f}"
        lines.append(f"{prefix}{lbl:6s}  {fmt(rmse)}   {fmt(anom_rmse)}   {fmt(mae)}  {fmt(bias, '+')}  {fmt(corr)}  {fmt(sigma)}  {fmt(cov1)}  {n:>6d}")
    return "\n".join(lines)
