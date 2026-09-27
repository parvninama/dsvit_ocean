import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LowRankMultivariateNormal
from typing import Dict, Tuple, Optional

# Physical depth spacing in metres: 15 target depths
TARGET_DEPTHS_M = torch.tensor(
    [0., 5., 10., 20., 30., 50., 75., 100., 125., 150., 200., 300., 500., 700., 1000.]
)
_DEPTH_DIFFS = TARGET_DEPTHS_M[1:] - TARGET_DEPTHS_M[:-1]  # [14] physical dz


def parse_depth_weights(cfg, device: torch.device = torch.device("cpu")) -> Optional[torch.Tensor]:
    """
    Parses depth weights from configuration for the 15 standard GLORYS depth levels:
      [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] m.
    Supports YAML structures:
      loss:
        depth_weights: {"0": 1.0, "5": 1.0, ...}
      or depth_weights: {"0": 1.0, ...} or list of 15 floats.
    """
    if cfg is None:
        return None

    raw = getattr(cfg, "depth_weights", None)
    if raw is None and hasattr(cfg, "loss") and isinstance(cfg.loss, dict):
        raw = cfg.loss.get("depth_weights")

    if raw is None:
        return None

    if isinstance(raw, (list, tuple)):
        if len(raw) != 15:
            raise ValueError(f"depth_weights list must have 15 elements, got {len(raw)}")
        return torch.tensor(raw, dtype=torch.float32, device=device)

    if isinstance(raw, dict):
        weights = []
        depth_ints = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
        for d in depth_ints:
            w = raw.get(str(d), raw.get(d, raw.get(float(d), 1.0)))
            weights.append(float(w))
        return torch.tensor(weights, dtype=torch.float32, device=device)

    return None


def masked_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    depth_weights: Optional[torch.Tensor] = None,
    delta: float = 1.0,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """
    Computes masked (optionally depth-weighted) Huber loss:
      L_temp = sum_i(w_i * m_i * Huber(p_i, t_i)) / (sum_i(w_i * m_i) + eps)
    Returns:
      (weighted_huber, n_valid, unweighted_huber)
    """
    valid = mask.bool()
    if not valid.any():
        zero = torch.tensor(0.0, device=pred.device, requires_grad=True)
        return zero, 0, zero.detach()

    diff = torch.abs(pred - target)
    huber = torch.where(diff < delta, 0.5 * (diff ** 2), delta * (diff - 0.5 * delta))

    m_float = mask.float()
    unweighted_loss = (huber * m_float).sum() / (m_float.sum() + 1e-8)

    if depth_weights is not None:
        w = depth_weights.to(pred.device).view(1, 15)
        weighted_mask = m_float * w
        weighted_loss = (huber * weighted_mask).sum() / (weighted_mask.sum() + 1e-8)
    else:
        weighted_loss = unweighted_loss

    return weighted_loss, int(valid.sum().item()), unweighted_loss


def surface_reconstruction_loss(
    recon: torch.Tensor,       # [B, 7, 11, 11]
    target: torch.Tensor,      # [B, 7, 11, 11]
    surface_mask: torch.Tensor # [B, 11, 11]
) -> torch.Tensor:
    """
    Computes MSE reconstruction loss across 7 physical surface variables, masked by surface_mask.
    """
    m = surface_mask.unsqueeze(1).bool()  # [B, 1, 11, 11]
    if not m.any():
        return torch.tensor(0.0, device=recon.device, requires_grad=True)
    diff = (recon - target) ** 2 * m.float()
    n_valid = m.float().sum() * recon.shape[1]
    return diff.sum() / (n_valid + 1e-8)


def vertical_gradient_consistency_loss(
    pred_temp: torch.Tensor,   # [B, 15]
    target_temp: torch.Tensor, # [B, 15]
    depth_mask: torch.Tensor,  # [B, 15]
    depth_weights: Optional[torch.Tensor] = None,
    loss_type: str = "l1",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Enforces physical vertical temperature gradient stratification:
      G_d = (T_(d+1) - T_d) / (z_(d+1) - z_d)
    using physical depth differences dz and adjacent valid depth pairs.
    Depth interval weights:
      w_grad,i = (w_i + w_(i+1)) / 2
    Returns:
      (weighted_grad_loss, unweighted_grad_loss)
    """
    dz = _DEPTH_DIFFS.to(pred_temp.device).unsqueeze(0)  # [1, 14]
    pair_mask = (depth_mask[:, :-1] * depth_mask[:, 1:]).float()   # [B, 14]
    if not pair_mask.any():
        zero = torch.tensor(0.0, device=pred_temp.device, requires_grad=True)
        return zero, zero.detach()

    g_pred = (pred_temp[:, 1:] - pred_temp[:, :-1]) / dz
    g_true = (target_temp[:, 1:] - target_temp[:, :-1]) / dz

    if loss_type == "l1":
        g_err = torch.abs(g_pred - g_true)
    else:
        g_err = (g_pred - g_true) ** 2

    unweighted_loss = (g_err * pair_mask).sum() / (pair_mask.sum() + 1e-8)

    if depth_weights is not None:
        w = depth_weights.to(pred_temp.device)  # [15]
        w_grad = (w[:-1] + w[1:]) / 2.0         # [14]
        weighted_pair_mask = pair_mask * w_grad.unsqueeze(0)
        weighted_loss = (g_err * weighted_pair_mask).sum() / (weighted_pair_mask.sum() + 1e-8)
    else:
        weighted_loss = unweighted_loss

    return weighted_loss, unweighted_loss


def correlated_uncertainty_nll_loss(
    anomaly_mean: torch.Tensor,     # [B, 15]
    diag_variance: torch.Tensor,    # [B, 15] (positive)
    low_rank_factor: torch.Tensor,  # [B, 15, rank]
    target_anomaly: torch.Tensor,   # [B, 15]
    depth_mask: torch.Tensor,       # [B, 15]
    jitter: float = 1e-4,
) -> torch.Tensor:
    """
    Multivariate Gaussian Negative Log-Likelihood with low-rank + diagonal covariance:
      Sigma = D + L L^T
    Respects depth_mask for bathymetry-truncated profiles by computing exact marginal
    distribution over valid vertical coordinates (principal submatrix is also low-rank + diagonal).
    """
    B = anomaly_mean.shape[0]
    total_nll = 0.0
    valid_samples = 0

    # Check if all depths are valid across the entire batch (fast path)
    if depth_mask.min() == 1.0:
        cov_diag = diag_variance + jitter
        try:
            dist = LowRankMultivariateNormal(loc=anomaly_mean, cov_factor=low_rank_factor, cov_diag=cov_diag)
            log_prob = dist.log_prob(target_anomaly)  # [B]
            return (-log_prob / 15.0).mean()
        except Exception:
            pass  # Fall back to sample-by-sample with extra jitter

    # Robust per-sample evaluation with masking
    for b in range(B):
        mask_b = depth_mask[b].bool()
        k = int(mask_b.sum().item())
        if k < 2:
            continue

        loc_b = anomaly_mean[b, mask_b]
        diag_b = diag_variance[b, mask_b] + jitter
        L_b = low_rank_factor[b, mask_b, :]
        y_b = target_anomaly[b, mask_b]

        try:
            dist = LowRankMultivariateNormal(loc=loc_b, cov_factor=L_b, cov_diag=diag_b)
            lp = dist.log_prob(y_b)
            if torch.isfinite(lp):
                total_nll += -lp / k
                valid_samples += 1
        except Exception:
            # Fallback to diagonal-only NLL with jitter
            var_diag = diag_b + (L_b ** 2).sum(dim=-1)
            nll_diag = 0.5 * torch.sum(torch.log(var_diag) + ((y_b - loc_b) ** 2) / var_diag) / k
            if torch.isfinite(nll_diag):
                total_nll += nll_diag
                valid_samples += 1

    if valid_samples == 0:
        return torch.tensor(0.0, device=anomaly_mean.device, requires_grad=True)
    return total_nll / valid_samples


def compute_upgraded_loss(
    model_output,                         # UpgradedModelOutput or dict
    target_temp: torch.Tensor,            # [B, 15] absolute temperature target
    climatology: Optional[torch.Tensor],  # [B, 15] climatological temperature or None in direct mode
    orig_surface: torch.Tensor,           # [B, 7, 11, 11]
    surface_mask: torch.Tensor,           # [B, 11, 11]
    depth_mask: torch.Tensor,             # [B, 15]
    cfg,
) -> Dict[str, torch.Tensor]:
    """
    Computes total composite loss for the upgraded model:
      L_total = lambda_temp * L_temp + lambda_grad * L_grad + lambda_recon * L_recon + lambda_unc * L_unc
    Supports depth-weighted Huber loss and depth-weighted vertical gradient loss.
    """
    # Unpack outputs
    pred_anom = model_output.anomaly_mean
    pred_abs  = model_output.absolute_temp
    diag_var  = model_output.diag_variance
    low_rank  = model_output.low_rank_factor
    recon     = model_output.recon

    # Parse depth weights (if specified in config)
    depth_weights = parse_depth_weights(cfg, device=target_temp.device)

    # 1. Anomaly loss (Huber) - active only when climatology is provided (Epoch 1)
    if climatology is not None:
        target_anom = target_temp - climatology
        l_anomaly, n_valid, unweighted_anom = masked_huber_loss(
            pred_anom, target_anom, depth_mask, depth_weights=None, delta=getattr(cfg, "huber_delta", 1.0)
        )
        lam_anom = getattr(cfg, "lambda_anomaly", 1.0)
        lam_temp = getattr(cfg, "lambda_temperature", getattr(cfg, "lambda_absolute", 0.5))
        unc_loc = pred_anom
        unc_tgt = target_anom
    else:
        # Epoch 2+: Direct temperature prediction (Fourier climatology removed)
        l_anomaly = torch.tensor(0.0, device=target_temp.device)
        unweighted_anom = torch.tensor(0.0, device=target_temp.device)
        lam_anom = 0.0
        lam_temp = getattr(cfg, "lambda_temperature", getattr(cfg, "lambda_absolute", 1.0))
        unc_loc = pred_abs
        unc_tgt = target_temp

    # 2. Absolute temperature loss (Depth-Weighted Huber)
    l_absolute, n_valid_abs, unweighted_temp = masked_huber_loss(
        pred_abs, target_temp, depth_mask, depth_weights=depth_weights, delta=getattr(cfg, "huber_delta", 1.0)
    )
    if climatology is None:
        n_valid = n_valid_abs

    # 3. Surface reconstruction loss
    l_recon = torch.tensor(0.0, device=target_temp.device)
    if getattr(cfg, "use_reconstruction", True) and recon is not None:
        l_recon = surface_reconstruction_loss(recon, orig_surface, surface_mask)

    # 4. Vertical gradient consistency loss (Depth-Weighted)
    l_gradient = torch.tensor(0.0, device=target_temp.device)
    unweighted_grad = torch.tensor(0.0, device=target_temp.device)
    if getattr(cfg, "use_gradient_loss", True):
        l_gradient, unweighted_grad = vertical_gradient_consistency_loss(
            pred_abs, target_temp, depth_mask, depth_weights=depth_weights, loss_type="l1"
        )

    # 5. Correlated uncertainty loss (NLL)
    l_uncertainty = torch.tensor(0.0, device=target_temp.device)
    if getattr(cfg, "use_correlated_uncertainty", True) and diag_var is not None and low_rank is not None:
        l_uncertainty = correlated_uncertainty_nll_loss(
            unc_loc, diag_var, low_rank, unc_tgt, depth_mask
        )

    # Weight coefficients
    lam_rec  = getattr(cfg, "lambda_reconstruction", 0.10)
    lam_grad = getattr(cfg, "lambda_gradient", 0.30 if climatology is None else 0.10)
    lam_unc  = getattr(cfg, "lambda_uncertainty", 0.10)

    total = (
        lam_anom * l_anomaly
        + lam_temp * l_absolute
        + lam_rec * l_recon
        + lam_grad * l_gradient
        + lam_unc * l_uncertainty
    )

    return {
        "total": total,
        "anomaly": l_anomaly,
        "absolute": l_absolute,
        "unweighted_temp_loss": unweighted_temp,
        "reconstruction": l_recon,
        "gradient": l_gradient,
        "unweighted_grad_loss": unweighted_grad,
        "uncertainty": l_uncertainty,
        "n_valid": n_valid,
    }


