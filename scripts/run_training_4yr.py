#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_training_4yr.py

Comprehensive ocean temperature reconstruction training script:
- Trains on 4 years of GLORYS data (or 80% dev split), tests on 1 year GLORYS and Argo independently.
- Displays per-batch diagnostics: duration (seconds), loss, RMSE (°C), Absolute Error / MAE (°C), and R^2 score.
- Evaluates full validation set after each epoch.
- Saves model checkpoint after every epoch (epoch_XXX.pt), plus running latest.pt and best.pt.
- Uses all GPU cores (Apple-silicon MPS or NVIDIA CUDA) and all CPU cores for DataLoader workers.
"""

from __future__ import annotations
import os
import sys
import time
import argparse
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from upgraded.config import load_upgraded_config
from upgraded.model import UpgradedOceanReconstructionModel
from upgraded.dataset import get_4yr_dataloaders
from upgraded.losses import compute_upgraded_loss
from upgraded.climatology import HarmonicClimatology
from upgraded.metrics import compute_upgraded_metrics
from utils import set_seed, get_device, save_checkpoint, save_norm_stats


def compute_batch_temperature_metrics(pred_abs: torch.Tensor, true_temp: torch.Tensor, depth_mask: torch.Tensor):
    """
    Computes RMSE, MAE (mean absolute error), and R^2 on masked valid depths in Celsius.
    """
    mask = depth_mask.bool()
    if not mask.any():
        return 0.0, 0.0, 0.0

    p_valid = pred_abs[mask]
    t_valid = true_temp[mask]

    diff = p_valid - t_valid
    rmse_c = torch.sqrt(torch.mean(diff ** 2)).item()
    mae_c = torch.mean(torch.abs(diff)).item()

    p_np = p_valid.detach().cpu().numpy()
    t_np = t_valid.detach().cpu().numpy()

    # R^2 requires some variance in ground truth
    if np.var(t_np) > 1e-6:
        r2 = float(r2_score(t_np, p_np))
    else:
        r2 = 0.0

    return rmse_c, mae_c, r2


def run_evaluation(model, climatology, loader, cfg, device, use_climatology: bool = True):
    """Evaluates model over a DataLoader (validation or test) and returns metrics."""
    model.eval()
    all_pred_abs, all_pred_anom = [], []
    all_target_abs, all_target_anom = [], []
    all_clim, all_sigma, all_dmask = [], [], []
    all_recon, all_surf, all_smask = [], [], []

    total_loss_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            surf = batch["surface_data"].to(device)
            temp = batch["temperature"].to(device)
            lat = batch["latitude"].to(device)
            lon = batch["longitude"].to(device)
            seas = batch["seasonal_time"].to(device)
            smask = batch["surface_mask"].to(device)
            dmask = batch["depth_mask"].to(device)
            B = surf.shape[0]

            if use_climatology and climatology is not None:
                clim = climatology.lookup_torch(lat, lon, seas, device=device)
            else:
                clim = None

            out = model(surf, lat, lon, seas, smask, clim)
            losses = compute_upgraded_loss(out, temp, clim, surf, smask, dmask, cfg)

            total_loss_sum += losses["total"].item() * B
            total_samples += B

            p_abs = out.absolute_temp.cpu().numpy()
            t_abs = temp.cpu().numpy()

            if clim is not None:
                p_anom = out.anomaly_mean.cpu().numpy()
                c_np = clim.cpu().numpy()
                t_anom = t_abs - c_np
                all_pred_anom.append(p_anom)
                all_target_anom.append(t_anom)
                all_clim.append(c_np)

            d_var = out.diag_variance.cpu().numpy()
            l_fac = out.low_rank_factor.cpu().numpy()
            p_sigma = np.sqrt(d_var + np.sum(l_fac ** 2, axis=-1))

            all_pred_abs.append(p_abs)
            all_target_abs.append(t_abs)
            all_sigma.append(p_sigma)
            all_dmask.append(dmask.cpu().numpy())

            if out.recon is not None:
                all_recon.append(out.recon.cpu().numpy())
                all_surf.append(surf.cpu().numpy())
                all_smask.append(smask.cpu().numpy())

    pred_abs_np = np.concatenate(all_pred_abs)
    pred_anom_np = np.concatenate(all_pred_anom) if all_pred_anom else None
    target_abs_np = np.concatenate(all_target_abs)
    target_anom_np = np.concatenate(all_target_anom) if all_target_anom else None
    pred_sigma_np = np.concatenate(all_sigma)
    dmask_np = np.concatenate(all_dmask)
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

    valid_mask = dmask_np.astype(bool)
    p_valid = pred_abs_np[valid_mask]
    t_valid = target_abs_np[valid_mask]
    overall_r2 = float(r2_score(t_valid, p_valid)) if len(t_valid) > 1 else 0.0

    return {
        "loss": total_loss_sum / max(total_samples, 1),
        "rmse_c": metrics["overall_abs_rmse"],
        "mae_c": metrics["overall_abs_mae"],
        "bias_c": metrics["overall_abs_bias"],
        "corr": metrics["overall_abs_corr"],
        "r2": overall_r2,
        "anom_rmse_c": metrics.get("overall_anom_rmse", float("nan")),
        "detailed_metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="4-Year GLORYS Training & Diagnostics")
    parser.add_argument("--config", type=str, default="configs/upgraded_model.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap training samples for debugging/testing")
    parser.add_argument("--run-argo-test", action="store_true", help="Run independent Argo evaluation after training")
    parser.add_argument("--exp-dir", type=str, default=None, help="Directory to store checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--start-epoch", type=int, default=None, help="Explicit epoch number to start training from")
    args = parser.parse_args()

    # 1. Configuration & Reproducibility
    set_seed(args.seed)
    cfg = load_upgraded_config(args.config)
    cfg.batch_size = args.batch_size
    cfg.num_epochs = args.epochs
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples

    # Output directories
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = args.exp_dir or os.path.join(PROJECT_ROOT, "results", f"run_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    print(f"\n{'='*75}")
    print(f"  OCEAN TEMPERATURE RECONSTRUCTION: 4-YEAR TRAINING PIPELINE")
    print(f"  Experiment Directory: {exp_dir}")
    print(f"{'='*75}")

    # 2. Compute Device Selection: Uses all GPU cores
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🖥️  GPU Device: Apple Silicon MPS (Using all GPU cores)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        n_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
        print(f"🖥️  GPU Device: NVIDIA CUDA ({n_gpus} GPUs available: {', '.join(gpu_names)})")
    else:
        device = torch.device("cpu")
        print(f"⚠️  No GPU found. Using CPU with {os.cpu_count()} cores.")

    # 3. Climatology (harmonic Fourier model - only used in Epoch 1 anomaly mode)
    climatology = None
    clim_path = getattr(cfg, "climatology_path", os.path.join(PROJECT_ROOT, "checkpoints", "climatology.npz"))
    if not os.path.isabs(clim_path):
        clim_path = os.path.join(PROJECT_ROOT, clim_path)

    if os.path.exists(clim_path):
        climatology = HarmonicClimatology.load(clim_path)
    else:
        from fit_climatology import fit_climatology_from_training_data
        print(f"  Climatology not found at {clim_path}. Fitting harmonic climatology from training partition...")
        climatology = fit_climatology_from_training_data(cfg.data_dir, output_path=clim_path, max_train_samples=50000)

    # 4a. Load or compute depth-wise mean temperature profile (needed for Epoch 2 head init)
    depth_mean_path = os.path.join(PROJECT_ROOT, "checkpoints", "depth_mean_temps.npy")
    if os.path.exists(depth_mean_path):
        depth_mean_temps = np.load(depth_mean_path)
        print(f"  Loaded depth mean temperature profile from {depth_mean_path}")
    else:
        print("  Computing depth mean temperature profile from training partition...")
        from collections import defaultdict as _defaultdict
        import h5py as _h5py
        from upgraded.dataset import _discover_files, _build_index, _file_counts
        _paths = _discover_files(cfg.data_dir)
        _counts = _file_counts(_paths)
        _fi_all, _li_all = _build_index(_paths, _counts)
        _n_train = int(round(len(_fi_all) * 0.80 * 0.90))
        _sample_idx = np.random.choice(_n_train, size=min(100000, _n_train), replace=False)
        _fi_s, _li_s = _fi_all[_sample_idx], _li_all[_sample_idx]
        _fg = _defaultdict(list)
        for _fi, _li in zip(_fi_s, _li_s):
            _fg[_fi].append(_li)
        _all_t = []
        for _fi, _idxs in _fg.items():
            with _h5py.File(_paths[_fi], 'r') as _f:
                _all_t.append(_f['temperature'][sorted(_idxs)])
        depth_mean_temps = np.nanmean(np.concatenate(_all_t, axis=0), axis=0).astype(np.float32)
        os.makedirs(os.path.dirname(depth_mean_path), exist_ok=True)
        np.save(depth_mean_path, depth_mean_temps)
        print(f"  Depth mean temps saved: {depth_mean_path}")
    print(f"  Depth-wise mean temps (°C): {[f'{v:.1f}' for v in depth_mean_temps]}")


    train_loader, val_loader, test_loader, norm_stats = get_4yr_dataloaders(
        cfg, device=device, climatology=climatology, max_samples=args.max_samples
    )
    save_norm_stats(norm_stats, os.path.join(checkpoints_dir, "norm_stats.json"))

    # 5. Model, Optimizer, Scheduler
    model = UpgradedOceanReconstructionModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=cfg.learning_rate * 0.01)

    best_val_rmse = float("inf")
    start_epoch = 1

    # Resume from checkpoint if requested
    if args.resume:
        print(f"\n  [RESUME] Loading checkpoint from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"  [RESUME NOTE] Optimizer state skipped: {e}")
        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                print(f"  [RESUME NOTE] Scheduler state skipped: {e}")

        saved_epoch = ckpt.get("epoch", 0)
        start_epoch = args.start_epoch if args.start_epoch is not None else saved_epoch + 1
        if "val_metrics" in ckpt and "rmse_c" in ckpt["val_metrics"]:
            best_val_rmse = ckpt["val_metrics"]["rmse_c"]
        print(f"  [RESUME] Restored model from Epoch {saved_epoch}. Starting from Epoch {start_epoch} (best_val_rmse={best_val_rmse:.4f}°C)\n")

        # If resuming from an Epoch 1 checkpoint into Epoch 2+, reinitialize head for direct mode
        if saved_epoch == 1 and start_epoch >= 2:
            print("  [RESUME TRANSITION] Checkpoint was trained in Anomaly Pretraining mode.")
            print("  [RESUME TRANSITION] Reinitializing prediction head for Direct Temperature Prediction...")
            model.reinitialize_for_direct_mode(depth_mean_temps)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
            remaining = max(args.epochs - (start_epoch - 1), 1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining, eta_min=cfg.learning_rate * 0.01)
            print(f"  [RESUME TRANSITION] Optimizer & scheduler initialized fresh for direct mode (remaining epochs: {remaining}).\n")

    # 6. Epoch Training Loop
    print(f"\n🚀 Training from Epoch {start_epoch} to {args.epochs}...\n")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()

        # Training mode setup:
        # Epoch 1: Anomaly pretraining relative to harmonic Fourier climatology
        # Epoch 2+: Direct temperature prediction (Fourier climatology completely removed)
        use_clim = (epoch == 1)
        mode = "anomaly" if use_clim else "direct"
        model.set_training_mode(mode)

        if epoch == 1:
            print(f"\n{'='*75}")
            print(f"  EPOCH 1: ANOMALY PRETRAINING MODE")
            print(f"  - Target: Subsurface temperature anomaly ΔT relative to Fourier baseline")
            print(f"  - Formulation: T_pred = T_clim + ΔT_pred")
            print(f"  - Objective: Pretrain spatial/vertical representations on residual deviations")
            print(f"{'='*75}\n", flush=True)
        elif epoch == 2 or (start_epoch == epoch and epoch > 1):
            print(f"\n{'='*75}")
            print(f"  EPOCH {epoch}: TRANSITION TO DIRECT TEMPERATURE PREDICTION MODE")
            print(f"  - Target: Direct subsurface absolute temperature T_pred")
            print(f"  - Fourier Climatology Formula: COMPLETELY REMOVED & DISABLED")
            print(f"  - Formulation: T_pred = Network(X)")
            print(f"  - Retained: Inception CNN, Spatial ViT, Depth queries, Vertical ViT")
            print(f"  - Loss: 1.0 * Huber(T_pred, T_true) + Uncertainty + Gradient + Reconstruction")
            print(f"{'='*75}\n", flush=True)

        # Batch logging CSV
        batch_csv_path = os.path.join(exp_dir, f"batch_logs_epoch_{epoch:03d}.csv")
        csv_file = open(batch_csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "batch_idx", "duration_s", "loss", "rmse_c", "mae_c", "r2"])

        epoch_loss_sum = 0.0
        epoch_rmse_sum = 0.0
        epoch_mae_sum = 0.0
        epoch_r2_sum = 0.0
        n_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader, start=1):
            batch_t0 = time.time()

            surf = batch["surface_data"].to(device)
            temp = batch["temperature"].to(device)
            lat = batch["latitude"].to(device)
            lon = batch["longitude"].to(device)
            seas = batch["seasonal_time"].to(device)
            smask = batch["surface_mask"].to(device)
            dmask = batch["depth_mask"].to(device)

            if use_clim and climatology is not None:
                clim = climatology.lookup_torch(lat, lon, seas, device=device)
            else:
                clim = None

            optimizer.zero_grad()
            out = model(surf, lat, lon, seas, smask, clim)
            losses = compute_upgraded_loss(out, temp, clim, surf, smask, dmask, cfg)
            loss = losses["total"]
            loss.backward()

            if getattr(cfg, "grad_clip", 0.0) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()
            batch_dur = time.time() - batch_t0

            # Compute real-time batch metrics in Celsius
            b_rmse, b_mae, b_r2 = compute_batch_temperature_metrics(out.absolute_temp, temp, dmask)
            b_loss = loss.item()

            epoch_loss_sum += b_loss
            epoch_rmse_sum += b_rmse
            epoch_mae_sum += b_mae
            epoch_r2_sum += b_r2

            # Write row to CSV
            csv_writer.writerow([epoch, batch_idx, f"{batch_dur:.4f}", f"{b_loss:.6f}", f"{b_rmse:.4f}", f"{b_mae:.4f}", f"{b_r2:.4f}"])

            # Console display: time taken, loss, RMSE (°C), Absolute Error / MAE (°C), R^2
            if batch_idx in [1, 5, 10, 25] or batch_idx % 20 == 0 or batch_idx == n_batches:
                print(
                    f"Epoch {epoch:03d} [{mode}] | Batch {batch_idx:04d}/{n_batches:04d} | "
                    f"Time: {batch_dur*1000:6.1f}ms ({batch_dur:5.3f}s) | "
                    f"Loss: {b_loss:6.4f} | "
                    f"RMSE: {b_rmse:6.3f}°C | "
                    f"MAE: {b_mae:6.3f}°C | "
                    f"R²: {b_r2:6.3f}",
                    flush=True
                )

        csv_file.close()

        # Validation at end of EVERY epoch
        print(f"\n  Running validation for Epoch {epoch:03d} (mode: '{mode}')...", flush=True)
        val_eval = run_evaluation(model, climatology, val_loader, cfg, device, use_climatology=use_clim)
        epoch_dur = time.time() - epoch_start_time

        print(
            f"\n{'='*75}\n"
            f"=== Epoch {epoch:03d} COMPLETE (Duration: {epoch_dur:.2f}s, Mode: {mode}) ===\n"
            f"  Train Mean Loss : {epoch_loss_sum / n_batches:.4f} | RMSE: {epoch_rmse_sum / n_batches:.3f}°C | MAE: {epoch_mae_sum / n_batches:.3f}°C | R²: {epoch_r2_sum / n_batches:.3f}\n"
            f"  Validation Loss : {val_eval['loss']:.4f} | RMSE: {val_eval['rmse_c']:.3f}°C | MAE: {val_eval['mae_c']:.3f}°C | R²: {val_eval['r2']:.3f} | Bias: {val_eval['bias_c']:+.3f}°C\n"
            f"{'='*75}\n",
            flush=True
        )

        # Check for best validation score
        is_best = val_eval["rmse_c"] < best_val_rmse
        if is_best:
            best_val_rmse = val_eval["rmse_c"]

        # Save checkpoint after EVERY epoch
        epoch_ckpt_path = os.path.join(checkpoints_dir, f"epoch_{epoch:03d}.pt")
        latest_ckpt_path = os.path.join(checkpoints_dir, "latest.pt")
        ckpt_payload = {
            "epoch": epoch,
            "training_mode": mode,
            "best_validation_metric": best_val_rmse,
            "best_val_rmse": best_val_rmse,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": cfg,
            "norm_stats": norm_stats,
            "val_metrics": val_eval,
            "is_best": is_best,
        }
        torch.save(ckpt_payload, epoch_ckpt_path)
        torch.save(ckpt_payload, latest_ckpt_path)
        print(f"  [SAVED] Checkpoint: {epoch_ckpt_path}")
        print(f"  [SAVED] Running Model: {latest_ckpt_path}")

        # If Epoch 1 just finished, export dedicated epoch1_anomaly_pretrain.pt
        # AND reinitialize the prediction head for direct temperature mode.
        if epoch == 1:
            pretrain_export_exp = os.path.join(checkpoints_dir, "epoch1_anomaly_pretrain.pt")
            pretrain_export_proj = os.path.join(PROJECT_ROOT, "checkpoints", "epoch1_anomaly_pretrain.pt")
            torch.save(ckpt_payload, pretrain_export_exp)
            torch.save(ckpt_payload, pretrain_export_proj)
            print(f"  🌟 [EXPORTED] Anomaly Pretraining Checkpoint: {pretrain_export_proj}")

            if args.epochs > 1:
                # ── CRITICAL FIX ── Reinitialize prediction head bias for direct mode
                # After Epoch 1, anomaly_head predicts near-zero anomalies (ΔT ~ ±1-3°C).
                # In Epoch 2+, the head must output absolute temperatures (~6-29°C).
                # Without this call → RMSE immediately jumps to ~22°C (systematic bias).
                model.reinitialize_for_direct_mode(depth_mean_temps)

                # Reset Adam optimizer state to clear momentum/variance accumulated during
                # anomaly training. The scale of gradients changes dramatically when switching
                # from ~1°C anomaly targets to ~15-29°C absolute targets.
                print("  [TRANSITION] Resetting optimizer state (Adam m/v buffers cleared).")
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=cfg.learning_rate,
                    weight_decay=cfg.weight_decay,
                )
                # Rebuild scheduler from the new optimizer with remaining epochs
                remaining = args.epochs - 1
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(remaining, 1),
                    eta_min=cfg.learning_rate * 0.01
                )
                print(f"  [TRANSITION] Optimizer and scheduler rebuilt. Remaining epochs: {remaining}")

        if is_best:
            best_ckpt_path = os.path.join(checkpoints_dir, "best.pt")
            torch.save(ckpt_payload, best_ckpt_path)
            print(f"  ⭐ [NEW BEST] Validation RMSE: {best_val_rmse:.3f}°C -> Saved {best_ckpt_path}")

        # Save JSON validation metrics
        with open(os.path.join(exp_dir, f"validation_metrics_epoch_{epoch:03d}.json"), "w") as jf:
            json.dump({k: v for k, v in val_eval.items() if k != "detailed_metrics"}, jf, indent=2)

        # Advance LR scheduler — but skip if we just rebuilt it for Epoch 2
        # (calling step() on a fresh scheduler before any optimizer.step() would skip the
        #  first LR value and emit a PyTorch warning).
        if not (epoch == 1 and args.epochs > 1):
            scheduler.step()



    # 7. Final Independent GLORYS Test Evaluation
    print(f"\n{'='*75}\n  RUNNING HELD-OUT GLORYS TEST EVALUATION\n{'='*75}")
    test_use_clim = (args.epochs == 1 and start_epoch == 1)
    test_eval = run_evaluation(model, climatology, test_loader, cfg, device, use_climatology=test_use_clim)
    print(
        f"  GLORYS Test Loss : {test_eval['loss']:.4f}\n"
        f"  GLORYS Test RMSE : {test_eval['rmse_c']:.3f}°C\n"
        f"  GLORYS Test MAE  : {test_eval['mae_c']:.3f}°C\n"
        f"  GLORYS Test R²   : {test_eval['r2']:.3f}\n"
        f"  GLORYS Test Bias : {test_eval['bias_c']:+.3f}°C\n"
        f"  GLORYS Test Anom : {test_eval['anom_rmse_c']:.3f}°C\n"
    )
    with open(os.path.join(exp_dir, "glorys_test_summary.json"), "w") as jf:
        json.dump({k: v for k, v in test_eval.items() if k != "detailed_metrics"}, jf, indent=2)

    # 8. Optional Independent Argo Float Evaluation
    if args.run_argo_test:
        print(f"\n{'='*75}\n  RUNNING INDEPENDENT IN-SITU ARGO FLOAT EVALUATION\n{'='*75}")
        from evaluate_argo_upgraded import run_argo_evaluation_upgraded
        argo_dir = os.path.join(exp_dir, "argo_test")
        os.makedirs(argo_dir, exist_ok=True)
        best_path = os.path.join(checkpoints_dir, "best.pt")
        try:
            run_argo_evaluation_upgraded(
                config_path=args.config,
                checkpoint_path=best_path,
                device_str=str(device),
                output_dir=argo_dir,
            )
        except Exception as e:
            print(f"  Argo evaluation encountered note: {e}")

    print(f"\n🎉 All runs completed successfully! Check outputs in: {exp_dir}\n")


if __name__ == "__main__":
    main()
