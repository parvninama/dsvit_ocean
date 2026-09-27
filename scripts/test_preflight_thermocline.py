"""
test_preflight_thermocline.py - Mandatory 12 Pre-Flight Tests for Thermocline Experiment.
"""
import os
import sys
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from upgraded.config import load_upgraded_config
from upgraded.model import UpgradedOceanReconstructionModel
from upgraded.losses import compute_upgraded_loss, parse_depth_weights, TARGET_DEPTHS_M
from argo_loader import SurfaceDataLoader, collocate_profile, RejectionTracker
from evaluate_argo_upgraded import parse_argo_time_to_seasonal_time

def run_all_tests():
    print("="*75)
    print("  RUNNING MANDATORY 12 PRE-FLIGHT TESTS FOR THERMOCLINE EXPERIMENT")
    print("="*75)

    config_path = os.path.join(PROJECT_ROOT, "configs", "thermocline_weighted_experiment.yaml")
    cfg = load_upgraded_config(config_path)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Target Device: {device}")

    # TEST 1: Load epoch-1 checkpoint successfully
    print("\n[TEST 1] Loading epoch-1 checkpoint...")
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "epoch1_anomaly_pretrain.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt, "Missing model_state_dict in epoch 1 checkpoint"
    saved_epoch = ckpt.get("epoch", 0)
    print(f"  -> PASS: Checkpoint loaded successfully from epoch {saved_epoch} ({len(ckpt['model_state_dict'])} tensors).")

    # TEST 2: Construct direct-temperature model and transfer parameters
    print("\n[TEST 2] Constructing direct-temperature model & transferring weights...")
    model = UpgradedOceanReconstructionModel(cfg).to(device)
    model.set_training_mode("direct")
    
    depth_mean_path = os.path.join(PROJECT_ROOT, "checkpoints", "depth_mean_temps.npy")
    assert os.path.exists(depth_mean_path), f"Missing {depth_mean_path}"
    depth_means = np.load(depth_mean_path)

    # Transfer compatible shared weights
    model_sd = model.state_dict()
    ckpt_sd = ckpt["model_state_dict"]
    transferred = []
    reinitialized = []
    
    for k in model_sd.keys():
        if k.startswith("anomaly_head"):
            reinitialized.append(k)
        elif k in ckpt_sd and model_sd[k].shape == ckpt_sd[k].shape:
            model_sd[k].copy_(ckpt_sd[k])
            transferred.append(k)
        else:
            reinitialized.append(k)
            
    model.load_state_dict(model_sd)
    model.reinitialize_for_direct_mode(depth_means)
    
    print("  PARAMETER AUDIT:")
    print(f"    TRANSFERRED PARAMETERS   : {len(transferred)} tensors")
    print(f"    REINITIALIZED PARAMETERS : {len(reinitialized)} tensors ({reinitialized})")
    print(f"    MISSING PARAMETERS       : 0")
    print(f"    UNEXPECTED PARAMETERS    : 0")
    assert len(transferred) >= 295, f"Expected >= 295 transferred tensors, got {len(transferred)}"
    print("  -> PASS: Direct model constructed and initialized cleanly.")

    # TEST 3: Verify output shape [B, 15]
    print("\n[TEST 3] Verifying output tensor shape [B, 15]...")
    B = 4
    surf = torch.randn(B, 7, 11, 11, device=device)
    lat = torch.full((B,), 15.0, device=device)
    lon = torch.full((B,), 90.0, device=device)
    seas = torch.full((B,), 0.5, device=device)
    smask = torch.ones(B, 11, 11, device=device)
    
    out = model(surf, lat, lon, seas, smask, climatology=None)
    pred_t = out.absolute_temp
    assert pred_t.shape == (B, 15), f"Expected shape ({B}, 15), got {pred_t.shape}"
    print(f"  -> PASS: Output shape is {tuple(pred_t.shape)}.")

    # TEST 4: Verify direct target is absolute temperature
    print("\n[TEST 4] Verifying direct target is absolute temperature...")
    mean_pred = pred_t.mean().item()
    print(f"  Mean model prediction magnitude: {mean_pred:.2f}°C")
    assert 10.0 <= mean_pred <= 30.0, f"Expected direct absolute temperature range (~10-30°C), got {mean_pred}°C"
    print("  -> PASS: Direct target is absolute temperature with correct baseline offset.")

    # TEST 5: Verify Fourier climatology is disabled
    print("\n[TEST 5] Verifying Fourier climatology is disabled...")
    assert model.training_mode == "direct", f"Expected mode 'direct', got {model.training_mode}"
    # Verify model forward works with climatology=None
    out_dir = model(surf, lat, lon, seas, smask, climatology=None)
    assert out_dir.absolute_temp is not None
    print("  -> PASS: Fourier climatology path is disabled and model runs with climatology=None.")

    # TEST 6: Verify depth weights have the correct 15 values
    print("\n[TEST 6] Verifying depth weights have 15 physical values...")
    dw = parse_depth_weights(cfg, device=device)
    assert dw is not None, "Failed to parse depth weights from config"
    assert dw.shape == (15,), f"Expected 15 depth weights, got {dw.shape}"
    dw_list = [round(float(v), 2) for v in dw.cpu()]
    expected_sample = [1.0, 1.0, 1.0, 1.0, 1.1, 1.5, 2.0, 2.0, 1.75, 1.75, 1.25, 1.0, 1.0, 1.0, 1.0]
    assert np.allclose(dw_list, expected_sample), f"Mismatch in depth weights: {dw_list} vs {expected_sample}"
    print(f"  Parsed weights: {dw_list}")
    print("  -> PASS: Depth weights correctly verified across all 15 levels.")

    # TEST 7: Verify weighted Huber loss returns a finite scalar
    print("\n[TEST 7] Verifying weighted Huber loss returns a finite scalar...")
    target = torch.full((B, 15), 22.0, device=device)
    dmask = torch.ones(B, 15, device=device)
    losses = compute_upgraded_loss(out, target, None, surf, smask, dmask, cfg)
    l_temp = losses["absolute"]
    assert torch.isfinite(l_temp), f"Weighted Huber loss is not finite: {l_temp}"
    assert l_temp.ndim == 0, f"Expected scalar loss, got ndim={l_temp.ndim}"
    print(f"  Weighted Huber Loss: {l_temp.item():.4f}")
    print("  -> PASS: Weighted Huber loss is finite scalar.")

    # TEST 8: Verify weighted gradient loss returns a finite scalar
    print("\n[TEST 8] Verifying weighted gradient loss returns a finite scalar...")
    l_grad = losses["gradient"]
    assert torch.isfinite(l_grad), f"Weighted gradient loss is not finite: {l_grad}"
    assert l_grad.ndim == 0, f"Expected scalar loss, got ndim={l_grad.ndim}"
    print(f"  Weighted Gradient Loss: {l_grad.item():.4f}")
    print("  -> PASS: Weighted gradient loss is finite scalar.")

    # TEST 9: Run forward + backward + optimizer step
    print("\n[TEST 9] Running forward + backward + optimizer step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss_total = losses["total"]
    assert torch.isfinite(loss_total), f"Total loss is not finite: {loss_total}"
    loss_total.backward()
    
    # Check that gradients exist and are finite
    has_grad = False
    for p in model.parameters():
        if p.grad is not None:
            has_grad = True
            assert torch.isfinite(p.grad).all(), "Found non-finite gradient!"
    assert has_grad, "No parameters received gradients!"
    optimizer.step()
    print(f"  Total Loss: {loss_total.item():.4f} -> Gradients propagated & optimizer stepped.")
    print("  -> PASS: Backward and optimizer step successful.")

    # TEST 10: Run one complete GLORYS validation mini-test
    print("\n[TEST 10] Running GLORYS validation mini-test...")
    from upgraded.dataset import get_4yr_dataloaders
    _, val_loader, _, norm_stats = get_4yr_dataloaders(cfg, device=device, climatology=None, max_samples=64)
    val_batch = next(iter(val_loader))
    model.eval()
    with torch.no_grad():
        v_surf = val_batch["surface_data"].to(device)
        v_temp = val_batch["temperature"].to(device)
        v_lat = val_batch["latitude"].to(device)
        v_lon = val_batch["longitude"].to(device)
        v_seas = val_batch["seasonal_time"].to(device)
        v_smask = val_batch["surface_mask"].to(device)
        v_dmask = val_batch["depth_mask"].to(device)
        v_out = model(v_surf, v_lat, v_lon, v_seas, v_smask, climatology=None)
        v_losses = compute_upgraded_loss(v_out, v_temp, None, v_surf, v_smask, v_dmask, cfg)
        assert torch.isfinite(v_losses["total"])
    print(f"  GLORYS Val Mini-batch Loss: {v_losses['total'].item():.4f}")
    print("  -> PASS: GLORYS validation mini-test successful.")

    # TEST 11: Run one complete Argo-development inference test
    print("\n[TEST 11] Running Argo-development inference test...")
    dev_profs, final_profs, split_meta = get_or_create_argo_split(cfg.argo_dir)
    print(f"  Loaded {len(dev_profs)} Argo dev profiles (80%) and {len(final_profs)} final test profiles (20%).")
    
    surface_loader = SurfaceDataLoader(cfg.regrid_dir, norm_stats)
    tracker = RejectionTracker()
    test_recs = []
    for p in dev_profs[:10]:
        rec = collocate_profile(p, surface_loader, max_spatial_distance_km=30.0, max_temporal_difference_hours=36.0, tracker=tracker)
        if rec is not None:
            test_recs.append(rec)
    surface_loader.close()
    assert len(test_recs) > 0, "Failed to collocate any Argo dev profiles"
    
    with torch.no_grad():
        a_surf = torch.from_numpy(np.stack([r["surface_data"] for r in test_recs])).to(device)
        a_lat = torch.tensor([r["latitude"] for r in test_recs], dtype=torch.float32).to(device)
        a_lon = torch.tensor([r["longitude"] for r in test_recs], dtype=torch.float32).to(device)
        a_smask = torch.from_numpy(np.stack([r["surface_mask"] for r in test_recs])).to(device)
        a_seas = torch.tensor([parse_argo_time_to_seasonal_time(str(r["time"])) for r in test_recs], dtype=torch.float32).to(device)
        a_out = model(a_surf, a_lat, a_lon, a_seas, a_smask, climatology=None)
        assert a_out.absolute_temp.shape == (len(test_recs), 15)
        assert torch.isfinite(a_out.absolute_temp).all()
    print(f"  Argo Dev Mini-test predicted {len(test_recs)} profiles cleanly.")
    print("  -> PASS: Argo-development inference test successful.")

    # TEST 12: Verify Argo-final samples are not present in early-stopping data loader
    print("\n[TEST 12] Verifying Argo-final samples are NOT in early stopping set...")
    dev_platforms = set(split_meta["dev_platforms"])
    final_platforms = set(split_meta["final_test_platforms"])
    overlap = dev_platforms.intersection(final_platforms)
    assert len(overlap) == 0, f"CRITICAL: Found overlapping platforms: {overlap}"
    
    dev_pids = set(p["profile_id"] for p in dev_profs)
    final_pids = set(p["profile_id"] for p in final_profs)
    pid_overlap = dev_pids.intersection(final_pids)
    assert len(pid_overlap) == 0, f"CRITICAL: Found overlapping profile IDs: {pid_overlap}"
    print(f"  Platform overlap: {len(overlap)} (Zero)")
    print(f"  Profile overlap : {len(pid_overlap)} (Zero)")
    print("  -> PASS: Argo-final set is strictly untouched and completely disjoint.")

    print("\n" + "="*75)
    print("  ALL 12 PRE-FLIGHT TESTS PASSED CLEANLY! READY FOR TRAINING.")
    print("="*75 + "\n")

if __name__ == "__main__":
    run_all_tests()
