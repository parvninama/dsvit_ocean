# Saved Models Directory

This directory contains the 3 primary milestone models preserved for evaluation, deployment, and benchmarking.

---

## 1. 2-Year Bay of Bengal Baseline Model (`bay_of_bengal_baseline/`)
- **File:** `model.pt` (SHA256: `1ce0a33f578b439d98aea6a3541c1aa5a0ae3b95dec29fd099d97a1e72149509`)
- **Training Mode:** Direct absolute temperature mode, trained on 2-year Bay of Bengal GLORYS monthly data (Epoch 2 checkpoint).
- **Validation RMSE (GLORYS):** 0.727 °C
- **Untouched Argo Test RMSE:** 1.043 °C
- **Thermocline 50-150m RMSE:** 1.391 °C

---

## 2. Trained Anomaly Pretrained Model (`anomaly_pretrained/`)
- **File:** `model.pt` (SHA256: `e2d5fecd02aa718ba34fd070efb8961ce5c27672b8f04ba827ed2ba778504e82`)
- **Training Mode:** Anomaly pretraining mode with Fourier climatology prior (Epoch 1 checkpoint).
- **Role:** Foundational representation weights used as warm-start for downstream thermocline-aware fine-tuning.

---

## 3. Overnight Champion Model (`overnight_champion/`)
- **File:** `model.pt` (SHA256: `d6c63b5c9f2ad7325851cacfa6f080686cc249b683d45127ea0cdfcb7c8993af`)
- **Configuration:** `config.yaml`
- **Full Benchmark Metrics:** `metrics.json`
- **Detailed Scientific Report:** `report.md`
- **Diagnostic Plots:** `plots/`
- **Architecture Highlights:**
  - 3 Vertical Transformer Layers (increased depth dedicated to vertical ocean stratification)
  - Peak Thermocline Weighting (3.5x loss penalty at 75m and 100m bottleneck depths)
  - Learning Rate: 7e-5
  - Vertical Gradient Loss Weight: 0.45
  - Huber Delta: 1.0
- **Untouched Independent Argo Final Benchmark:**
  - **Overall RMSE:** **0.863 °C** (improved by -0.180 °C from baseline)
  - **Thermocline 50-150m RMSE:** **1.265 °C** (improved by -0.126 °C)
  - **Thermocline 75-150m RMSE:** **1.306 °C** (improved by -0.140 °C)
  - **Overall MAE:** **0.530 °C**
  - **Systematic Bias:** **+0.227 °C** (reduced by 32%)
  - **Correlation:** **0.994**
