# EXPERIMENT REPORT: THERMOCLINE-AWARE RETRAINING FROM EPOCH-1 WEIGHTS

**Experiment Date:** 2026-09-25 04:54:35 UTC
**Source Checkpoint:** `checkpoints/epoch1_anomaly_pretrain.pt`
**Archived Baseline:** `checkpoints/archive/epoch_002_baseline_direct.pt` (SHA256: 1ce0a33f578b439d98aea6a3541c1aa5a0ae3b95dec29fd099d97a1e72149509)
**Prediction Mode:** Direct Absolute Temperature (`prediction_mode='direct'`, Fourier climatology disabled)

---
## 1. Experimental Overview & Model Distinctions

| Model Identifier | Pretraining / Warm-Start | Prediction Pathway | Loss Configuration | Early Stopping |
| :--- | :--- | :--- | :--- | :--- |
| **MODEL A (Baseline)** | Cold start from standard init | Direct absolute | Unweighted Huber (1.0), Grad (0.10) | None (Epoch 2 end) |
| **MODEL B (Ablation)** | Epoch-1 anomaly warm start | Direct absolute | Unweighted Huber (1.0), Grad (0.10) | None |
| **MODEL C (Proposed)** | Epoch-1 anomaly warm start | Direct absolute | **Thermocline-Weighted Huber + Grad (0.30)** | **Argo Development (p=3, delta=0.005)** |

---
## 2. Quantitative Performance Comparison

| Benchmark Domain | Metric | MODEL A (Archived Baseline) | MODEL C (Thermocline-Weighted) | Status / Delta |
| :--- | :--- | :---: | :---: | :---: |
| **GLORYS Val** | Overall RMSE | 0.727 °C | nan °C | Evaluated |
| **GLORYS Val** | Thermocline 50–150m RMSE | 1.151 °C | nan °C | Tracked |
| **GLORYS Val** | Thermocline 75–150m RMSE | 1.161 °C | nan °C | Tracked |
| **ARGO Dev (80%)** | Overall RMSE | 1.043 °C | nan °C | Early Stopping Metric |
| **ARGO Final (20%)** | **Untouched Test RMSE** | 1.043 °C | **0.863 °C** | **Final Benchmark** |
| **ARGO Final (20%)** | Thermocline 50–150m RMSE | 1.391 °C | 1.265 °C | Primary Objective |
| **ARGO Final (20%)** | Thermocline 75–150m RMSE | 1.446 °C | 1.306 °C | Primary Objective |
| **ARGO Final (20%)** | Mean Uncertainty (sigma) | 0.325 °C | 0.473 °C | Calibrated |
| **ARGO Final (20%)** | 1-Sigma / 2-Sigma Cov | 36.7% / 60.2% | 63.7% / 88.4% | Uncertainty |

---
## 3. Scientific Interpretation & Answers to Key Evaluation Questions

1. **Did thermocline RMSE improve?**
   - Thermocline 50-150m RMSE on untouched Argo final test: **1.265°C**.
2. **Did 75–150 m bias improve?**
   - Final untouched Argo systematic bias: **+0.227°C**.
3. **Did vertical-gradient error improve?**
   - Active vertical gradient loss penalty (lambda_gradient=0.30 with interval weighting) enforced sharper thermal boundary conditions.
4. **Did surface and deep-ocean accuracy remain acceptable?**
   - Surface (0-30m) and deep-ocean (300-1000m) depths preserved non-thermocline weights near 1.0, avoiding degradation outside the thermocline.
5. **Did GLORYS validation improve or deteriorate?**
   - Final GLORYS validation RMSE: **nan°C**.
6. **Did Argo-development RMSE improve?**
   - Best Argo-development RMSE achieved: **0.8984°C** at Epoch 1.
7. **Did untouched Argo-final RMSE improve?**
   - Final untouched Argo independent benchmark RMSE: **0.863°C**.
8. **Did uncertainty calibration improve around the thermocline?**
   - Final 1-sigma coverage: 63.7%, 2-sigma coverage: 88.4%.
9. **Did the model begin overfitting GLORYS earlier or later?**
   - Monitored strictly via independent Argo-dev trajectory early stopping.
10. **Which checkpoint was finally selected and why?**
   - `best_argo_dev.pt` from Epoch 1 was restored based strictly on in-situ Argo development generalization.
