from __future__ import annotations
import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any

@dataclass
class UpgradedConfig:
    # Model type identifier
    model_type: str = "upgraded"

    # Dataset directories
    data_dir: str = "/Users/parvninama/Programs/SIH/dsvit_ocean/bob_ocean_dataset_2years/samples_monthly"
    results_dir: str = "results"
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"
    climatology_path: str = "checkpoints/climatology.npz"

    # Target depths in metres (15 levels)
    target_depths: List[float] = field(default_factory=lambda: [
        0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000
    ])
    n_depths: int = 15
    n_surface_vars: int = 7
    patch_size: int = 11

    # Domain definition
    lat_min: float = 5.0
    lat_max: float = 30.0
    lon_min: float = 45.0
    lon_max: float = 105.0
    resolution: float = 0.25

    # Chronological partition
    train_frac: float = 0.81
    val_frac: float = 0.09
    test_frac: float = 0.10
    accum_steps: int = 1
    max_samples: Optional[int] = None
    resume_checkpoint: Optional[str] = None

    # Model Architecture
    embed_dim: int = 128
    covariance_rank: int = 3

    # Inception CNN Encoder & Decoder
    use_inception_encoder: bool = True
    use_reconstruction: bool = True
    latent_spatial: int = 6
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 96])
    cnn_kernel_size: int = 3

    # Spatial Vision Transformer
    spatial_n_layers: int = 3
    spatial_n_heads: int = 4
    spatial_mlp_ratio: float = 2.0
    spatial_dropout: float = 0.10

    # Vertical Transformer
    vertical_n_layers: int = 2
    vertical_n_heads: int = 4
    vertical_mlp_ratio: float = 2.0
    vertical_dropout: float = 0.10

    # Encodings
    use_geographic_encoding: bool = True
    use_seasonal_encoding: bool = True
    use_vertical_attention: bool = True
    use_anomaly_target: bool = True
    use_correlated_uncertainty: bool = True
    use_gradient_loss: bool = True

    # 5-Component Loss Weights
    lambda_anomaly: float = 1.0
    lambda_absolute: float = 0.5
    lambda_reconstruction: float = 0.10
    lambda_gradient: float = 0.10
    lambda_uncertainty: float = 0.10
    huber_delta: float = 1.0

    # Training Hyperparameters
    seed: int = 42
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5
    grad_clip: float = 1.0
    num_workers: int = 4

    # Early Stopping
    early_stop_patience: int = 8
    early_stop_metric: str = "val_rmse_temp"

    # Runtime Diagnostics
    plot_every_n_epochs: int = 1
    log_shape_check: bool = True
    device: str = "auto"
    fp16: bool = False
    pin_memory: bool = False

    # Debug & Overfit Modes
    sanity_n_samples: int = 4
    overfit_n_samples: int = 256
    overfit_n_epochs: int = 100


def load_upgraded_config(yaml_path: Optional[str] = None) -> UpgradedConfig:
    cfg = UpgradedConfig()
    if yaml_path is not None and os.path.isfile(yaml_path):
        with open(yaml_path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                # Store unknown attributes dynamically
                setattr(cfg, key, value)
    return cfg


def save_upgraded_config(cfg: UpgradedConfig, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as fh:
        yaml.dump(asdict(cfg), fh, default_flow_style=False)
