import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, NamedTuple

from upgraded.inception_encoder import InceptionCNNEncoder
from upgraded.inception_decoder import InceptionCNNDecoder
from upgraded.positional_encoding import LocalPositionalEncoding2D
from upgraded.geographic_encoding import GeographicEncoding
from upgraded.seasonal_encoding import SeasonalEncoding
from upgraded.depth_encoding import DepthEncoder

class TransformerBlock(nn.Module):
    """Pre-LN Transformer Encoder Block with Multi-Head Self-Attention and MLP."""
    def __init__(self, dim: int = 128, n_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class UpgradedModelOutput:
    """
    Standard output container for UpgradedOceanReconstructionModel.
    Supports attribute access, dict indexing, and tuple unpacking.
    """
    def __init__(
        self,
        anomaly_mean: torch.Tensor,
        absolute_temp: torch.Tensor,
        diag_variance: torch.Tensor,
        low_rank_factor: torch.Tensor,
        recon: Optional[torch.Tensor] = None,
    ):
        self.anomaly_mean = anomaly_mean
        self.absolute_temp = absolute_temp
        self.diag_variance = diag_variance
        self.low_rank_factor = low_rank_factor
        self.recon = recon

    def __iter__(self):
        return iter((self.anomaly_mean, self.absolute_temp, self.diag_variance, self.low_rank_factor, self.recon))

    def __getitem__(self, item):
        return getattr(self, item)

    def keys(self):
        return ["anomaly_mean", "absolute_temp", "diag_variance", "low_rank_factor", "recon"]


class UpgradedOceanReconstructionModel(nn.Module):
    """
    Upgraded Ocean Surface-to-Subsurface Temperature Reconstruction Model.

    Scientific formulation:
      1. Single-day 7-channel 11x11 surface data (+ surface mask)
      2. Inception-style multi-scale CNN Encoder -> [B, 128, 6, 6]
      3. Parallel Inception CNN Decoder for surface reconstruction -> [B, 7, 11, 11]
      4. 36 spatial tokens + 2D Positional Encoding + Geographic Encoding + Periodic Seasonal Encoding
      5. Prepend learnable Target Token -> [B, 37, 128]
      6. Spatial Vision Transformer (3 layers, 4 heads) -> extract target_context [B, 128]
      7. 15 Depth-conditioned tokens = target_context + depth_embeddings
      8. Vertical Transformer (2 layers, 4 heads) -> [B, 15, 128]
      9. Temperature Anomaly Head -> anomaly_mean Delta_T [B, 15]
         Absolute temperature -> T_hat = T_clim + Delta_T
     10. Correlated Uncertainty Head:
         - Positive diagonal variance D [B, 15] via Softplus
         - Low-rank covariance factor L [B, 15, 3]
         Covariance: Sigma = D + L * L^T
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        C = getattr(cfg, "embed_dim", 128)
        self.embed_dim = C
        self.covariance_rank = getattr(cfg, "covariance_rank", 3)
        self.n_depths = getattr(cfg, "n_depths", 15)
        self.target_depths = getattr(cfg, "target_depths", [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000])

        # 1. Inception CNN Encoder
        self.use_inception_encoder = getattr(cfg, "use_inception_encoder", True)
        cnn_channels = getattr(cfg, "cnn_channels", [32, 64, 96])
        cnn_kernel_size = getattr(cfg, "cnn_kernel_size", 3)
        self.cnn_encoder = InceptionCNNEncoder(in_ch=8, embed_dim=C, cnn_channels=cnn_channels, kernel_size=cnn_kernel_size)

        # 2. Parallel Inception Reconstruction Decoder
        self.use_reconstruction = getattr(cfg, "use_reconstruction", True)
        if self.use_reconstruction:
            self.cnn_decoder = InceptionCNNDecoder(embed_dim=C, out_ch=7, target_size=11)
        else:
            self.cnn_decoder = None

        # 3. Spatial Tokenization & Conditioning
        self.pos_enc = LocalPositionalEncoding2D(embed_dim=C, height=6, width=6)
        
        self.use_geographic_encoding = getattr(cfg, "use_geographic_encoding", True)
        if self.use_geographic_encoding:
            self.geo_enc = GeographicEncoding(embed_dim=C, hidden_dim=32)
        else:
            self.geo_enc = None

        self.use_seasonal_encoding = getattr(cfg, "use_seasonal_encoding", True)
        if self.use_seasonal_encoding:
            self.seasonal_enc = SeasonalEncoding(embed_dim=C, hidden_dim=32, num_harmonics=2)
        else:
            self.seasonal_enc = None

        # 4. Target Token & Spatial Transformer
        self.target_token = nn.Parameter(torch.zeros(1, 1, C))
        nn.init.trunc_normal_(self.target_token, std=0.02)

        spatial_layers = getattr(cfg, "spatial_n_layers", 3)
        spatial_heads = getattr(cfg, "spatial_n_heads", 4)
        spatial_drop = getattr(cfg, "spatial_dropout", 0.1)
        spatial_mlp_ratio = getattr(cfg, "spatial_mlp_ratio", 2.0)

        self.spatial_transformer = nn.ModuleList([
            TransformerBlock(dim=C, n_heads=spatial_heads, mlp_ratio=spatial_mlp_ratio, dropout=spatial_drop)
            for _ in range(spatial_layers)
        ])
        self.spatial_norm = nn.LayerNorm(C)

        # 5. Depth Query Conditioning & Vertical Transformer
        self.depth_encoder = DepthEncoder(embed_dim=C, depths=self.target_depths, mlp_hidden=32)

        vertical_layers = getattr(cfg, "vertical_n_layers", 2)
        vertical_heads = getattr(cfg, "vertical_n_heads", 4)
        vertical_drop = getattr(cfg, "vertical_dropout", 0.1)
        vertical_mlp_ratio = getattr(cfg, "vertical_mlp_ratio", 2.0)

        self.use_vertical_attention = getattr(cfg, "use_vertical_attention", True)
        if self.use_vertical_attention:
            self.vertical_transformer = nn.ModuleList([
                TransformerBlock(dim=C, n_heads=vertical_heads, mlp_ratio=vertical_mlp_ratio, dropout=vertical_drop)
                for _ in range(vertical_layers)
            ])
            self.vertical_norm = nn.LayerNorm(C)
        else:
            self.vertical_transformer = None
            self.vertical_norm = None

        # 6. Prediction Heads
        # Anomaly prediction head: [128 -> 64 -> 1]
        self.anomaly_head = nn.Sequential(
            nn.Linear(C, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Correlated Uncertainty Head:
        # Diagonal variance head: [128 -> 64 -> 1]
        self.diag_var_head = nn.Sequential(
            nn.Linear(C, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        # Low-rank covariance factor head: [128 -> 64 -> rank]
        self.low_rank_head = nn.Sequential(
            nn.Linear(C, 64),
            nn.GELU(),
            nn.Linear(64, self.covariance_rank),
        )

        self.training_mode = "anomaly"  # "anomaly" (Epoch 1) or "direct" (Epoch 2+)
        self._shape_logged = False

    def set_training_mode(self, mode: str):
        """
        Switch between 'anomaly' pretraining (Epoch 1) and 'direct' temperature prediction (Epoch 2+).

        In 'anomaly' mode (Epoch 1):
          The head predicts ΔT_pred (temperature anomaly).
          Forward requires Fourier climatology T_clim to compute T_abs = T_clim + ΔT_pred.

        In 'direct' mode (Epoch 2+):
          Fourier climatology is completely removed/disabled.
          The head directly predicts T_abs subsurface temperatures.
        """
        assert mode in ("anomaly", "direct"), f"Mode must be 'anomaly' or 'direct', got {mode}"
        self.training_mode = mode
        desc = "Anomaly Pretraining (ΔT + T_clim)" if mode == "anomaly" else "Direct Temperature Prediction (No Fourier Climatology)"
        print(f"  [Model] Training mode set to: '{mode}' ({desc})")

    def reinitialize_for_direct_mode(self, depth_mean_temps: "np.ndarray"):
        """
        MUST be called when transitioning from 'anomaly' mode (Epoch 1) to 'direct' mode (Epoch 2+).

        WHY THIS IS CRITICAL:
          After Epoch 1, anomaly_head's output layer has learned to produce near-zero residuals
          (typical anomaly ΔT ~ ±1-3°C). In 'direct' mode the head must predict absolute
          temperatures whose mean ranges from ~29°C at surface to ~6°C at 1000m depth.
          Without bias reinitialization, the first Epoch 2 forward pass outputs ~0°C for all depths
          → catastrophic RMSE of ~22°C immediately.

        WHAT THIS DOES:
          - Reinitializes the bias of anomaly_head's final Linear layer to the per-depth
            mean temperature profile from the training partition.
          - Leaves all other weights (encoder, ViTs, uncertainty heads) UNCHANGED so
            the pretrained representations are fully retained.
          - Also reinitializes the hidden layer (128→64) of anomaly_head with kaiming_uniform
            to give the head fresh residual-learning capacity from the correct baseline.

        Args:
            depth_mean_temps: np.ndarray of shape [15] — mean absolute temperature at each
                              depth level computed from the TRAINING partition only.
        """
        import numpy as np

        depth_means = np.asarray(depth_mean_temps, dtype=np.float32)
        assert len(depth_means) == 15, f"Expected 15 depth means, got {len(depth_means)}"

        # Re-initialize the final output bias to depth-wise mean temperatures
        final_linear = self.anomaly_head[2]  # nn.Linear(64, 1)
        with torch.no_grad():
            final_linear.bias.data.fill_(0.0)  # will be broadcast; use uniform scalar

            # Set a single scalar bias = overall mean temperature (mean over all depths)
            # The depth-specific differentiation will be learned on top of this offset.
            # This removes the ~20°C systematic DC offset immediately.
            overall_mean = float(np.mean(depth_means[depth_means > 0.0]))
            final_linear.bias.data.fill_(overall_mean)

        # Re-initialize hidden layer weights of anomaly_head to provide fresh gradients
        # while keeping the structural weights learned in Epoch 1 as the starting point.
        # We only reset anomaly_head[0] (Linear 128->64); the shared encoder blocks are KEPT.
        nn.init.kaiming_uniform_(self.anomaly_head[0].weight, nonlinearity="leaky_relu")
        nn.init.zeros_(self.anomaly_head[0].bias)

        print(
            f"  [Model] reinitialize_for_direct_mode() applied:\n"
            f"          anomaly_head output bias reset to: {overall_mean:.3f}°C (training mean)\n"
            f"          anomaly_head hidden layer (128→64) weights re-initialized (kaiming_uniform)\n"
            f"          All encoder + ViT + uncertainty head weights PRESERVED from Epoch 1."
        )

    def _log_shape(self, name: str, tensor: torch.Tensor):
        if getattr(self.cfg, "log_shape_check", False) and not self._shape_logged:
            print(f"  [shape] {name:35s}: {tuple(tensor.shape)}")

    def forward(
        self,
        surface_data: torch.Tensor,             # [B, 7, 11, 11]
        latitude: torch.Tensor,                 # [B]
        longitude: torch.Tensor,                # [B]
        seasonal_time: torch.Tensor,            # [B]
        surface_mask: Optional[torch.Tensor] = None,  # [B, 11, 11]
        climatology: Optional[torch.Tensor] = None,   # [B, 15] (only used in 'anomaly' mode)
    ) -> UpgradedModelOutput:
        B = surface_data.shape[0]
        log = self._log_shape

        # 1. Inception CNN Encoder
        latent = self.cnn_encoder(surface_data, surface_mask)  # [B, 128, 6, 6]
        log("CNN Inception latent", latent)

        # 2. Parallel Inception CNN Decoder (Reconstruction branch)
        recon = None
        if self.use_reconstruction and self.cnn_decoder is not None:
            recon = self.cnn_decoder(latent)  # [B, 7, 11, 11]
            log("Decoder reconstruction", recon)

        # 3. Spatial Tokenization: 6x6 grid -> 36 tokens
        spatial_tokens = latent.flatten(2).transpose(1, 2)  # [B, 36, 128]
        log("Spatial tokens (flattened)", spatial_tokens)

        # 2D Positional Encoding
        spatial_tokens = self.pos_enc(spatial_tokens)  # [B, 36, 128]
        log("Tokens after 2D Positional Enc", spatial_tokens)

        # Geographic Encoding
        if self.use_geographic_encoding and self.geo_enc is not None:
            geo_emb = self.geo_enc(latitude, longitude)  # [B, 128]
            spatial_tokens = spatial_tokens + geo_emb.unsqueeze(1)
            log("Tokens after Geographic Enc", spatial_tokens)

        # Seasonal Encoding (neural embedding, distinct from Fourier climatology)
        if self.use_seasonal_encoding and self.seasonal_enc is not None:
            seas_emb = self.seasonal_enc(seasonal_time)  # [B, 128]
            spatial_tokens = spatial_tokens + seas_emb.unsqueeze(1)
            log("Tokens after Seasonal Enc", spatial_tokens)

        # 4. Prepend Target Token: [B, 37, 128]
        target_tokens = self.target_token.expand(B, -1, -1)  # [B, 1, 128]
        transformer_input = torch.cat([target_tokens, spatial_tokens], dim=1)  # [B, 37, 128]
        log("Spatial Transformer input (with target token)", transformer_input)

        # Spatial Vision Transformer
        tokens = transformer_input
        for blk in self.spatial_transformer:
            tokens = blk(tokens)
        tokens = self.spatial_norm(tokens)
        log("Spatial Transformer output", tokens)

        # Extract target context token (token index 0)
        target_context = tokens[:, 0, :]  # [B, 128]
        log("Extracted Target Context token", target_context)

        # 5. Depth Token Construction: 15 depth levels
        depth_embeddings = self.depth_encoder()  # [15, 128]
        log("Depth embeddings", depth_embeddings)

        depth_tokens = target_context.unsqueeze(1) + depth_embeddings.unsqueeze(0)  # [B, 15, 128]
        log("Depth tokens (before vertical ViT)", depth_tokens)

        # 6. Vertical Transformer
        if self.use_vertical_attention and self.vertical_transformer is not None:
            for blk in self.vertical_transformer:
                depth_tokens = blk(depth_tokens)
            depth_tokens = self.vertical_norm(depth_tokens)
            log("Vertical Transformer output", depth_tokens)

        # 7. Prediction Head (Anomaly vs Direct)
        head_pred = self.anomaly_head(depth_tokens).squeeze(-1)  # [B, 15]

        if self.training_mode == "anomaly":
            # Epoch 1: Head predicts anomaly relative to climatology
            anomaly_mean = head_pred
            log("Predicted anomaly_mean", anomaly_mean)
            if climatology is None:
                raise ValueError("climatology tensor is required in 'anomaly' training mode (Epoch 1)")
            absolute_temp = climatology + anomaly_mean
            log("Absolute temperature prediction (climatology + ΔT)", absolute_temp)
        else:
            # Epoch 2+: Head directly predicts absolute temperature without Fourier climatology
            absolute_temp = head_pred
            anomaly_mean = absolute_temp  # alias so model output fields remain populated
            log("Direct absolute temperature prediction", absolute_temp)

        # 8. Correlated Uncertainty Head
        # Positive diagonal variance D with softplus + jitter
        raw_diag = self.diag_var_head(depth_tokens).squeeze(-1)  # [B, 15]
        diag_variance = F.softplus(raw_diag) + 1e-4  # [B, 15]
        log("Diagonal variance D", diag_variance)

        # Low-rank covariance factor L: [B, 15, rank]
        low_rank_factor = self.low_rank_head(depth_tokens)  # [B, 15, 3]
        log("Low-rank covariance factor L", low_rank_factor)

        self._shape_logged = True
        return UpgradedModelOutput(
            anomaly_mean=anomaly_mean,
            absolute_temp=absolute_temp,
            diag_variance=diag_variance,
            low_rank_factor=low_rank_factor,
            recon=recon,
        )

    def reset_shape_log(self):
        self._shape_logged = False
