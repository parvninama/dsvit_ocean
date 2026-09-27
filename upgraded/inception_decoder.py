import torch
import torch.nn as nn
import torch.nn.functional as F
from upgraded.inception_encoder import InceptionBlock

class InceptionCNNDecoder(nn.Module):
    """
    Parallel surface reconstruction decoder.
    Receives CNN latent representation [B, 128, 6, 6] and reconstructs the 7 physical surface variables [B, 7, 11, 11].
    Uses bilinear upsampling and lightweight Inception-style processing.
    Its output is used strictly for an auxiliary reconstruction loss and never fed into the Transformer.
    """
    def __init__(self, embed_dim: int = 128, out_ch: int = 7, target_size: int = 11):
        super().__init__()
        self.target_size = target_size

        # Inception processing after upsampling: 128 -> 64
        self.block1 = InceptionBlock(embed_dim, 64)

        # Refinement conv: 64 -> 32
        self.refine = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        # Final projection to 7 physical surface channels
        self.out_conv = nn.Conv2d(32, out_ch, kernel_size=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: [B, 128, 6, 6]
        Returns:
            recon: [B, 7, 11, 11]
        """
        # Bilinear upsample from 6x6 to target 11x11
        x = F.interpolate(latent, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        h1 = self.block1(x)         # [B, 64, 11, 11]
        h2 = self.refine(h1)        # [B, 32, 11, 11]
        recon = self.out_conv(h2)   # [B, 7, 11, 11]
        return recon
