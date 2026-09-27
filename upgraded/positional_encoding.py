import math
import torch
import torch.nn as nn

class LocalPositionalEncoding2D(nn.Module):
    """
    2D Positional Encoding for the 6x6 latent spatial grid (36 tokens).
    Encodes spatial positions relative to the center of the grid, ensuring
    directional awareness (north/south/east/west/center) for the Spatial Transformer.
    """
    def __init__(self, embed_dim: int = 128, height: int = 6, width: int = 6):
        super().__init__()
        self.embed_dim = embed_dim
        self.height = height
        self.width = width
        pe = self._make_pe(embed_dim, height, width)
        self.register_buffer("pe", pe)

    @staticmethod
    def _make_pe(embed_dim: int, height: int, width: int) -> torch.Tensor:
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")
        d_half = embed_dim // 2
        div_term = torch.exp(torch.arange(0, d_half, 2).float() * (-math.log(10000.0) / d_half))

        # Center coordinates so (0,0) is at the center of the patch
        row_center = (height - 1.0) / 2.0
        col_center = (width - 1.0) / 2.0
        rows = (torch.arange(height).float() - row_center).unsqueeze(1)  # [H, 1]
        cols = (torch.arange(width).float() - col_center).unsqueeze(1)   # [W, 1]

        pe_row = torch.zeros(height, d_half)
        pe_col = torch.zeros(width, d_half)

        pe_row[:, 0::2] = torch.sin(rows * div_term)
        pe_row[:, 1::2] = torch.cos(rows * div_term)
        pe_col[:, 0::2] = torch.sin(cols * div_term)
        pe_col[:, 1::2] = torch.cos(cols * div_term)

        pe_2d = torch.zeros(height, width, embed_dim)
        pe_2d[:, :, :d_half] = pe_row.unsqueeze(1).expand(-1, width, -1)
        pe_2d[:, :, d_half:] = pe_col.unsqueeze(0).expand(height, -1, -1)
        return pe_2d.view(height * width, embed_dim)  # [36, embed_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 36, embed_dim]
        Returns:
            x + pe: [B, 36, embed_dim]
        """
        assert x.shape[1] == self.height * self.width, f"Expected {self.height * self.width} tokens, got {x.shape[1]}"
        return x + self.pe.unsqueeze(0)
