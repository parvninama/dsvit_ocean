import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    """
    Lightweight Inception-style multi-branch block.
    Branches:
      Branch A: 1x1 conv
      Branch B: 1x1 conv -> 3x3 conv
      Branch C: 1x1 conv -> 3x3 conv -> 3x3 conv
      Branch D: 3x3 avg pool -> 1x1 conv
    Features concatenated across branches with residual connection.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        b_ch = out_ch // 4
        rem = out_ch - b_ch * 3  # Ensure exact sum matches out_ch

        # Branch A: 1x1 conv
        self.branch_a = nn.Sequential(
            nn.Conv2d(in_ch, b_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(b_ch),
            nn.GELU(),
        )

        # Branch B: 1x1 conv -> 3x3 conv
        self.branch_b = nn.Sequential(
            nn.Conv2d(in_ch, b_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(b_ch),
            nn.GELU(),
            nn.Conv2d(b_ch, b_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(b_ch),
            nn.GELU(),
        )

        # Branch C: 1x1 conv -> 3x3 conv -> 3x3 conv
        self.branch_c = nn.Sequential(
            nn.Conv2d(in_ch, b_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(b_ch),
            nn.GELU(),
            nn.Conv2d(b_ch, b_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(b_ch),
            nn.GELU(),
            nn.Conv2d(b_ch, b_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(b_ch),
            nn.GELU(),
        )

        # Branch D: 3x3 avg pool -> 1x1 conv
        self.branch_d = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, rem, kernel_size=1, bias=False),
            nn.BatchNorm2d(rem),
            nn.GELU(),
        )

        # Residual shortcut
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_a = self.branch_a(x)
        out_b = self.branch_b(x)
        out_c = self.branch_c(x)
        out_d = self.branch_d(x)
        concat = torch.cat([out_a, out_b, out_c, out_d], dim=1)
        res = self.shortcut(x)
        return self.act(concat + res)


class InceptionCNNEncoder(nn.Module):
    """
    Inception-inspired CNN Encoder for 11x11 ocean surface patches.
    Input: [B, 7, 11, 11] physical channels (+ optional 1-channel surface_mask -> [B, 8, 11, 11])
    Block 1: channels ~ 32, spatial size = 11x11
    Spatial reduction: 11x11 -> 6x6
    Block 2: channels ~ 64, spatial size = 6x6
    Block 3: channels ~ 96, spatial size = 6x6
    Final 1x1 projection: 96 -> 128
    Output: [B, 128, 6, 6] latent spatial representation
    """
    def __init__(self, in_ch: int = 8, embed_dim: int = 128, cnn_channels: list = None, kernel_size: int = 3):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 96]
        c1, c2, c3 = cnn_channels[0], cnn_channels[1], cnn_channels[2]
        self.in_ch = in_ch
        self.embed_dim = embed_dim
        self.cnn_channels = cnn_channels
        self.kernel_size = kernel_size

        # Block 1 (11x11)
        self.block1 = InceptionBlock(in_ch, c1)

        # Major spatial reduction: 11x11 -> 6x6
        pad = kernel_size // 2
        self.downsample = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=kernel_size, stride=2, padding=pad, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
        )

        # Block 2 (6x6)
        self.block2 = InceptionBlock(c1, c2)

        # Block 3 (6x6)
        self.block3 = InceptionBlock(c2, c3)

        # Final projection: c3 -> embed_dim
        self.proj = nn.Sequential(
            nn.Conv2d(c3, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, surface_data: torch.Tensor, surface_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            surface_data: [B, 7, 11, 11]
            surface_mask: [B, 11, 11] or None
        Returns:
            latent: [B, 128, 6, 6]
        """
        if self.in_ch == 8:
            if surface_mask is not None:
                mask_ch = surface_mask.unsqueeze(1).float()  # [B, 1, 11, 11]
            else:
                mask_ch = torch.ones_like(surface_data[:, :1, :, :])
            x = torch.cat([surface_data, mask_ch], dim=1)  # [B, 8, 11, 11]
        else:
            x = surface_data

        h1 = self.block1(x)              # [B, 32, 11, 11]
        h_down = self.downsample(h1)      # [B, 32, 6, 6]
        h2 = self.block2(h_down)          # [B, 64, 6, 6]
        h3 = self.block3(h2)              # [B, 96, 6, 6]
        out = self.proj(h3)               # [B, 128, 6, 6]
        return out
