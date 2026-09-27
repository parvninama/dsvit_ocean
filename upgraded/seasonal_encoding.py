import math
import torch
import torch.nn as nn

class SeasonalEncoding(nn.Module):
    """
    Encodes normalized seasonal time [B] in [0, 1) into a continuous periodic embedding [B, embed_dim].
    Uses 1st and 2nd harmonics of the annual cycle:
      phase = 2 * pi * seasonal_time
      features = [sin(phase), cos(phase), sin(2*phase), cos(2*phase)]
    followed by an MLP: Linear(4, 32) -> GELU -> Linear(32, embed_dim)
    """
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 32, num_harmonics: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_harmonics = num_harmonics
        in_dim = 2 * num_harmonics
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, seasonal_time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seasonal_time: [B] or [B, 1] scalar values in [0, 1) representing (day_of_year - 1) / 365.25
        Returns:
            seasonal_embedding: [B, embed_dim]
        """
        if seasonal_time.dim() == 2:
            seasonal_time = seasonal_time.squeeze(-1)
        phase = 2.0 * math.pi * seasonal_time.unsqueeze(-1)  # [B, 1]
        
        feats = []
        for h in range(1, self.num_harmonics + 1):
            feats.append(torch.sin(h * phase))
            feats.append(torch.cos(h * phase))
        feat_tensor = torch.cat(feats, dim=-1)  # [B, 2 * num_harmonics]
        return self.mlp(feat_tensor)
