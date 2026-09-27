import math
import torch
import torch.nn as nn

class GeographicEncoding(nn.Module):
    """
    Encodes center latitude and longitude into an Earth geographic embedding [B, embed_dim].
    Converts (lat, lon) in degrees to radians:
      [sin(lat), cos(lat), sin(lon), cos(lon)]
    Passes through:
      Linear(4, 32) -> GELU -> Linear(32, embed_dim)
    """
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, latitude: torch.Tensor, longitude: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latitude: [B] in degrees (-90 to 90)
            longitude: [B] in degrees (-180 to 180 or 0 to 360)
        Returns:
            geo_embedding: [B, embed_dim]
        """
        lat_rad = torch.deg2rad(latitude).unsqueeze(-1)
        lon_rad = torch.deg2rad(longitude).unsqueeze(-1)
        
        feats = torch.cat([
            torch.sin(lat_rad),
            torch.cos(lat_rad),
            torch.sin(lon_rad),
            torch.cos(lon_rad),
        ], dim=-1)  # [B, 4]
        return self.mlp(feats)
