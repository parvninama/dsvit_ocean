import torch
import torch.nn as nn
import numpy as np

TARGET_DEPTHS_DEFAULT = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

class DepthEncoder(nn.Module):
    """
    Encodes physical depth levels (in metres) into continuous physical depth embeddings [n_depths, embed_dim].
    Uses normalized actual physical depths (log1p scaling relative to maximum depth)
    fed through:
      Linear(1, 32) -> GELU -> Linear(32, embed_dim)
    """
    def __init__(self, embed_dim: int = 128, depths: list = None, mlp_hidden: int = 32):
        super().__init__()
        if depths is None:
            depths = TARGET_DEPTHS_DEFAULT
        self.depths = depths
        self.n_depths = len(depths)
        depths_arr = np.array(depths, dtype=np.float32)
        
        # Log1p normalization to capture logarithmic physical depth spacing
        log_depths = np.log1p(depths_arr)
        max_log = log_depths[-1] if log_depths[-1] > 0 else 1.0
        normalized_depths = log_depths / max_log  # in [0, 1]
        
        self.register_buffer("depth_vals", torch.from_numpy(normalized_depths))
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def forward(self) -> torch.Tensor:
        """
        Returns:
            depth_embeddings: [n_depths, embed_dim]
        """
        return self.mlp(self.depth_vals.unsqueeze(-1))
