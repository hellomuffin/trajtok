import math
from typing import Tuple

import torch
from torch import nn
from einops import rearrange

# =============================================================================
# If you have Meta-AI’s official “segment-anything” repo installed, the following
# import will succeed.  Otherwise, make sure that the TwoWayTransformer class
# from the repo’s `modeling/mask_decoder.py` is in your PYTHONPATH.
# =============================================================================
from trajtok_segmenter.model.two_way_attention import TwoWayTransformer
from einops import repeat
# -----------------------------------------------------------------------------#
# Helper: a very light 2-D LayerNorm (mirrors the one used in SAM’s decoder)
# -----------------------------------------------------------------------------#
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B,C,H,W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


# -----------------------------------------------------------------------------#
# Patch-decoder: 2× → 4× spatial up-scaling (conv-transposed) as in SAM-small
# -----------------------------------------------------------------------------#

class SamPatchDecoder(nn.Module):
    """
    Converts (B, T·H·W, D) → (B, T·H′·W′, D_out)
    where H′ = W′ = output_res and  D_out = embed_dim//8  (32 for SAM-small).

    `output_res / latent_res` must be 1, 2 or 4.
    """

    def __init__(self, *, embed_dim: int, latent_res: int, output_res: int):
        super().__init__()
        self.latent_res = latent_res
        self.output_res = output_res
        self.out_dim    = embed_dim // 8

        scale = output_res // latent_res
        assert scale in {1, 2, 4, 8}, "scale must be 1, 2, or 4, 8"

        layers = []

        # First 2× up-scale if needed
        if scale >= 2:
            layers += [
                nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2),
                LayerNorm2d(embed_dim // 2),
                nn.GELU(),
            ]
            ch = embed_dim // 2
        else:
            ch = embed_dim

        # Second 2× up-scale if needed (overall 4×)
        if scale >= 4:
            layers += [
                nn.ConvTranspose2d(ch, ch // 2, 2, 2),
                nn.GELU(),
            ]
            ch = ch // 2

        if scale == 8:
            layers += [
                nn.ConvTranspose2d(ch, ch // 2, 2, 2),
                nn.GELU(),
            ]
            ch = ch // 2
            

        # Final 1×1 to reach 32 channels for hyper-network dot-product
        layers += [nn.Conv2d(ch, self.out_dim, 1), nn.GELU()]
        self.conv_up = nn.Sequential(*layers)

    # --------------------------------------------------------------------- #
    def forward(self, tokens: torch.Tensor, T: int) -> torch.Tensor:
        """
        tokens : (B, T·H·W, D)
        returns: (B, T·H′·W′, 32)
        """
        B, N, D = tokens.shape
        H = W = self.latent_res
        assert N == T * H * W, "Token count mismatch with (T, H, W)"

        # (B, T, H, W, D) → (B*T, D, H, W)
        x = tokens.view(B, T, H, W, D).permute(0, 1, 4, 2, 3).reshape(B*T, D, H, W)

        # Up-scale spatially (time is a batch dimension here)
        x = self.conv_up(x)                      # (B*T, 32, H′, W′)

        # (B*T, 32, H′, W′) → (B, T, H′, W′, 32)
        H2, W2 = x.shape[-2:]
        x = x.view(B, T, self.out_dim, H2, W2).permute(0, 1, 3, 4, 2)

        # Flatten back to sequence
        return x.reshape(B, T * H2 * W2, self.out_dim)

# -----------------------------------------------------------------------------#
# Trajectory “perceiver” – thin wrapper around SAM’s Two-Way Transformer
# -----------------------------------------------------------------------------#




class SamTrajectoryPerceiver(nn.Module):
    """
    • `num_latents` learnable queries encode video-long trajectories.
    • All `T·H·W` patch tokens attend to all queries in a **single** call.
    • Output:
        traj_features  – (B, M, D)    (query tokens after decoding)
        patch_features – (B, T·H·W, D) (updated video patch tokens)
    """

    def __init__(
        self,
        *,
        embed_dim:   int = 256,
        latent_res:  int,            # H == W of backbone grid per frame
        num_frames:  int,            # T
        num_latents: int = 256,      # M
        depth:       int = 2,
        num_heads:   int = 8,
        mlp_dim:     int = 2048,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.latent_res  = latent_res          # H = W
        self.num_frames  = num_frames          # T
        self.num_latents = num_latents

        # ------------------------------------------------------------- #
        # Learnable trajectory queries (M × D)
        # ------------------------------------------------------------- #
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))

        # ------------------------------------------------------------- #
        # Learnable positional encoding for the entire (T,H,W) grid
        # Shape:  (1, D,  T·H,  W)
        # ------------------------------------------------------------- #
        total_h = num_frames * latent_res
        self.image_pe = nn.Parameter(torch.zeros(1, embed_dim, total_h, latent_res))

        # Two-Way Transformer from SAM
        self.decoder = TwoWayTransformer(
            depth                     = depth,
            embedding_dim             = embed_dim,
            num_heads                 = num_heads,
            mlp_dim                   = mlp_dim,
            attention_downsample_rate = 2,
        )

    # --------------------------------------------------------------------- #
    def forward(self, patch_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        patch_tokens : (B, T·H·W, D)

        Returns
        -------
        traj_features  : (B, M, D)
        patch_features : (B, T·H·W, D)
        """
        B, N, D = patch_tokens.shape
        H = W = self.latent_res
        T = self.num_frames
        assert N == T * H * W, "Token count mismatch – check T, H, W."

        # ----------------------------------------------------------------- #
        # 1.  Reshape tokens into a pseudo-image:   B, D,  (T·H),  W
        # ----------------------------------------------------------------- #
        img = (
            patch_tokens
              .view(B, T, H, W, D)        # B, T, H, W, D
              .permute(0, 4, 1, 2, 3)     # B, D, T, H, W
              .reshape(B, D, T * H, W)    # B, D, T·H, W
        )

        # ----------------------------------------------------------------- #
        # 2.  Prepare queries (B, M, D)
        # ----------------------------------------------------------------- #
        queries = repeat(self.latents, "m d -> b m d", b=B)

        # ----------------------------------------------------------------- #
        # 3.  Two-Way Transformer (single call)
        # ----------------------------------------------------------------- #
        queries, keys = self.decoder(
            image_embedding = img,             # B, D, T·H, W
            image_pe        = self.image_pe,   # 1, D, T·H, W
            point_embedding = queries,         # B, M, D
        )
        # keys: (B, T·H·W, D)   after internal flatten

        return queries, keys                   # traj_features, patch_features
    
    

# -----------------------------------------------------------------------------#
# Trajectory → mask-hypernetwork head
# -----------------------------------------------------------------------------#
def make_traj_seg_head(d_in: int, d_out: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_in, d_in),
        nn.ReLU(),
        nn.Linear(d_in, d_out),
    )





