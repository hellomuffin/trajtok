"""Phase-1 smoke test for the TrajViT backbone's forward path.

Validates that the soft-mask weighted-pool aggregation produces the
expected (B, num_traj, D) trajectory tokens, without yet requiring the
full molmo2 install. Run with:

    PYTHONPATH=reference_code/molmo2:.  /weka/.../trajvit/bin/python \
        reference_code/molmo2/tests/smoke_trajvit_forward.py
"""
import os
import sys

import torch
import torch.nn.functional as F
from einops import rearrange

# Import modules directly (avoid pulling in olmo.config)
HERE = os.path.dirname(os.path.abspath(__file__))
MOLMO2_OLMO = os.path.normpath(os.path.join(HERE, "..", "olmo", "nn"))
sys.path.insert(0, MOLMO2_OLMO)

from trajvit_dinov3 import DINOResNetHierFeat  # noqa: E402
from trajvit_perceiver import PerceiverResampler  # noqa: E402


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparams (mirror our seg pretraining)
    D = 512
    NUM_TRAJ = 128
    LATENT_RES = 56
    IMG_RES = 224
    DEPTH = 2

    # Build modules
    print("[init] DINOResNetHierFeat (small) ...", flush=True)
    backbone = DINOResNetHierFeat(model_size="small", out_channel=D, upsample_first=True).to(device).eval()
    print("[init] PerceiverResampler ...", flush=True)
    perceiver = PerceiverResampler(
        dim=D,
        depth=DEPTH,
        dim_head=64,
        heads=8,
        num_latents=NUM_TRAJ,
        use_rotary=True,
        use_latent_transformer=False,
        update_x=False,
    ).to(device).eval()

    # Dummy inputs
    cases = [
        ("single image", 1, 1),
        ("video clip (T=8)", 1, 8),
        ("two clips in batch (B=2, T=8)", 2, 8),
    ]
    for label, B, T in cases:
        print(f"\n=== {label}: B={B}, T={T} ===", flush=True)
        x = torch.randn(B * T, 3, IMG_RES, IMG_RES, device=device)

        with torch.no_grad():
            feats = backbone(x, pool="sum", output_size=(LATENT_RES, LATENT_RES))
        # feats: (B*T, D, h, w)
        print(f"  backbone feats: {tuple(feats.shape)}  (expected ({B*T}, {D}, {LATENT_RES}, {LATENT_RES}))")
        assert feats.shape == (B * T, D, LATENT_RES, LATENT_RES)

        feats = rearrange(feats, "bt d h w -> bt (h w) d").view(B, T * LATENT_RES * LATENT_RES, D)
        print(f"  flat patch features: {tuple(feats.shape)}")

        with torch.no_grad():
            q = perceiver(feats.detach())                                         # (B, K, D)
        print(f"  perceiver queries: {tuple(q.shape)}  (expected ({B}, {NUM_TRAJ}, {D}))")
        assert q.shape == (B, NUM_TRAJ, D)

        # Soft mask + pool
        qn = F.normalize(q, dim=-1)
        fn = F.normalize(feats, dim=-1)
        logits = torch.einsum("bnd,bkd->bnk", fn, qn)
        soft_mask = F.softmax(logits, dim=-1)
        z = torch.einsum("bnk,bnd->bkd", soft_mask, feats)
        print(f"  trajectory tokens z: {tuple(z.shape)}  (expected ({B}, {NUM_TRAJ}, {D}))")
        assert z.shape == (B, NUM_TRAJ, D)

        # Quick sanity: soft_mask rows sum to 1
        sm = soft_mask.sum(dim=-1)
        print(f"  soft_mask row-sum: min={sm.min().item():.4f}, max={sm.max().item():.4f}")
        assert torch.allclose(sm, torch.ones_like(sm), atol=1e-4)

    print("\n[ok] all cases passed")


if __name__ == "__main__":
    main()
