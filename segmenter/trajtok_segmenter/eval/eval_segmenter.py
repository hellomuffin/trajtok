"""Reusable helpers for evaluating + visualising the trajectory segmenter.

For end-to-end benchmark evaluation, use the standalone scripts under
`segmenter/scripts/`:

    scripts/eval_davis.py     # DAVIS-2017 val (Hungarian-matched VEQ + STQ_EN)
    scripts/eval_mose.py      # MOSE val
    scripts/eval_ytvis.py     # YouTube-VIS 2019 / 2021 val

This module exposes three utilities those scripts (and downstream users) can
import directly:

  * :func:`save_pca_feature_maps` — PCA-RGB visualisation of patch features
    (useful for qualitative figures showing what the backbone "sees" before
    trajectory grouping).
  * :func:`merge_tracklets` — link trajectory IDs across consecutive clips
    when running on videos longer than the segmenter's per-forward window
    (default 16 frames). Uses cosine similarity + Hungarian matching.
  * :func:`downsample_segmentation_probs` — convenience: bilinear-downsample
    a soft-assignment volume and argmax to integer IDs.

The bulky research-grade driver (with hard-coded benchmark paths and a
MODEL_CLS_DICT) that lived here pre-OSS has been removed; see the standalone
scripts above instead.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------- #
# 1. PCA feature-map visualisation
# --------------------------------------------------------------------------- #

class _PyrHead(nn.Module):
    """Compress each ResNet-50 stage to 64ch -> upsample to full size -> concat."""

    def __init__(self, in_chs: Tuple[int, int, int, int] = (256, 512, 1024, 2048), out_each: int = 64):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, out_each, kernel_size=1, bias=False) for c in in_chs])

    def forward(self, feats, out_hw):
        outs = []
        for i, key in enumerate(("layer1", "layer2", "layer3", "layer4")):
            x = self.proj[i](feats[key])                                # (B, 64, h_i, w_i)
            x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
            outs.append(x)
        return torch.cat(outs, dim=1)                                    # (B, 256, H, W)


@torch.no_grad()
def save_pca_feature_maps(
    imgs: torch.Tensor,
    out_dir: str,
    min_hw: int = 512,
    filenames: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
    pca_fit_hw: int = 128,
):
    """Save per-image PCA-RGB visualisations of ResNet-50 patch features.

    The output is the first 3 principal components of a multi-scale ResNet-50
    feature pyramid, mapped to RGB. Useful for qualitative figures showing
    "what the network sees" before any trajectory grouping is applied.

    Args:
        imgs: (B, 3, H, W) ImageNet-normalised image tensor.
        out_dir: directory to dump PNGs into.
        min_hw: ensures inputs are upsampled to at least this many pixels per
            side before feature extraction (improves PCA stability on tiny
            crops).
        filenames: per-image output filenames (defaults to 000.png, 001.png, ...).
        device: cuda/cpu (auto-detected if None).
        pca_fit_hw: spatial size used to fit the PCA basis (full-res projection
            uses a 1×1 conv, so this only affects basis quality vs. compute).
    """
    assert imgs.ndim == 4 and imgs.size(1) == 3, "imgs must be (B, 3, H, W)"
    os.makedirs(out_dir, exist_ok=True)
    B, _, H, W = imgs.shape

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tgtH, tgtW = max(H, min_hw), max(W, min_hw)
    if (tgtH, tgtW) != (H, W):
        imgs = F.interpolate(imgs, size=(tgtH, tgtW), mode="bilinear", align_corners=False)

    # Lazy-import torchvision so the segmenter package doesn't hard-require it.
    from torchvision import models
    from torchvision.models.feature_extraction import create_feature_extractor

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    backbone = models.resnet50(weights=weights)
    extractor = create_feature_extractor(
        backbone, return_nodes={f"layer{i}": f"layer{i}" for i in (1, 2, 3, 4)},
    ).to(device).eval()
    pyr = _PyrHead().to(device).eval()
    imgs = imgs.to(device, non_blocking=True)

    for i in range(B):
        x = imgs[i:i + 1]
        feats = extractor(x)
        full = pyr(feats, out_hw=(tgtH, tgtW))                          # (1, 256, H, W)

        # PCA basis fit on a downsampled copy
        fit_map = F.interpolate(full, size=(pca_fit_hw, pca_fit_hw), mode="bilinear", align_corners=False)
        C = fit_map.size(1)
        X_fit = fit_map.permute(0, 2, 3, 1).reshape(-1, C)               # (N_fit, C)
        mean_c = X_fit.mean(dim=0, keepdim=True)
        _U, _S, V = torch.pca_lowrank(X_fit - mean_c, q=min(3, C))
        k = min(3, V.shape[1])
        V3 = V[:, :k]

        # Project full-res features via a 1×1 conv: W = V3^T, b = -V3^T·mean
        proj = nn.Conv2d(C, k, kernel_size=1, bias=True).to(device)
        with torch.no_grad():
            proj.weight.copy_(V3.t().unsqueeze(-1).unsqueeze(-1))
            proj.bias.copy_(-(V3.t() @ mean_c.squeeze(0)))
        scores = proj(full).squeeze(0).cpu().numpy()                     # (k, H, W)

        # Per-channel min-max → uint8 RGB
        viz = []
        for ch in range(k):
            c = scores[ch]
            mn, mx = float(c.min()), float(c.max())
            viz.append((c - mn) / (mx - mn) if mx > mn else np.zeros_like(c))
        while len(viz) < 3:
            viz.append(np.zeros_like(viz[0]))
        rgb_u8 = (np.stack(viz[:3], axis=-1) * 255.0 + 0.5).astype(np.uint8)

        name = filenames[i] if (filenames and i < len(filenames)) else f"{i:03d}.png"
        Image.fromarray(rgb_u8).save(os.path.join(out_dir, name))


# --------------------------------------------------------------------------- #
# 2. Tracklet linking across clips
# --------------------------------------------------------------------------- #

def merge_tracklets(
    predicted_label_map: torch.Tensor,
    all_traj_features: list,
    cur_traj_features: torch.Tensor,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Stitch trajectory IDs across consecutive segmenter forwards.

    The segmenter operates on bounded-length clips (default 8 or 16 frames).
    For long videos you forward each clip separately and then need to link
    "trajectory 5 in clip B" to "trajectory 12 in clip A" if they describe
    the same object. We do that with cosine-similarity Hungarian matching
    over the trajectory-query features.

    Args:
        predicted_label_map: (n_frame, h, w) int trajectory IDs in the CURRENT
            clip. IDs in the current clip start at N (the cumulative count of
            previously-seen trajectories); after this function they're rewritten
            to point at matched IDs in the global numbering.
        all_traj_features: list of (num_traj, D) tensors — trajectory-query
            features from PRIOR clips, in original (unnormalised) form.
        cur_traj_features: (num_traj, D) tensor from the CURRENT clip.
        threshold: minimum cosine similarity for a match (drop weaker links).
    """
    if len(all_traj_features) == 0:
        return predicted_label_map

    all_feat = torch.cat(all_traj_features, dim=0)                       # (N_prev, D)
    N = all_feat.shape[0]

    all_norm = F.normalize(all_feat, dim=-1)
    cur_norm = F.normalize(cur_traj_features, dim=-1)
    sim = cur_norm @ all_norm.T                                          # (n, N_prev)

    row_idx, col_idx = linear_sum_assignment((-sim).cpu().numpy())
    for i, j in zip(row_idx, col_idx):
        if threshold <= 0 or sim[i, j] >= threshold:
            predicted_label_map[predicted_label_map == int(i) + N] = int(j)
    return predicted_label_map


# --------------------------------------------------------------------------- #
# 3. Downsample soft-assignment volume and argmax
# --------------------------------------------------------------------------- #

def downsample_segmentation_probs(
    prob: torch.Tensor,
    scale_factor: float,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    """Bilinear-downsample a per-pixel class-probability map then argmax to IDs.

    Args:
        prob: (T, W, H, C) probability/logit tensor in **channel-last** layout.
        scale_factor: passed to F.interpolate (e.g. 0.5 for 2× downsample).

    Returns:
        (T, W*scale_factor, H*scale_factor) int64 tensor of argmaxed class IDs.
    """
    prob_4d = prob.permute(0, 3, 2, 1).contiguous()                      # (T, C, H, W)
    prob_low = F.interpolate(prob_4d, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    seg_low = prob_low.argmax(dim=1).to(torch.int64)                     # (T, H', W')
    return seg_low.permute(0, 2, 1)                                      # (T, W', H')
