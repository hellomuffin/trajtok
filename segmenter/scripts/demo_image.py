"""Run the trajectory segmenter on a single image, dump trajectory masks + overlay.

Usage:
    python scripts/demo_image.py --image path/to/img.jpg \
                                 --ckpt checkpoints/segmenter_filteredmixdata_all.pth \
                                 --output_dir demo_out
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import yaml
from easydict import EasyDict as edict

from trajtok_segmenter.model.segmenter import SimpleSegmenter


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize_normalize(img: np.ndarray, image_res: int = 224) -> torch.Tensor:
    """img: (H, W, 3) uint8 -> (1, 3, image_res, image_res) float tensor."""
    pil = Image.fromarray(img).resize((image_res, image_res), Image.BICUBIC)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = arr.transpose(2, 0, 1)[None]                  # (1, 3, H, W)
    return torch.from_numpy(arr)


def _palette(K: int) -> np.ndarray:
    """Distinct colors for K trajectories via HSV → RGB."""
    import colorsys
    rng = np.random.RandomState(0)
    hues = rng.permutation(K) / K
    rgb = np.array([colorsys.hsv_to_rgb(h, 0.85, 0.95) for h in hues])
    return (rgb * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--ckpt", required=True, help="Segmenter checkpoint .pth")
    parser.add_argument("--config", default=None, help="YAML config (defaults to configs/pretrain.yaml)")
    parser.add_argument("--output_dir", default="./demo_out", help="Where to dump outputs")
    parser.add_argument("--image_res", type=int, default=224)
    parser.add_argument("--num_traj", type=int, default=128, help="K trajectories (must match ckpt)")
    parser.add_argument("--alpha", type=float, default=0.55, help="Overlay alpha for mask vs image")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load config + build model ----
    if args.config is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.config = os.path.join(here, "..", "configs", "pretrain.yaml")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    traj_cfg = edict(cfg["traj_model"])
    bb_cfg = edict(cfg["backbone"])
    per_cfg = edict(cfg["perceiver"])
    traj_cfg.num_traj = args.num_traj

    model = SimpleSegmenter(
        config=traj_cfg, backbone_config=bb_cfg, perceiver_config=per_cfg,
        high_res=False,
    ).cuda().eval()

    # ---- Load checkpoint ----
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "model" in sd:
        sd = sd["model"]
    # Strip outer SegmentWrapper prefix if present.
    sd = {k[len("vision_encoder."):] if k.startswith("vision_encoder.") else k: v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f"loaded ckpt: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")

    # ---- Run forward ----
    raw_img = np.asarray(Image.open(args.image).convert("RGB"))
    x = _resize_normalize(raw_img, image_res=args.image_res).cuda()      # (1, 3, H, W)
    with torch.no_grad():
        logits = model(x.unsqueeze(1))                                    # (1, N=56*56, K=128)
    # Pick assignment per patch
    assign = logits.argmax(dim=-1).squeeze(0).cpu().numpy()               # (56*56,)
    H = W = int(np.sqrt(assign.size))
    assign = assign.reshape(H, W)                                         # (56, 56)

    # Upsample assignment back to image_res using nearest-neighbour
    assign_up = np.asarray(
        Image.fromarray(assign.astype(np.uint8)).resize(
            (args.image_res, args.image_res), Image.NEAREST,
        )
    )

    # Colorise + overlay
    palette = _palette(args.num_traj)
    color = palette[assign_up]                                            # (H, W, 3)
    base = np.asarray(
        Image.fromarray(raw_img).resize((args.image_res, args.image_res), Image.BICUBIC)
    ).astype(np.float32)
    overlay = (args.alpha * color + (1 - args.alpha) * base).astype(np.uint8)

    out_base = os.path.splitext(os.path.basename(args.image))[0]
    Image.fromarray(color).save(os.path.join(args.output_dir, f"{out_base}__mask.png"))
    Image.fromarray(overlay).save(os.path.join(args.output_dir, f"{out_base}__overlay.png"))
    np.save(os.path.join(args.output_dir, f"{out_base}__assignment.npy"), assign_up)

    print(f"Wrote {args.output_dir}/{out_base}__{{mask,overlay,assignment}}.{{png,png,npy}}")
    print(f"  unique trajectories used: {len(np.unique(assign))} / {args.num_traj}")


if __name__ == "__main__":
    main()
