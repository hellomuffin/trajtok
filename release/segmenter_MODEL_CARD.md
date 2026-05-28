---
license: apache-2.0
library_name: pytorch
tags:
  - video-segmentation
  - object-grouping
  - trajectory-tokens
  - perceiver
  - dinov3
  - vision
datasets:
  - facebook/segment-anything
  - facebook/sav-dataset
pipeline_tag: image-segmentation
---

# TrajTok-v2 Segmenter

The trajectory segmenter from **TrajTok-v2** — a class-agnostic spatio-temporal
object grouper that maps an image or video clip into ≤ K (default 128)
**trajectory tokens**. Each token binds patches that belong to the same
object instance over space and time.

This checkpoint is the headline release: trained on a mixture of ~12 M
samples (SA-1B, SA-V, filtered image / video pairs) for 3 epochs on
2 nodes × 8 H100s.

## Architecture

```
video / image  (T, 3, 224, 224)
        │
        ↓
  DINOv3-small ConvNeXt  →  patch features F  (T·56·56, D=512)
        │
        ↓
  PerceiverResampler (K=128 learnable trajectory queries, depth=2)
        │
        ↓
  soft-mask assignment:  M[k, p] = softmax_k(q_k · F_p)    (paper Eq. 1)
        │
        ↓
  trajectory tokens:     z_k = Σ_p M[k, p] · F_p           (paper Eq. 2)
```

Total parameters: ~59 M.

## Intended use

- **Token-efficient visual encoders** for downstream models (e.g. VLMs):
  swap your patch-token-based vision encoder for this segmenter to get ≤ 128
  object-grounded tokens per clip instead of hundreds of grid patches.
- **Class-agnostic object proposal / tracking** for retrieval, captioning,
  or analytics pipelines that need lightweight instance grouping.
- **Starting point for fine-tuning** on specialized domains (medical,
  satellite, robotics) where you have unlabeled video.

## How to use

```python
import torch, yaml
from easydict import EasyDict as edict
from trajtok_segmenter.model.segmenter import SimpleSegmenter

# Load the released checkpoint
state = torch.load("path/to/latest.pth", map_location="cpu", weights_only=False)
sd = state["model"]
# Strip outer SegmentWrapper prefix (the training script wraps SimpleSegmenter)
sd = {k[len("vision_encoder."):] if k.startswith("vision_encoder.") else k: v for k, v in sd.items()}

# Build matching architecture
cfg = yaml.safe_load(open("trajtokv2/segmenter/configs/pretrain.yaml"))
model = SimpleSegmenter(
    config=edict(cfg["traj_model"]),
    backbone_config=edict(cfg["backbone"]),
    perceiver_config=edict(cfg["perceiver"]),
    high_res=False,
).cuda().eval()
model.load_state_dict(sd, strict=False)

# Run forward on a clip
video = torch.randn(1, 8, 3, 224, 224).cuda()         # (B, T, 3, H, W); T=1 for images
with torch.no_grad():
    logits = model(video)                              # (B, N=T·56·56, K=128)
traj_id = logits.argmax(-1)                            # per-patch trajectory ID
soft_mask = logits.softmax(-1)                         # per-patch trajectory weight
```

See the [main repository](https://github.com/hellomuffin/trajtokv2) for the
full demo (`segmenter/scripts/demo_image.py`), evaluation drivers
(DAVIS / MOSE / YT-VIS), and training code.

## Training data

The released checkpoint was trained on the `filteredmixdata_all` mixture:

| Source | Samples | Type |
|---|---|---|
| `big_image_new` | ~300 K | filtered image-caption pairs with auto-generated trajectory masks |
| `big_video_new` | ~1 M | filtered video-caption pairs with auto-generated per-frame trajectory masks |
| **SA-1B** ([Meta AI](https://ai.meta.com/datasets/segment-anything/)) | ~11 M | original SA-1B images + instance masks |
| **SA-V** ([Meta AI](https://ai.meta.com/datasets/segment-anything-video/)) | ~48 K | SA-V videos + per-frame instance masks |

Roughly 12.4 M samples in total, interleaved by media type via a MetaLoader.
The segmenter's perceiver was trained from scratch (random Fourier init); the
DINOv3-small backbone was initialised from Meta's
[DINOv3 ConvNeXt-small public release](https://github.com/facebookresearch/dinov3)
and fine-tuned end-to-end.

## Training configuration

| Knob | Value |
|---|---|
| Trajectory tokens K | 128 |
| Embedding dim | 512 |
| Backbone | DINOv3-small ConvNeXt |
| Perceiver depth | 2 |
| Input resolution | 224×224 |
| Latent grid | 56×56 |
| Loss | dice + focal (per-patch class loss) + per-patch pixel loss |
| Optimizer | AdamW (lr=1e-4, wd=0.02) |
| Schedule | cosine, 1-epoch warmup |
| Epochs | 3 |
| Per-modality batch size | image=64, video=8, sa1b=64, sav=8 |
| Hardware | 2 nodes × 8 × H100 (80 GB) |

## Limitations

- **Frame count cap**: trained at T ≤ 8 frames per clip. Longer-clip
  behaviour at inference is untested; use `merge_tracklets` from
  `trajtok_segmenter.eval.eval_segmenter` to stitch IDs across windows.
- **Resolution**: trained at 224×224 inputs producing a 56×56 trajectory
  grid. Other resolutions work but degrade quality away from this point.
- **Class-agnostic only**: outputs trajectory IDs, not class labels. Pair
  with an open-vocabulary captioner / classifier for semantic tags.
- **Domain bias**: SA-1B + SA-V are skewed towards everyday scenes; expect
  domain-shift drops on medical, satellite, or stylised content.

## Citation

```bibtex
@article{zheng2026trajtokv2,
  title   = {TrajTok-v2: Trajectory-aware visual tokenization for vision-language models},
  author  = {Zheng, Chenhao and others},
  journal = {arXiv preprint arXiv:2602.22779},
  year    = {2026},
}
```

## License

Apache-2.0. Bundled DINOv3 ConvNeXt-small backbone weights (downloaded
separately) are also Apache-2.0 (Meta AI). SA-1B and SA-V training data
are licensed under their respective terms by Meta AI.
