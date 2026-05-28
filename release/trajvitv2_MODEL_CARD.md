---
license: apache-2.0
library_name: pytorch
tags:
  - clip
  - video-text-retrieval
  - vision-language
  - trajectory-tokens
  - perceiver
  - vit
datasets:
  - mlfoundations/datacomp_pools
pipeline_tag: zero-shot-image-classification
---

# TrajViT-v2 (small-scale image+video pretrain)

**TrajViT-v2** wraps the [TrajTok-v2 segmenter](https://huggingface.co/michaelzch001/trajtokv2-segmenter)
with a ViT-Large transformer and trains it under a CLIP-style image-text
contrastive objective. Trajectory tokens (K=128 per clip from the segmenter)
are contextualised by a ViT-Large, then the `[CLS]` feature is contrasted
against BERT text embeddings via InfoNCE.

This release is a **proof-of-concept checkpoint** — it shows the
trajectory-CLIP architecture trains end-to-end and produces reasonable
embeddings. **It is not a final-scale model and should not be used as a
production retrieval encoder.** See the limitations section.

## Architecture

```
video / image  (T, 3, 224, 224)
     │
     ↓
SimpleSegmenter (DINOv3 + perceiver)         →  trajectory tokens  z  (K=128, D=512)
     │                                                              ↑ warm-started from
     │                                                              TrajTok-v2 segmenter
     ↓
ViT-Large transformer over z                  →  contextualised z' (K+1=129, D=512)
     │
     ↓
[CLS]-pool → vision_proj (linear) ────────────→ image embedding for InfoNCE
                                                  ↕
                                            text embedding ← BERT(caption)
```

Total parameters: ~316 M (segmenter ~59 M + ViT-Large ~303 M + BERT text
encoder ~110 M + small projections).

## Intended use

- **Sanity-check baseline** for trajectory-CLIP architectures — fine-tune on
  larger data + with better recipes to get competitive numbers.
- **Starting point for fine-tuning** on specialised retrieval or
  classification tasks.
- **Demonstration of the architecture** for research / teaching purposes.

## How to use

```python
import torch, yaml
from easydict import EasyDict as edict
from trajtok_segmenter.model.model_pretrain import SegmentCLIP
from trajtok_segmenter.text.tokenization_bert import BertTokenizer

cfg = edict(yaml.safe_load(open("trajtokv2/trajvitv2/configs/pretrain.yaml")))
tokenizer = BertTokenizer.from_pretrained(cfg.text_encoder)
model = SegmentCLIP(config=cfg, tokenizer=tokenizer).cuda().eval()

state = torch.load("path/to/latest.pth", map_location="cpu", weights_only=False)
model.load_state_dict(state["model"], strict=False)

# Encode an image (treat as T=1 clip)
import numpy as np; from PIL import Image
img = np.asarray(Image.open("cat.jpg").convert("RGB"))
# (preprocess to (1, 1, 3, 224, 224), then:)
# image_feat = model.vision_proj(model.encode_image(...)[1][:, 0])

# Encode text
tok = tokenizer(["a cat sleeping on a couch"], padding="max_length",
                truncation=True, max_length=77, return_tensors="pt").to("cuda")
text_feat = model.text_proj(model.encode_text(tok)[1][:, 0])
text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
```

Full retrieval / zero-shot scripts: see
[trajvitv2/scripts/](https://github.com/hellomuffin/trajtokv2/tree/main/trajvitv2/scripts).

## Training data

**`filteredmixdata_new`** — a small image+video mixture (~1.3 M samples):

| Source | Samples | Type |
|---|---|---|
| `big_image_new` | ~300 K | filtered image-caption pairs |
| `big_video_new` | ~1 M | filtered video-caption pairs |

**Important caveat**: this is a much smaller corpus than what production
CLIP-style models (CLIP, OpenCLIP, SigLIP) train on (typically 100 M – 5 B
pairs). The numbers this checkpoint achieves on standard benchmarks will be
correspondingly lower.

## Training configuration

| Knob | Value |
|---|---|
| Vision backbone | DINOv3-small ConvNeXt (init from Meta) |
| Trajectory transformer | ViT-Large (random init, our `CustomTransformer` wrapper) |
| Text encoder | bert-base-uncased |
| Trajectory tokens K | 128 |
| Embedding dim (segmenter) | 512 |
| Contrastive embedding dim | 256 |
| Input resolution | 224×224 |
| Loss | InfoNCE (`loss_weight.itc=1.0`) + segmentation loss (`loss_weight.icl=1.0`) jointly |
| Optimizer | AdamW (lr=1e-4, wd=0.02) |
| Schedule | cosine, 1-epoch warmup |
| Epochs | 20 |
| Per-modality batch size | image=64, video=8 |
| Hardware | 1 node × 8 × H100 (80 GB) |
| Segmenter warm-start | from TrajTok-v2 released segmenter ckpt |

## Limitations

- **Small-scale training data** (~1.3 M pairs) — expect retrieval R@1 and
  zero-shot ImageNet numbers well below CLIP / SigLIP. This is a research
  proof-of-concept, not a production model.
- **Frame count cap**: trained at T ≤ 8 frames per clip.
- **BERT text encoder** is a deliberate choice for parity with the original
  segmenter training pipeline; modern alternatives (T5, OpenCLIP's text
  encoder) would likely lift retrieval performance.
- **The ViT-Large transformer is trained from scratch** (random init) —
  initialising from a pretrained ViT and adapting it to trajectory tokens
  could close some of the gap to large-scale baselines.

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

Apache-2.0. The wrapped DINOv3 and BERT components retain their original
Apache-2.0 / Apache-2.0 licenses respectively.
