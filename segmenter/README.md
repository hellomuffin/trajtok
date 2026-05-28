# TrajTok Segmenter

📄 **Paper:** [arXiv:2602.22779](https://arxiv.org/abs/2602.22779)

The trajectory segmenter from **TrajTok-v2** — a class-agnostic spatio-temporal
object grouper that maps an image or video clip into ≤ K (default 128)
*trajectory tokens*. Each trajectory binds together patches that belong to the
same object instance over space and time.

Architecture in one diagram:

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

Released checkpoint is trained on **filteredmixdata_all** (~11 M SA-1B images
+ ~48 K SA-V videos + ~300 K big_image_new + ~1 M big_video_new, totaling ~12 M
samples) for 3 epochs on 2 nodes × 8 H100s.

## Quickstart

### 1. Install

```bash
# from the repo root
cd segmenter/
pip install -e .

# additionally download DINOv3 ConvNeXt-small weights (Apache-2.0, Meta AI):
git clone https://github.com/facebookresearch/dinov3.git
wget -P dinov3/ https://dl.fbaipublicfiles.com/dinov3/dinov3_convnext_small_pretrain_lvd1689m/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth
export TRAJTOK_DINOV3_ROOT=$PWD/dinov3
```

### 2. Get the released checkpoint

```bash
python scripts/download_ckpt.py  # → ./checkpoints/segmenter_filteredmixdata_all.pth
```

### 3. Run on one image

```bash
python scripts/demo_image.py \
  --image examples/cat.jpg \
  --ckpt checkpoints/segmenter_filteredmixdata_all.pth \
  --output_dir demo_out
# → demo_out/cat__mask.png       (color-coded trajectory map)
# → demo_out/cat__overlay.png    (alpha-blended over original)
# → demo_out/cat__assignment.npy (per-pixel trajectory IDs, 0..127)
```

### 4. Use as a library

```python
import torch
from trajtok_segmenter.model.segmenter import SimpleSegmenter
import yaml; from easydict import EasyDict as edict

cfg = yaml.safe_load(open("configs/pretrain.yaml"))
model = SimpleSegmenter(
    config=edict(cfg["traj_model"]),
    backbone_config=edict(cfg["backbone"]),
    perceiver_config=edict(cfg["perceiver"]),
    high_res=False,
).cuda().eval()

# Load the released checkpoint (strip outer SegmentWrapper prefix):
sd = torch.load("checkpoints/segmenter_filteredmixdata_all.pth", weights_only=False)["model"]
sd = {k[len("vision_encoder."):] if k.startswith("vision_encoder.") else k: v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)

video = torch.randn(1, 8, 3, 224, 224).cuda()       # (B, T, 3, H, W); image: T=1
with torch.no_grad():
    logits = model(video)                            # (B, N=T·56·56, K=128) assignment logits

# Per-patch trajectory ID:
traj_id = logits.argmax(-1)                          # (B, N)
# Soft masks (per-trajectory contribution per patch):
M = logits.softmax(-1)                               # (B, N, K)
```

## Qualitative results

Trajectory-token assignments on three real scenes. Each color = one
trajectory token; pixels with the same color get pooled into the same
`z_k`. All images below are 224×224 model inputs producing a 56×56
trajectory map (upsampled with nearest-neighbour for display).

| Input | Trajectory map | Overlay |
|---|---|---|
| ![](../assets/qual/example_breakdance_input.jpg)     | ![](../assets/qual/example_breakdance_mask.png)     | ![](../assets/qual/example_breakdance_overlay.png)     |
| ![](../assets/qual/example_dance-twirl_input.jpg)    | ![](../assets/qual/example_dance-twirl_mask.png)    | ![](../assets/qual/example_dance-twirl_overlay.png)    |
| ![](../assets/qual/example_horsejump-high_input.jpg) | ![](../assets/qual/example_horsejump-high_mask.png) | ![](../assets/qual/example_horsejump-high_overlay.png) |

Each of these crowded scenes activates ~30 distinct trajectories (out of
K=128 available) — the model adaptively allocates one token per major
object / region. Cleaner single-subject scenes use fewer (10–15). On
videos the count rises as new objects appear across frames.

*Reproduce with* `scripts/demo_image.py --image <YOUR_IMG> --ckpt <DOWNLOADED_CKPT>`.

> Source images are from the [DAVIS-2017 dataset](https://davischallenge.org)
> (CC BY 4.0). See [`../assets/qual/CREDITS.md`](../assets/qual/CREDITS.md)
> for full attribution.

## Quantitative results

Three eval drivers ship in `scripts/`:

| Benchmark | Driver | Metrics |
|---|---|---|
| DAVIS-2017 val (480p) | `scripts/eval_davis.py`  | VEQ, STQ_EN (Hungarian-matched, IoU≥0.5) |
| MOSE val              | `scripts/eval_mose.py`   | VEQ, STQ_EN |
| YT-VIS 2019/2021      | `scripts/eval_ytvis.py`  | VEQ, STQ_EN |

Run with:

```bash
python scripts/eval_davis.py \
  --ckpt checkpoints/segmenter_filteredmixdata_all.pth \
  --davis_root /path/to/DAVIS
```

Each driver writes a `metrics.json` (aggregate scores) + a `per_video.csv`
(breakdown). Numbers depend on `--num_frames`, `--image_res`, and
`--iou_thr`; defaults follow the values used in the paper.

## Training

### Data preparation

The released checkpoint was trained on:

| Source | Annotation file | Notes |
|---|---|---|
| **big_image_new** | `${TRAJTOK_DATA_ROOT}/metadata/big_images_new.json` | ~300 K filtered image-caption pairs with auto-generated trajectory masks. Caption + per-image instance masks per JSON record. |
| **big_video_new** | `${TRAJTOK_DATA_ROOT}/metadata/big_videos_new.json` | ~1 M filtered video-caption pairs with auto-generated per-frame trajectory masks. |
| **SA-1B** | `${TRAJTOK_DATA_ROOT}/sa1b/sa_*.tar` | Original SA-1B as sharded webdataset; load via `data.sa1b_dataset.SA1BDataset`. |
| **SA-V** | `${TRAJTOK_DATA_ROOT}/sav/videos_fps6/`, `${TRAJTOK_DATA_ROOT}/sav/sav_instances/` | Frames + per-frame instance polygons. |

JSON schema for `big_*` (per record):
- `image` *(or)* `video`: absolute path to media
- `caption`: text description (used only if you flip to the contrastive `simpletrajvitv2` mode; ignored for segmentation-only training)
- `mask`: per-frame run-length-encoded instance masks
- `graph`: trajectory linking IDs across frames

See `data/caption_dataset.py:ImgGraphTrainDataset` / `VidGraphTrainDataset` for
the full loader spec.

### Launch

Single-node, 8 GPUs:

```bash
TRAJTOK_DATA_ROOT=/path/to/data \
TRAJTOK_OUTPUT_DIR=/path/to/results \
TRAJTOK_DINOV3_ROOT=/path/to/dinov3 \
bash scripts/train.sh \
  --ngpus 8 \
  --train_corpus filteredmixdata_all \
  --exp_name myrun \
  --epoch 20 \
  --log_wandb
```

Multi-node (via your scheduler — set `MASTER_ADDR` + `MASTER_PORT` + `NODE_RANK`
and call `torchrun --nnodes N` directly; see `scripts/train.sh` for the args
it forwards).

### Notable flags

| Flag | Default | What it does |
|---|---|---|
| `--train_corpus` | `filteredmixdata_all` | One of the keys in `configs/pretrain.yaml:available_corpus`. Pick `filteredmixdata_new` for image+video only (no SA-1B). |
| `--num_traj` | `128` | Number of trajectory tokens K. Larger = finer granularity, more LLM tokens downstream. |
| `--vit_name` | `vit-large` | Trajectory transformer size (used in `simpletrajvitv2` mode, see `../trajvitv2/`). Ignored for `simplesegmenter`. |
| `--ckpt` | `None` | Resume from a checkpoint. Auto-sets `--resume`. |
| `--lr` | `1e-4` | Peak LR (cosine decay, 1-epoch warmup). |

## Evaluation

Three clean drivers in `scripts/`:

```bash
python scripts/eval_davis.py  --ckpt CKPT --davis_root /path/to/DAVIS
python scripts/eval_mose.py   --ckpt CKPT --mose_root  /path/to/MOSE
python scripts/eval_ytvis.py  --ckpt CKPT --ytvis_root /path/to/ytvis2019 \
                              --ann_file /path/to/ytvis2019/instances_valid.json
```

Each driver:
- Loads the released checkpoint via `--ckpt`
- Samples `--num_frames` frames per video (default 8)
- Runs the segmenter at `--image_res` (default 224)
- Hungarian-matches predicted trajectory IDs against GT instance IDs at IoU ≥ `--iou_thr` (default 0.5)
- Writes aggregate metrics to `<output_dir>/metrics.json` + per-video breakdown to `per_video.csv`
- Optionally saves overlay visualisations for the first `--save_viz N` videos

Reusable helpers (`save_pca_feature_maps`, `merge_tracklets`,
`downsample_segmentation_probs`) live in
`trajtok_segmenter/eval/eval_segmenter.py` for custom pipelines.

## Repository layout

```
segmenter/
├── README.md                       (this file)
├── pyproject.toml
├── configs/
│   ├── pretrain.yaml              (training config; reads TRAJTOK_* env vars)
│   ├── config_bert.json           (BERT text-encoder config, used by simpletrajvitv2)
│   └── beit-base-patch16-...json  (kept for backward compat)
├── scripts/
│   ├── train.sh                   (single-node launch)
│   ├── download_ckpt.py           (HF Hub fetch)
│   ├── demo_image.py              (run on one image, save mask + overlay)
│   ├── eval_davis.py              (DAVIS-2017 val: Hungarian-matched VEQ + STQ_EN)
│   ├── eval_mose.py               (MOSE val: same metrics, MOSE layout)
│   └── eval_ytvis.py              (YT-VIS 2019/2021: COCO-RLE annotations)
└── trajtok_segmenter/
    ├── model/                     (SimpleSegmenter + PerceiverResampler + DINOv3 + ...)
    ├── data/                      (datasets + loaders for the corpora above)
    ├── train/                     (training loop, optimizer, scheduler, distributed helpers)
    ├── eval/                      (eval_segmenter.py + seg_metric.py)
    └── text/                      (BERT tokenizer + xbert; used by simpletrajvitv2 contrastive mode)
```

## Known limitations of this release

- **Frame count cap**: the released checkpoint was trained at T ≤ 8 frames per
  clip. Longer-clip behaviour at inference is untested.
- **Resolution**: trained at 224×224 inputs producing a 56×56 trajectory grid.
  Other resolutions work but lose quality away from this point.
- **Caption / contrastive head** (`simpletrajvitv2` mode): the code is bundled
  for reproducibility but **uses a BERT text encoder and ImageNet-style
  contrastive loss** that you may want to swap for a more modern alternative.
- **`eval/eval_segmenter.py`** is a research artifact with weka-specific
  paths; use it as a reference, not a turnkey tool. The clean
  `scripts/eval_davis.py` is the supported entry point (coming in a follow-up).
- **DINOv3 weights are not bundled** — download separately (Apache-2.0).

## Citation

If you use this code or checkpoint, please cite the paper:

```bibtex
@article{zheng2026trajtokv2,
  title   = {TrajTok-v2: Trajectory-aware visual tokenization for vision-language models},
  author  = {Zheng, Chenhao and others},
  journal = {arXiv preprint arXiv:2602.22779},
  year    = {2026},
}
```

Apache-2.0 — see the [LICENSE](../LICENSE) file at the repo root.
