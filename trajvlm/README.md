# TrajVLM

**TrajVLM** is a vision-language model that uses our trajectory segmenter as a
trajectory-aware connector over SigLIP2 visual features, feeding the pooled
trajectory tokens into a Qwen3-4B-Instruct LLM.

```
input frames @378×378
    │
    ├──→ SigLIP2-so400m-14-378 ViT                  →  F   (T·729, 1152)   [trainable]
    │
    └──→ resize → 224×224 → SimpleSegmenter          →  assignment_logits (T·56·56, 128)
                                                        + initial_queries  (128, 512)
                                                        [trainable; init from ../segmenter ckpt]
                                                          │
                                          interp 56→27   │
                                                          ↓
                              assignment_logits (T·729, 128)
                                                          │
                          argmax → predicted_label → query_mask
                                                          │
       initial_queries ── Linear(512→1152) ──→ Q (128, 1152)
                                                          │
                                                          ↓
                    TrajPerceiver cross-attn @ 1152-dim (depth=2)
                    Q attends to F with query_mask
                                                          │
                                                          ↓
                              trajectory tokens (128, 1152)
                                                          │
                                projector MLP 1152 → d_llm
                                                          ↓
                                                      Qwen3-4B-Instruct LLM
```

For T frames, this produces `ceil(T / 8) × 128` LLM tokens
(default `frames_per_pool=8`, so a 128-frame video → 2,048 tokens — **5×
fewer than SigLIP2-style spatial pooling at the same resolution**).

## What's in this package vs what you need to install

This package vendors the **TrajVLM-specific files**:

| File | Purpose |
|---|---|
| `trajtok_trajvlm/nn/siglip2_trajgroup_vision_backbone.py` | The new vision backbone (SigLIP2 + segmenter + cross-attn pool) |
| `trajtok_trajvlm/nn/trajvit_perceiver.py` | RoPE-aware perceiver used by the cross-attn pool |
| `trajtok_trajvlm/nn/trajvit_dinov3.py` | Meta-device-safe DINOv3 wrapper for the segmenter |
| `trajtok_trajvlm/checkpoints/load_trajvit_segmenter_full.py` | DTensor-aware loader for the released segmenter ckpt |
| `trajtok_trajvlm/launch_scripts/trajvlm_{pretrain,sft}.py` | Pretrain + SFT entry points (with TrajVlmConfig shim) |
| `trajtok_trajvlm/preprocessing/trajvit_preprocessor.py` | Image/video preprocessor matching our token count contract |
| `scripts/{pretrain,sft}.sh` | torchrun launchers (single-node example; adapt for multi-node) |
| `scripts/apply_molmo2_patches.py` | Copies the above into a molmo2 source tree (recommended install path) |

**You also need** (these are NOT bundled):

| Dependency | License | Where |
|---|---|---|
| **Molmo2 source** | Apache-2.0 | https://github.com/allenai/molmo2 (or your fork) |
| **SigLIP2-so400m-14-378 weights** | Apache-2.0, Google | https://github.com/google-research/big_vision |
| **Qwen3-4B-Instruct weights** | Apache-2.0, Alibaba | https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 |
| **Released TrajTok segmenter ckpt** | Apache-2.0, this repo | Download with `../segmenter/scripts/download_ckpt.py` |
| **PixMoCap data** (for pretrain) | Apache-2.0, Allen AI | Follow the Molmo2 README |
| **Molmo2 SFT mixture data** (for SFT) | mixed; see Molmo2 README | Follow the Molmo2 README |

## Install

### Recommended path (vendor-into-molmo2)

```bash
# 1. Clone molmo2 + install
git clone https://github.com/allenai/molmo2.git
pip install -e ./molmo2

# 2. Apply our TrajVLM patches into the molmo2 tree
cd trajvlm/
python scripts/apply_molmo2_patches.py --molmo2_root ../molmo2 --rewrite_imports
```

After this, `from olmo.nn.siglip2_trajgroup_vision_backbone import ...` works
inside the molmo2 namespace, and the launch scripts at
`molmo2/launch_scripts/trajvlm_pretrain.py` are ready to go.

### Alternative path (sibling-package)

If you'd rather keep our code separate from molmo2's source tree:

```bash
git clone https://github.com/allenai/molmo2.git
pip install -e ./molmo2
pip install -e ./trajvlm
```

Then run via the trajvlm-package entry point:

```bash
PYTHONPATH=./molmo2:./trajvlm \
torchrun --nproc-per-node=8 \
  -m trajtok_trajvlm.launch_scripts.trajvlm_pretrain ...
```

## Pretrain (image-only PixMoCap, ~4 epochs)

```bash
export MOLMO_DATA_DIR=/path/to/molmo-data   # has pretrained_image_encoders/, pretrained_llms/, PixMoCap data
export TRAJTOK_SEG_CKPT=/path/to/segmenter_filteredmixdata_all.pth

bash scripts/pretrain.sh \
  --save_folder /path/to/save/trajvlm_pretrain_v1 \
  --seg_ckpt $TRAJTOK_SEG_CKPT
```

At 2 nodes × 8 H100s this takes ~24 hours for 22k steps (4 epochs of PixMoCap).

## SFT (Molmo2 default SFT mixture)

```bash
bash scripts/sft.sh \
  --pretrain_ckpt /path/to/save/trajvlm_pretrain_v1/step22347 \
  --save_folder /path/to/save/trajvlm_sft_v1
```

Default mixture follows Molmo2's `sft.py:get_training_mixture("molmo2")`
(image_academic + video_academic + pointing + nlp + hardcodes + tracking).
LRs match the TrajVLM paper: connector 5e-6, vit 5e-6, llm 1e-5.

## Evaluation

The trajvlm forward integrates cleanly with **Molmo2's existing eval harness**.
Once you've patched molmo2 and run training, evaluate any saved checkpoint with:

```bash
python molmo2/launch_scripts/eval_molmo2.py \
  --checkpoint /path/to/save/trajvlm_sft_v1/stepNNNN \
  --evaluations point_bench mvbench chart_qa
```

(See molmo2 README for the full list of supported evals.)

## Implementation gotchas

A few non-obvious things this code handles for you — useful if you fork:

1. **Meta-device safety**: SimpleSegmenter's DINOv3 wrapper (`trajvit_dinov3.py`)
   defers eager weight loading so `Molmo2.__init__` on meta device works.
2. **Frozen segmenter sub-modules**: `traj_seg_head_low_res`, `patch_decoder_low_res`,
   and `log_scale` are set to `requires_grad=False` because they only feed
   into the `argmax → query_mask` path, which is non-differentiable. Without
   freezing, AdamW would track them with zero gradient → their `step` counter
   stays uninitialised → checkpoint resume breaks with a "Missing key" error.
3. **DTensor-aware ckpt load**: `load_trajvit_segmenter_full.py` handles both
   pre-FSDP (plain Tensor) and post-FSDP (DTensor) loading via per-key
   broadcast from rank 0.
4. **Bilinear interp 56→27**: the segmenter operates at 56×56 spatial grid
   (224 input × 1/4 ConvNeXt stride), SigLIP2 at 27×27 (378 input / 14 patch).
   Masks are bilinearly interpolated down to match the SigLIP2 grid.

## Released checkpoint

**Not released in this version.** The pretrain (~125 GPU-hr) + SFT
(~300k steps) is multi-day compute; we release the **training + evaluation
code** so you can train your own variant. Check back in a future update.

## Repository layout

```
trajvlm/
├── README.md                       (this file)
├── pyproject.toml
├── configs/                        (placeholder — configs live inside the vendored launch scripts)
├── scripts/
│   ├── pretrain.sh                 (single-node torchrun launcher)
│   ├── sft.sh                      (single-node SFT launcher)
│   └── apply_molmo2_patches.py     (vendor → molmo2 source tree)
└── trajtok_trajvlm/
    ├── nn/                         (our vision backbone + perceiver + DINOv3 wrapper)
    ├── checkpoints/                (segmenter-ckpt loaders, DTensor-aware)
    ├── preprocessing/              (trajvit_preprocessor: image + video token contract)
    └── launch_scripts/             (trajvlm_pretrain.py + trajvlm_sft.py)
```

## Citation

See [`../README.md`](../README.md).
