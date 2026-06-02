# TrajVLM

📄 **Paper:** [arXiv:2602.22779](https://arxiv.org/abs/2602.22779)

**TrajVLM** is a trajectory-token vision-language model: SigLIP2 visual features
are grouped into 128 trajectory tokens per pool-chunk by our [released
segmenter](../segmenter), then fed into a Qwen3-4B-Instruct LLM. For a
128-frame video at `frames_per_pool=8` this is **2,048 LLM tokens — 5× fewer
than SigLIP2 spatial pooling at the same resolution**.

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

This codebase is a **self-contained Molmo2 fork** — the `olmo/` and
`launch_scripts/` directories are Molmo2's, with our trajvlm-specific
additions merged in. Clone this repo and you have everything you need to
pretrain + SFT trajvlm.

## Install

```bash
git clone https://github.com/hellomuffin/trajtok.git
cd trajtok/trajvlm
pip install -e .
```

Python ≥ 3.10. See `Dockerfile` for a reproducible CUDA env, or follow
Molmo2's install notes (below) for additional video deps.

## Download external assets

| Asset | License | How |
|---|---|---|
| **TrajTok segmenter ckpt** | Apache-2.0 (this repo) | `python ../segmenter/scripts/download_ckpt.py` |
| **SigLIP2-so400m-14-378 weights** | Apache-2.0 (Google) | Follow Molmo2's "Downloading Pretrained Models" — `python scripts/prepare_pretrained_model.py siglip2` |
| **Qwen3-4B-Instruct weights** | Apache-2.0 (Alibaba) | `python scripts/prepare_pretrained_model.py qwen3_4b_instruct` |
| **PixMoCap data** (for pretrain) | Apache-2.0 (Ai2) | `python scripts/download_datasets.py pixmo_cap` |
| **Molmo2 SFT mixture data** | mixed; per-dataset | `python scripts/download_datasets.py all --n_proc 8` (long-running) |

Set `MOLMO_DATA_DIR` to where these all live:

```bash
export MOLMO_DATA_DIR=/path/to/molmo-data
export HF_HOME=$MOLMO_DATA_DIR/huggingface
export HF_DATASETS_CACHE=$MOLMO_DATA_DIR/hf_datasets
```

## Pretrain (image-only PixMoCap, ~4 epochs ≈ 22k steps)

```bash
# Get the segmenter ckpt first
python ../segmenter/scripts/download_ckpt.py
SEG_CKPT=$(pwd)/../segmenter/checkpoints/segmenter_filteredmixdata_all.pth

bash scripts/pretrain.sh \
  --save_folder ./results/trajvlm_pretrain \
  --seg_ckpt $SEG_CKPT
```

At 2 nodes × 8 H100s this takes ~24 hours for 22k steps. Adjust `--nnodes`
for multi-node (also set `MASTER_ADDR` / `MASTER_PORT` / `NODE_RANK` for
torchrun).

## SFT (Molmo2 default SFT mixture)

```bash
bash scripts/sft.sh \
  --pretrain_ckpt ./results/trajvlm_pretrain/step22347 \
  --save_folder ./results/trajvlm_sft
```

Default mixture follows Molmo2's `sft.py:get_training_mixture("molmo2")`
(image_academic + video_academic + pointing + nlp + tracking, with multi-image
datasets removed since TrajVit is single-image / single-video per sample).
LRs match the paper: connector 5e-6, vit 5e-6, llm 1e-5.

## Evaluation

The trajvlm forward integrates cleanly with **Molmo2's eval harness**.
Evaluate any saved checkpoint with:

```bash
python launch_scripts/eval_molmo2.py \
  --checkpoint ./results/trajvlm_sft/stepNNNN \
  --evaluations point_bench mvbench chart_qa
```

See `MOLMO_POINT_README.md` and the original Molmo2 docs for the full list of
supported evals.

## Where the trajvlm-specific code lives

The vast majority of files here are unmodified Molmo2 source. Our additions:

| File | Purpose |
|---|---|
| [`olmo/nn/siglip2_trajgroup_vision_backbone.py`](olmo/nn/siglip2_trajgroup_vision_backbone.py) | The new vision backbone (SigLIP2 + segmenter + cross-attn pool) |
| [`olmo/nn/trajvit_perceiver.py`](olmo/nn/trajvit_perceiver.py) | RoPE-aware perceiver used by the cross-attn pool |
| [`olmo/nn/trajvit_dinov3.py`](olmo/nn/trajvit_dinov3.py) | Meta-device-safe DINOv3 wrapper for the segmenter |
| [`olmo/checkpoints/load_trajvit_segmenter_full.py`](olmo/checkpoints/load_trajvit_segmenter_full.py) | DTensor-aware loader for the released segmenter ckpt |
| [`olmo/preprocessing/trajvit_preprocessor.py`](olmo/preprocessing/trajvit_preprocessor.py) | Image / video preprocessor matching our token-count contract |
| [`launch_scripts/trajvlm_pretrain.py`](launch_scripts/trajvlm_pretrain.py) | Pretrain entry point (TrajVlmConfig shim) |
| [`launch_scripts/trajvlm_sft.py`](launch_scripts/trajvlm_sft.py) | SFT entry point (TrajVlmConfig shim) |
| [`scripts/pretrain.sh`](scripts/pretrain.sh), [`scripts/sft.sh`](scripts/sft.sh) | torchrun launchers |

A handful of in-place edits to Molmo2's data + preprocessor paths fix
edge cases that trajvit-style training hits (multi-image gates, video-loader
bad-sample handling, etc.). Everything else is unchanged upstream Molmo2.

## Implementation gotchas

A few non-obvious things this code handles — useful if you fork:

1. **Meta-device safety**: SimpleSegmenter's DINOv3 wrapper (`trajvit_dinov3.py`)
   defers eager weight loading so `Molmo2.__init__` on meta device works.
2. **Frozen segmenter sub-modules**: `traj_seg_head_low_res`, `patch_decoder_low_res`,
   and `log_scale` are set to `requires_grad=False` because they only feed
   into the `argmax → query_mask` path, which is non-differentiable. Without
   freezing, AdamW would track them with zero gradient → their `step` counter
   stays uninitialised → checkpoint resume breaks.
3. **DTensor-aware ckpt load**: `load_trajvit_segmenter_full.py` handles both
   pre-FSDP (plain Tensor) and post-FSDP (DTensor) loading via per-key
   broadcast from rank 0.
4. **Bilinear interp 56→27**: the segmenter operates at 56×56 spatial grid
   (224 input × 1/4 ConvNeXt stride), SigLIP2 at 27×27 (378 input / 14 patch).
   Masks are bilinearly interpolated down to match the SigLIP2 grid.
5. **Per-entry backbone forward**: the SigLip2TrajGroup backbone slices the
   padded `images` tensor down to the real-clip count using `cum_image_bounds`
   before SigLip2 / segmenter / cross-attn — otherwise the Molmo2 collator's
   global-max-shape padding produces more visual features than the text has
   slots for, which crashes at `molmo2.py:713` with a tensor-size mismatch.

## Released checkpoint

**Not released in this version.** Pretrain (~125 GPU-hr) + SFT (~300 GPU-day)
is multi-day compute; we release the training + eval code so you can train
your own. Check back in a future update.

## Acknowledgements

This repository is a fork of [**Molmo2**](https://github.com/allenai/molmo2)
(© 2025–2026 Allen Institute for AI, Apache-2.0). All training infrastructure,
the LLM stack, the data pipeline, and the eval harness are theirs; we only
add the trajectory-grouping vision backbone and its supporting pieces. If you
use this code, please cite **both** TrajTok and Molmo2:

```bibtex
@article{zheng2026trajtok,
  title   = {TrajTok: Trajectory-aware visual tokenization for vision-language models},
  author  = {Zheng, Chenhao and others},
  journal = {arXiv preprint arXiv:2602.22779},
  year    = {2026},
}

@article{molmo2,
  title   = {Molmo 2: State-of-the-art video understanding, pointing, and tracking},
  author  = {{Allen Institute for AI}},
  journal = {arXiv preprint arXiv:2601.10611},
  year    = {2026},
}
```

Apache-2.0 — see the [LICENSE](LICENSE) file (Molmo2's, retained as-is).
