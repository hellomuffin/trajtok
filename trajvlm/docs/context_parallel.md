# Context Parallelism (CP)

Context Parallelism shards long sequences across multiple GPUs so that each GPU only processes
a fraction of the tokens and images. This enables training on sequences that exceed a single GPU's
memory (e.g., 384-frame videos at 36K+ token sequence lengths).

## Overview

With CP degree N, the N GPUs in a CP group each process ~1/N of the sequence. During attention,
GPUs communicate keys and values so that each rank can compute attention over the full sequence
using only its local queries. The result is mathematically equivalent to single-GPU training.

```
GPU 0: tokens [0, S/2)    ──┐
                             ├── communicate K,V during attention ──> same result as full sequence
GPU 1: tokens [S/2, S)    ──┘
```

## How It Fits Into the Training Pipeline

CP touches four stages of the pipeline:

```
Packing ──> Collation ──> Vision Backbone ──> LLM Forward
  │              │              │                  │
  │              │              │                  └─ _prepare_cp_inputs() shards text
  │              │              └─ Each rank encodes a subset of images
  │              └─ Passes image_shard_boundaries through
  └─ Pre-computes how to split images across ranks
```

### 1. Packing (`olmo/data/dynamic_packer.py`)

When CP is enabled (`PackingConfig.cp_world_size > 1`), the packer calls
`compute_image_shard_boundaries()` after packing examples together. This function:

- Takes the cumulative image bounds and token pooling bounds from each packed example
- Divides images across CP ranks to balance the load
- Returns per-rank boundaries: `{rank_id: {start_image_idx, end_image_idx, start_pool_idx, end_pool_idx, max_num_tokens}}`
- Handles edge cases like GPU starvation (a rank getting zero images) by stealing from neighbors

These boundaries are stored in the packed example as `image_shard_boundaries`.

### 2. Collation (`olmo/preprocessing/multimodal_collator.py`)

The collator passes three CP-related fields through without modification:

- `cum_image_bounds`: Cumulative count of images per frame/example
- `cum_token_pooling_bounds`: Cumulative count of pooling rows per frame/example
- `image_shard_boundaries`: Pre-computed per-rank boundaries from the packer

These are stored as lists (one entry per batch element) rather than collated into padded tensors.

### 3. Vision Backbone (`olmo/nn/vision_backbone.py`)

When `enable_cp=True`, the vision backbone's `forward()` shards images across CP ranks
before encoding them through the ViT + connector. It uses three fallback strategies:

1. **Pre-computed boundaries** (preferred): Uses `image_shard_boundaries` from the packer.
   Each rank slices its assigned images and pooling indices directly. No communication needed.

2. **On-the-fly sharding**: If no pre-computed boundaries are available but `cum_image_bounds`
   and `cum_token_pooling_bounds` are present, `_shard_by_image_bounds()` computes the split
   at runtime.

3. **Uniform sharding**: Last resort. Uses the CP load balancer's `batch_shard()` to split
   images evenly on the image dimension.

After each rank encodes its subset of images through the ViT and connector, the results are
**all-gathered** back so every rank has the full set of image features for embedding into the
token sequence.

### 4. LLM Forward (`olmo/models/molmo2/molmo2.py`)

After image features are embedded into the token sequence, `_prepare_cp_inputs()` shards
all text-related tensors on the sequence dimension:

- `input_ids`, `labels`, `loss_masks`, `response_mask`, `position_ids`, `subsegment_ids`, `input_embeds`

It also collects per-block RoPE buffers (sin/cos positional embeddings) so each rank
uses position-correct embeddings for its shard.

The sharded tensors are then passed through the transformer blocks. Each block's attention
layer uses **Ulysses** attention to communicate K,V across ranks.

## CP vs Non-CP Comparison

| Stage | Without CP | With CP |
|---|---|---|
| Packing | `pack()` | `pack_with_cp()` + `compute_image_shard_boundaries()` |
| Collation | Standard | Same, plus passes `image_shard_boundaries` |
| Vision Backbone | All images on all ranks | Each rank encodes a subset, then all-gather |
| Text Inputs | Full sequence per rank | `_prepare_cp_inputs()` shards on seq dimension |
| Attention | Standard SDPA | Ulysses attention with cross-rank communication |
| Loss | Full sequence | Each rank computes loss on its portion only |

When CP is disabled, `cum_image_bounds`, `cum_token_pooling_bounds`, and `image_shard_boundaries`
still flow through the data pipeline but are never consumed by the model (gated by
`enable_cp` in the vision backbone and `_cp_load_balancer is not None` in the model forward).

## Ulysses Attention

We currently only support Ulysses attention for CP. Our attention masks are complex
(bidirectional attention for image tokens, subsegment masks for packing, causal masks
for text) which makes Ring attention difficult to support correctly. Ulysses is more
flexible with arbitrary attention masks since each rank computes full-sequence attention
on its assigned heads, so any mask shape works without special handling. The codebase
contains load balancer implementations for Ring attention, but these are not actively
used or tested.

Ulysses attention ([DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509)) works by
partitioning the sequence across ranks and using all-to-all communication to redistribute
along the head dimension before and after attention:

1. **Input**: Each rank holds a contiguous chunk of the sequence for all attention heads.
   With CP degree N and sequence length S, rank i holds tokens `[i*S/N, (i+1)*S/N)`.

2. **All-to-all (scatter by head, gather by sequence)**: Before attention, ranks exchange
   data so that each rank holds the *full sequence* but only for a *subset of attention heads*.
   For example, with 32 heads and CP degree 2: rank 0 gets heads 0-15 for the full sequence,
   rank 1 gets heads 16-31 for the full sequence.

3. **Local attention**: Each rank computes standard attention (Q @ K^T / sqrt(d) -> softmax -> @ V)
   on its assigned heads over the full sequence. This is mathematically identical to
   non-CP attention since each head sees all tokens.

4. **All-to-all (scatter by sequence, gather by head)**: After attention, ranks exchange data
   back so that each rank holds all heads but only for its sequence chunk again.

This approach has two key advantages:
- Each rank computes exact (not approximate) attention
- Only two all-to-all collectives per layer, which scale well on high-bandwidth interconnects like NVLink

## Load Balancer (`olmo/nn/cp_load_balancer.py`)

The `UlyssesLoadBalancer` handles sequence sharding for Ulysses attention. It implements
`batch_shard()` which takes a list of tensors and their sequence dimensions, pads them to
be divisible by the CP world size, and returns the contiguous shard for the current rank.

For example, with CP degree 2 and a sequence of 2048 tokens:
- Rank 0 gets tokens `[0, 1024)`
- Rank 1 gets tokens `[1024, 2048)`

## Configuration

### In `launch_scripts/sft.py`

CP is configured through two mechanisms:

1. **`--cp_degree` argument** (line 273): Sets `PackingConfig.cp_world_size` for the data pipeline
2. **`cfg.parallelism.context_parallel_config`** (line 439): Sets the model-level CP config

Both must agree on the degree. The launch script also sets `model.cp_enabled = True` when
degree > 1 (line 445-446), which affects the collator's validation logic.

### Key Config Fields

```python
# In TrainConfig.parallelism.context_parallel_config:
degree: int = 1              # Number of GPUs per CP group
attention_type: str = "ulysses"  # Currently only "ulysses" is supported
load_balancer: str = "ulysses"   # Load balancer type
head_stride: int = 1

# In PackingConfig:
cp_world_size: int = 1       # Must match CP degree

# In Molmo2Config:
cp_enabled: bool = False          # Set automatically when degree > 1
apply_cp_to_vision_backbone: bool = False  # Shard ViT encoding too
```

### Example: 8 GPUs with CP degree 2

```bash
torchrun --nproc-per-node=8 launch_scripts/sft.py /path/to/checkpoint molmo2 \
    --cp_degree=2 \
    --parallelism.context_parallel_config.degree=2 \
    --parallelism.context_parallel_config.attention_type=ulysses \
    --seq_len=36864 \
    --model.mm_preprocessor.video.max_frames=384 \
    --model.llm.max_sequence_length=36864 \
    --save_folder=/path/to/save
```

This creates 4 data-parallel groups, each containing 2 CP ranks. Each CP pair jointly
processes sequences of up to 36864 tokens, with each rank handling ~18432 tokens.

## Key Files Reference

| File | CP-Related Code |
|---|---|
| `olmo/nn/cp_load_balancer.py` | Load balancer implementations |
| `olmo/data/dynamic_packer.py` | `compute_image_shard_boundaries()`, `pack_with_cp()` |
| `olmo/preprocessing/multimodal_collator.py` | Passes CP fields through collation |
| `olmo/models/molmo2/molmo2.py` | `apply_cp()`, `_prepare_cp_inputs()`, forward gating |
| `olmo/nn/vision_backbone.py` | Image sharding and all-gather in `forward()` |
| `olmo/nn/llm.py` | `OLMoBlock.apply_cp()`, attention layer CP setup |
| `olmo/train/run_trainer.py` | `build_world_mesh()`, `parallelize_model()` |
| `launch_scripts/sft.py` | CLI args, `cp_degree`, config wiring |
