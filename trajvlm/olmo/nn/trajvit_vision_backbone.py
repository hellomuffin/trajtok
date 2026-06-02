"""TrajViT vision backbone for Molmo2.

Replaces Molmo2's SigLIP ViT + multi-crop + ViTMultiHeadDotProductAttention
connector with our pretrained DINOv3-small + PerceiverResampler trajectory
encoder ("TrajTok" in Zheng et al. 2026, arXiv 2602.22779).

Per-image / per-clip output is exactly `num_traj` trajectory tokens
(default 128) projected to the LLM's `d_model`. The Molmo2 forward expects
a flat `(total_image_tokens_across_batch, d_model)` tensor where image
features are added at positions where `input_ids == _image_high_res_id`.

Paper Eq. 1: M_k,t,i,j = softmax_k(q_k · F_t,i,j)
Paper Eq. 2: z_k       = Σ_{t,i,j} M_k · F_t,i,j

(We use the *soft* mask formulation here; the segmenter pretraining used
argmax + Hungarian, but per the paper the soft pooling carries richer
information for the LLM.)
"""
from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import field, replace
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributed.fsdp import fully_shard

from olmo.config import BaseConfig
from olmo.nn.trajvit_dinov3 import DINOResNetHierFeat
from olmo.nn.trajvit_perceiver import PerceiverResampler
from olmo.nn.vision_backbone import MolmoVisionBackboneConfig


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrajVitVisionBackboneConfig(MolmoVisionBackboneConfig):
    """Config for the TrajViT trajectory-token connector.

    Inherits from MolmoVisionBackboneConfig so that OmegaConf's strict type check
    on Molmo2Config.vision_backbone passes. The inherited fields (`vit`,
    `image_pooling_2d`, `vit_layers`, etc.) are unused by our backbone — we only
    rely on `build_preprocessor` + `build` being overridden below.
    """

    # --- DINOv3 backbone ---
    backbone_model_size: str = "small"          # one of {tiny, small, base, large}
    backbone_pool: str = "sum"                  # {sum, concat, lastlayer}
    backbone_upsample_first: bool = True

    # --- Perceiver / trajectory tokens ---
    segment_embed_dim: int = 512
    num_traj: int = 128
    perceiver_depth: int = 2
    perceiver_heads: int = 8
    perceiver_dim_head: int = 64
    use_rotary: bool = True

    # --- Spatial resolution at which we compute soft masks / pool ---
    image_res: int = 224
    latent_res: int = 56                        # feature-map resolution
    image_patch_size: int = 4                   # image_res / latent_res — used only for token-count bookkeeping

    # --- Soft mask + pooling ---
    mask_softmax_temperature: float = 1.0       # base temperature; effective = T / scale
    detach_patch_features_for_perceiver: bool = True
    init_log_scale: float = math.log(16.0)      # mirror seg pretraining default
    """Learnable scale applied to assignment logits before softmax (paper Eq. 1).
    With unit-normalised q,f the raw dot product is in [-1, 1]; without scaling
    the softmax over K=128 latents collapses to ~uniform and all trajectory
    tokens become identical (the segmenter trained `log_scale ≈ log(16)`)."""
    """Mirror our segmenter training: detach patch features before the
    perceiver so backbone grads come only from the soft-mask pool path."""

    # --- Projector ---
    image_projector: str = "mlp"                # {mlp, linear}

    # --- Misc ---
    image_feature_dropout: float = 0.0
    connector_activation_checkpointing: bool = True
    compile_vit: Optional[str] = None
    compile_connector: Optional[str] = None
    normalize_on_gpu: bool = False
    """Match Molmo2's flag; we keep CPU normalisation by default."""

    # --- Pretrained-segmenter checkpoint to load at init (optional) ---
    pretrained_segmenter_path: Optional[str] = None
    """Path to our seg pretraining `latest.pth`. If set, backbone +
    perceiver weights are loaded from it."""

    def build_preprocessor(self):
        # `vision_backbone.image_preprocessor` is the *low-level* numpy-side
        # ImagePreprocessor — only used downstream by tokenizer-aware
        # preprocessors that want patch-size / base-input-size. The actual
        # token-emitting preprocessor for our backbone is built inside
        # `TrajVitPreprocessorConfig.build_image_preprocessor` from
        # olmo/preprocessing/trajvit_preprocessor.py.
        from olmo.preprocessing.image_preprocessor import ImagePreprocessor
        return ImagePreprocessor(
            normalize="dino",
            resize="dino",
            pad_value=0.0,
            # We treat the whole 224×224 image as one "patch" so the data
            # pipeline's batch_pixels_to_patches reshape works trivially.
            image_patch_size=self.image_res,
            base_image_input_size=(self.image_res, self.image_res),
            normalize_on_gpu=self.normalize_on_gpu,
        )

    def build(self, llm_config, device=None):
        return TrajVitVisionBackbone(self, llm_config, device)

    @classmethod
    def update_legacy_settings(cls, config):
        return config


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class _ProjectorMLP(nn.Module):
    """2-layer MLP projector: D_traj -> 4*d_model -> d_model.

    Mirrors Molmo2's `ImageProjectorMLP` shape — gated SwiGLU-ish.
    """

    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, 4 * d_model, bias=False)
        self.w3 = nn.Linear(in_dim, 4 * d_model, bias=False)
        self.w2 = nn.Linear(4 * d_model, d_model, bias=False)
        self.act = nn.SiLU()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02, a=-2.0, b=2.0)

    def forward(self, x):
        return self.w2(self.act(self.w1(x)) * self.w3(x))


class TrajVitVisionBackbone(nn.Module):
    """DINOv3 ConvNeXt-small + PerceiverResampler + soft-mask pool +
    projector → LLM d_model.
    """

    def __init__(self, config: TrajVitVisionBackboneConfig, llm_config, device=None):
        super().__init__()
        self.config = config
        self.image_preprocessor = config.build_preprocessor()
        self.image_feature_dropout = nn.Dropout(config.image_feature_dropout)

        D = config.segment_embed_dim

        # Backbone — construct on the passed device (may be `meta`); weights deferred
        self.patch_encoder = DINOResNetHierFeat(
            model_size=config.backbone_model_size,
            out_channel=D,
            upsample_first=config.backbone_upsample_first,
            load_pretrained=False,        # see reset_with_pretrained_weights
        )

        # Perceiver — same hyperparams as our seg pretraining
        self.trajectory_perceiver = PerceiverResampler(
            dim=D,
            depth=config.perceiver_depth,
            dim_head=config.perceiver_dim_head,
            heads=config.perceiver_heads,
            num_latents=config.num_traj,
            use_rotary=config.use_rotary,
            use_latent_transformer=False,
            update_x=False,
        )

        # LayerNorm before projector: the soft-mask weighted sum (paper Eq. 2,
        # z_k = Σ_{N pixels} M·F) produces O(N)-scale features (N≈T·56·56 ≈
        # 3 000 per clip), which would otherwise be ~500× larger than LLM
        # wte embeddings (~0.02 std) and swamp the residual stream.
        self.pre_projector_ln = nn.LayerNorm(D)
        # Learnable scale on assignment logits (paper Eq. 1, before softmax).
        # See config docstring for why this is necessary.
        self.log_scale = nn.Parameter(torch.tensor([config.init_log_scale]))

        # Projector to LLM d_model
        d_model = llm_config.d_model
        if config.image_projector == "mlp":
            self.image_projector = _ProjectorMLP(D, d_model)
        elif config.image_projector == "linear":
            self.image_projector = nn.Linear(D, d_model, bias=False)
        else:
            raise NotImplementedError(config.image_projector)

        # Honour the device argument when it's a real device (not meta);
        # under Molmo2's trainer this is `meta` at construction, then to_empty()+pretrained-load later.
        if device is not None and getattr(device, "type", str(device)) != "meta":
            self.to(device)

    # ---- training-infra hooks ----

    def reset_parameters(self):
        """Called after meta->materialise to fill in random init for everything that
        won't be pretrained-loaded. Pretrained loads happen in
        reset_with_pretrained_weights() instead."""
        # Re-init projector (random); perceiver self-inits via its own ctor (already random).
        if hasattr(self.image_projector, "reset_parameters"):
            self.image_projector.reset_parameters()

    def reset_with_pretrained_weights(self):
        """Load:
          * If `pretrained_segmenter_path` is set: load our seg pretraining ckpt
            (contains patch_encoder + trajectory_perceiver weights — DINOv3 weights
            are the fine-tuned version from segmenter stage, so we skip the DINOv3
            base load to avoid double work).
          * Otherwise: load DINOv3 base weights into self.patch_encoder.backbone.
          * Always: re-init image_projector (random, no pretrained source).
        """
        if self.config.pretrained_segmenter_path is not None:
            from olmo.checkpoints.load_trajvit_segmenter import load_segmenter_into_backbone
            load_segmenter_into_backbone(self, self.config.pretrained_segmenter_path)
        else:
            # No seg ckpt → fall back to DINOv3 base weights only (perceiver stays random).
            # NOTE: this path is not FSDP-aware; safe only when called pre-FSDP.
            try:
                msg = self.patch_encoder.load_dinov3_pretrained()
                log.info(f"[trajvit] DINOv3 base weights loaded: missing={len(getattr(msg,'missing_keys',[]))}, "
                         f"unexpected={len(getattr(msg,'unexpected_keys',[]))}")
            except Exception as e:
                log.warning(f"[trajvit] DINOv3 base-weight load failed (continuing): {e}")

        # Re-init projector after any state_dict load (defensively; load shouldn't touch it).
        if hasattr(self.image_projector, "reset_parameters"):
            self.image_projector.reset_parameters()

    def apply_fsdp2(self, **kwargs):
        # Only shard the perceiver + projector (DTensor). The DINOv3 ConvNeXt
        # stays as plain Tensors — its internal LayerNorms are not auto-recursed
        # by fully_shard, and shard-ing the parent (`self`) mixes plain and
        # DTensor params, which breaks `dist_cp_sd.set_model_state_dict`.
        # The backbone is small (~50M), so memory-wise replication is fine.
        fully_shard(self.trajectory_perceiver, **kwargs)
        fully_shard(self.image_projector, **kwargs)

    def apply_activation_checkpointing(self):
        if self.config.connector_activation_checkpointing:
            try:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
                self.trajectory_perceiver = checkpoint_wrapper(self.trajectory_perceiver)
                self.image_projector = checkpoint_wrapper(self.image_projector)
            except Exception as e:
                log.warning(f"checkpoint_wrapper unavailable: {e}")

    def apply_compile(self, **kwargs):
        # Optional torch.compile hooks. Skipped by default — DINOv3 ConvNeXt
        # tends to be incompatible with dynamic shapes.
        pass

    # Matches Molmo2.get_connector_parameters() contract: returns *non-ViT* params
    def get_connector_parameters(self):
        vit_params = set(self.patch_encoder.parameters())
        return (p for p in self.parameters() if p not in vit_params)

    @property
    def image_vit(self):
        """Alias so `Molmo2.get_vit_parameters` (which assumes
        `vision_backbone.image_vit`) returns our DINOv3 ConvNeXt backbone."""
        return self.patch_encoder

    # ---- forward path ----

    @torch.no_grad()
    def _normalize_if_needed(self, images: torch.Tensor) -> torch.Tensor:
        # Preprocessor already normalises on CPU when normalize_on_gpu=False.
        return images

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Accept either:
          (B, num_image, 3, H, W) — direct tensor form (smoke tests), OR
          (B, num_image, n_patches, H*W*3) — the molmo2 data-pipeline form.
            For images: (n_image=1, n_patches=1).
            For videos: (n_image=n_clips, n_patches=frames_per_clip).
            All four leading dims fold into one "frame" axis for the backbone.
        Returns: (B*total_frames, latent_res*latent_res, D) patch features.
        Total frames per example = num_image * n_patches (with padding zeros for
        shorter samples — we still process them; loss masking handles correctness
        downstream).
        """
        cfg = self.config
        H = W = cfg.image_res
        if images.dim() == 4:
            B, n_image, n_patches = images.shape[:3]
            T = n_image * n_patches                                        # total frames per example
            frames = images.view(B * T, H, W, 3)
            frames = frames.permute(0, 3, 1, 2).contiguous()               # (B*T, 3, H, W)
        elif images.dim() == 5:
            B, n_image, C, H_, W_ = images.shape
            assert H_ == H and W_ == W, f"image_res mismatch: cfg={H} got={H_}"
            T = n_image
            frames = images.view(B * T, C, H_, W_).to(memory_format=torch.contiguous_format)
        else:
            raise ValueError(f"unexpected images shape: {tuple(images.shape)}")
        # Remember total frames so the soft-pool knows how to reshape.
        self._last_T = T
        feats = self.patch_encoder(
            frames,
            pool=cfg.backbone_pool,
            output_size=(cfg.latent_res, cfg.latent_res),
        )                                                                  # (B*T, D, h, w)
        feats = rearrange(feats, "bt d h w -> bt (h w) d")
        return feats

    def soft_pool_to_trajectories(
        self, patch_features: torch.Tensor, T: int
    ) -> torch.Tensor:
        """patch_features: (B*T, h*w, D) for B examples, T frames each.
        Returns: (B, num_traj, D) trajectory-token features pooled across t,h,w.
        """
        cfg = self.config
        D = patch_features.shape[-1]
        # reshape to per-example (B, T*h*w, D)
        BT, N, _ = patch_features.shape
        assert BT % T == 0, f"BT={BT} not divisible by T={T}"
        B = BT // T
        x = patch_features.view(B, T * N, D)

        # detached x is what feeds the perceiver (mirror seg pretraining)
        ctx = x.detach() if cfg.detach_patch_features_for_perceiver else x
        traj_queries = self.trajectory_perceiver(ctx)                      # (B, K, D)

        # Soft assignment (paper Eq. 1)
        # logits[b, n, k] = q_k · F_n. We use RAW dot product (not cosine) here:
        # with cosine + clamped scale ≤ 20, softmax over K=128 was nearly uniform
        # (~1.5× uniform peak), collapsing all z_k to the same vector. Raw dot
        # products scale as sqrt(D)×|q|×|f|, peaking the softmax naturally.
        # We add the learnable `log_scale` as a mild extra temperature.
        scale = F.softplus(self.log_scale) + 1.0                            # > 1
        scale = torch.clamp(scale, max=20.0)
        logits = torch.einsum("bnd,bkd->bnk", x, traj_queries)              # raw
        logits = logits * (scale / cfg.mask_softmax_temperature)
        soft_mask = F.softmax(logits, dim=-1)                               # (B, N, K)

        # Weighted pool (paper Eq. 2): z_k = Σ_n M[n,k] * F_n
        z = torch.einsum("bnk,bnd->bkd", soft_mask, x)                     # (B, K, D)
        return z

    def forward(
        self,
        images: torch.Tensor,
        image_masks: Optional[torch.Tensor] = None,
        token_pooling: Optional[torch.Tensor] = None,
        enable_cp: bool = False,
        cum_token_pooling_bounds: Optional[torch.Tensor] = None,
        cum_image_bounds: Optional[torch.Tensor] = None,
        image_shard_boundaries=None,
        **_,
    ) -> torch.Tensor:
        """Args mirror MolmoVisionBackbone.forward, but most are unused by us.

        images: (B, num_image, 3, H, W) float, normalised.
        Returns: (total_image_tokens_across_batch, d_model) flat tensor.
                 total_image_tokens_across_batch = B * num_image * num_traj
        """
        cfg = self.config
        B = images.shape[0]

        # ---- skip padded batch slots ----
        # With the mixed mixture (caption + pointing + NLP), some batch slots
        # have NO images. The data loader keeps `images` shape (B, n_image, ...)
        # and pads slots without an image (NLP-only samples). The padding values
        # depend on dtype and may be non-zero (e.g. after normalisation), so the
        # robust signal is `token_pooling >= 0`: padded rows of token_pooling
        # are all -1. We use it to identify which batch slots actually carry
        # a real image and emit features only for those.
        if token_pooling is not None:
            # token_pooling shape: (B, num_rows, pool_dim). For our preprocessor,
            # a real image contributes 128 rows of [0]; padding rows are all -1.
            per_sample_valid_rows = (token_pooling >= 0).any(dim=-1).sum(dim=-1)  # (B,)
            real_mask = per_sample_valid_rows > 0
        else:
            # Fallback heuristic — non-zero pixels.
            flat = images.reshape(B, -1)
            real_mask = (flat.abs().sum(dim=-1) > 0)
        if not bool(real_mask.all()):
            keep_idx = real_mask.nonzero(as_tuple=True)[0]
            if keep_idx.numel() == 0:
                d_model = self.image_projector.w2.out_features if hasattr(self.image_projector, "w2") \
                    else self.image_projector.weight.shape[0]
                return images.new_zeros((0, d_model))
            images = images.index_select(0, keep_idx)
            B = images.shape[0]

        # Patch features at latent_res. encode_image sets self._last_T = total
        # frames per example (n_image * n_patches for the 4D data-pipeline shape).
        patch_features = self.encode_image(images)                         # (B*T, h*w, D)
        patch_features = self.image_feature_dropout(patch_features)
        T_total = self._last_T

        # Soft-mask pool to trajectory tokens
        traj_features = self.soft_pool_to_trajectories(
            patch_features, T=T_total
        )                                                                  # (B, num_traj, D)

        # Normalise before projector so image_features land at ~unit scale
        # compatible with wte embeddings (~0.02 std × √d_model after LLM scaling).
        traj_features = self.pre_projector_ln(traj_features)

        # Project to LLM d_model
        z = self.image_projector(traj_features)                            # (B, num_traj, d_model)

        # Flatten across batch for the high-res-patch scatter in Molmo2.forward
        return z.reshape(-1, z.shape[-1])
