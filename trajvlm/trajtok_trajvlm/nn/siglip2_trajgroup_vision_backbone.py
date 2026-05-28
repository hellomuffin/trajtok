"""SigLIP2 ViT + Segmenter-based trajectory pooling vision backbone for Molmo2.

Architecture (TrajVLM paper, Zheng et al. 2026):

    video frames
       |
       +-> SigLIP2 ViT @ 378x378                                  -> F  (B, T*729, 1152)   [trainable]
       |
       +-> resize -> 224x224 -> SimpleSegmenter                   -> assignment_logits (B, T*56*56, 128)
                                (full segmenter ckpt loaded)         initial_queries_512   (B, 128, 512)
                                                                     [trainable, ~60M]
                                                                  |
                                                interp 56->27     |
                                                                  v
                                            assignment_logits (B, T*729, 128)
                                                                  |
                                  argmax -> predicted_label_map -> query_mask
                                                                  |
       initial_queries_512 -- Linear(512->1152) (NEW) --> initial_queries_1152 (B, 128, 1152)
                                                                  |
                                                                  v
                            TrajPerceiver cross-attn @ 1152-dim (NEW, depth=2)
                            queries attend to F with query_mask
                                                                  |
                                                                  v
                                               trajectory tokens (B, 128, 1152)
                                                                  |
                                              projector MLP 1152 -> d_llm  (NEW)
                                                                  v
                                                              LLM input

For video with T frames, we pool per `frames_per_pool` (default 8) frames:
each chunk produces 128 trajectory tokens. Total tokens per video =
ceil(T / frames_per_pool) * 128. Images (T=1) -> 128 tokens.

Init sources:
  - SigLIP2 ViT: from Molmo2's pretrained siglip2-so400m-14-384 (via .init_path)
  - SimpleSegmenter (DINOv3 + perceiver + assignment heads): from
    filteredmixdata_all/latest.pth (loaded via load_full_segmenter_into_backbone)
  - Linear 512->1152, TrajPerceiver, projector MLP: random init.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import os
from dataclasses import field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributed.fsdp import fully_shard

from olmo.config import BaseConfig
from olmo.nn.image_vit import SiglipVisionTransformer, VitConfig
from olmo.nn.vision_backbone import MolmoVisionBackboneConfig

# ---------------------------------------------------------------------------
# Import SimpleSegmenter + TrajPerceiver from the segmenter sibling package
# (trajtok-segmenter, installed via `pip install -e ./segmenter` from the
# trajtokv2 repo root).
#
# CRITICAL: the segmenter package's DINOResNetHierFeat may eagerly load
# pretrained weights to CUDA in __init__, which crashes when Molmo2
# constructs the parent model on meta device (under FSDP). We monkey-patch
# `trajtok_segmenter.model.backbones.dinov3_convnext.DINOResNetHierFeat`
# with our meta-safe variant (deferred weight load, no .to('cuda')) BEFORE
# importing the segmenter so SimpleSegmenter picks the safe version up.
# ---------------------------------------------------------------------------
import trajtok_segmenter.model.backbones.dinov3_convnext as _seg_dinov3  # noqa: E402
from trajtok_trajvlm.nn.trajvit_dinov3 import DINOResNetHierFeat as _MetaSafeDINOResNet  # noqa: E402
_seg_dinov3.DINOResNetHierFeat = _MetaSafeDINOResNet

from trajtok_segmenter.model.segmenter import SimpleSegmenter  # noqa: E402
from trajtok_segmenter.model.traj_perceiver import TrajPerceiver  # noqa: E402


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SigLip2TrajGroupVisionBackboneConfig(MolmoVisionBackboneConfig):
    """Config for the SigLIP2 + Segmenter trajectory-grouping backbone."""

    # Segmenter resolution (its trained native).
    segmenter_input_res: int = 224
    segmenter_latent_res: int = 56

    # Trajectory grouping
    num_traj: int = 128
    segment_embed_dim: int = 512        # segmenter's internal dim (matches ckpt)

    # Cross-attention TrajPerceiver (new, at SigLIP2 dim)
    cross_attn_depth: int = 2
    cross_attn_heads: int = 8

    # Pooling granularity along time (per how many frames do we collapse).
    frames_per_pool: int = 8

    # Soft-mask query-mask flags (mirror SegmentTokenizer defaults)
    attend_other_latents: bool = False
    num_latent_per_traj: int = 1

    # Optional: ckpt to warm-start the SimpleSegmenter from.
    pretrained_segmenter_path: Optional[str] = None

    # Misc
    connector_activation_checkpointing: bool = True
    compile_connector: Optional[str] = None

    # Segmenter sub-configs (used to build SimpleSegmenter); duck-typed dicts
    # so OmegaConf serialisation stays simple. Populated by build_model().
    segmenter_traj_model_kwargs: Optional[dict] = field(default_factory=dict)
    segmenter_backbone_kwargs: Optional[dict] = field(default_factory=dict)
    segmenter_perceiver_kwargs: Optional[dict] = field(default_factory=dict)

    # ---- Required hooks for MolmoVisionBackboneConfig contract ----

    def build_preprocessor(self):
        # We reuse the existing TrajVit image/video preprocessors which already
        # emit `num_traj` <im_patch> tokens per image/clip. They get configured
        # via TrajVitImageConfig / TrajVitVideoConfig in launch_scripts.
        # (Returning None here signals that the molmo2 builder should look at
        #  Molmo2PreprocessorConfig.image / .video instead.)
        return None

    def build(self, llm_config, device=None):
        return SigLip2TrajGroupVisionBackbone(self, llm_config, device=device)

    @classmethod
    def update_legacy_settings(cls, config):
        return config


# ---------------------------------------------------------------------------
# Projector MLP (SwiGLU, mirrors Molmo2's connector projector style)
# ---------------------------------------------------------------------------

class _ProjectorMLP(nn.Module):
    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        hidden = d_model * 2
        self.w1 = nn.Linear(in_dim, hidden, bias=False)
        self.w_gate = nn.Linear(in_dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def reset_parameters(self):
        nn.init.trunc_normal_(self.w1.weight, std=0.02)
        nn.init.trunc_normal_(self.w_gate.weight, std=0.02)
        nn.init.trunc_normal_(self.w2.weight, std=0.02)

    def forward(self, x):
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class SigLip2TrajGroupVisionBackbone(nn.Module):
    """SigLIP2 ViT features + trajectory pooling guided by SimpleSegmenter."""

    def __init__(self, config: SigLip2TrajGroupVisionBackboneConfig, llm_config, device=None):
        super().__init__()
        self.config = config

        # ---- SigLIP2 ViT (Molmo2 stock) ----
        # config.vit is a VitConfig (SIGLIP2_VISION_BACKBONE), input 378x378
        self.image_vit = SiglipVisionTransformer(config.vit, device=device)
        self.vit_emb_dim = config.vit.image_emb_dim       # 1152
        # SigLIP2 grid (e.g. 378/14 = 27)
        self.vit_grid_h, self.vit_grid_w = config.vit.image_num_patch

        # ---- SimpleSegmenter (full, with its DINOv3 + perceiver + assignment heads) ----
        # We construct via dict configs to avoid pulling in entry-side config classes.
        from easydict import EasyDict as _edict
        seg_cfg = _edict(self._default_segmenter_traj_cfg())
        seg_cfg.update(config.segmenter_traj_model_kwargs or {})
        bb_cfg = _edict(self._default_segmenter_backbone_cfg())
        bb_cfg.update(config.segmenter_backbone_kwargs or {})
        per_cfg = _edict(self._default_segmenter_perceiver_cfg())
        per_cfg.update(config.segmenter_perceiver_kwargs or {})
        self.segmenter = SimpleSegmenter(
            config=seg_cfg,
            backbone_config=bb_cfg,
            perceiver_config=per_cfg,
            high_res=False,                # we only need low-res assignment_logits
        )
        self.seg_emb_dim = config.segment_embed_dim   # 512

        # ---- Bridge: segmenter's 512-d trajectory queries -> SigLIP2's 1152-d ----
        self.query_proj = nn.Linear(self.seg_emb_dim, self.vit_emb_dim)

        # ---- Cross-attention TrajPerceiver at SigLIP2 dim (1152) ----
        # queries (trajectory latents) attend to SigLIP2 patches; query_mask
        # restricts each trajectory to its assigned patches (per segmenter argmax).
        # external_latent=True since we pass our own initial latents per-forward.
        self.cross_attn = TrajPerceiver(
            dim=self.vit_emb_dim,
            depth=config.cross_attn_depth,
            dim_head=self.vit_emb_dim // config.cross_attn_heads,
            heads=config.cross_attn_heads,
            num_latents=config.num_traj * config.num_latent_per_traj,
            use_rotary=True,
            use_latent_transformer=False,
            update_x=False,
            external_latent=True,
        )

        # ---- Projector to LLM d_model ----
        self.image_projector = _ProjectorMLP(self.vit_emb_dim, llm_config.d_model)

        # ---- Freeze segmenter sub-modules that feed ONLY into argmax-killed path ----
        # The cross-attn pool consumes the segmenter via two routes:
        #   (a) traj_features_512 → query_proj → cross_attn input_latent  [DIFFERENTIABLE]
        #   (b) assignment_logits → argmax → query_mask                    [argmax kills ∂]
        # Anything that contributes ONLY to (b) gets no gradient. Exclude those
        # from the optimizer entirely so the AdamW state stays consistent across
        # save/load (otherwise log_scale et al. show up in the model but never
        # accumulate an AdamW.step entry, breaking resume).
        for p in self.segmenter.traj_seg_head_low_res.parameters():
            p.requires_grad_(False)
        for p in self.segmenter.patch_decoder_low_res.parameters():
            p.requires_grad_(False)
        self.segmenter.log_scale.requires_grad_(False)

        # Move to device if it's a real device (not meta).
        if device is not None and getattr(device, "type", str(device)) != "meta":
            self.to(device)

    # ---- segmenter config defaults (match filteredmixdata_all ckpt training) ----

    def _default_segmenter_traj_cfg(self):
        return dict(
            model_name="vit-large",
            embed_dim=self.config.segment_embed_dim,
            segment_embed_dim=self.config.segment_embed_dim,
            latent_res=self.config.segmenter_latent_res,
            output_res=self.config.segmenter_latent_res,
            loss_func="dice_focal",     # unused at inference, but required at init
            num_traj=self.config.num_traj,
            total_latent_level=0,
            pretrained=False,
            pool="cls",
            pre_select_latents=True,
            attend_other_latents=False,
            rope_3d=False,
            no_high_res=True,
            add_mean_traj=False,
        )

    def _default_segmenter_backbone_cfg(self):
        # backbone_pretrained=False because SimpleSegmenter.build_vision_backbone
        # eagerly calls .to('cuda') on the loaded weights, which fails when the
        # parent Molmo2 model is being constructed on meta device. The DINOv3
        # weights come in via the full segmenter ckpt in reset_with_pretrained_weights.
        return dict(
            backbone_model="dinov3_small",
            backbone_pretrained=False,
            backbone_output_hierarchy=False,
            backbone_pool="sum",
            freeze=False,
        )

    def _default_segmenter_perceiver_cfg(self):
        return dict(depth=2)

    # ---- training-infra hooks ----

    def reset_parameters(self):
        self.query_proj.reset_parameters()
        self.image_projector.reset_parameters()
        if hasattr(self.cross_attn, "reset_parameters"):
            self.cross_attn.reset_parameters()

    def reset_with_pretrained_weights(self):
        # 1) SigLIP2 ViT weights via stock molmo2 path
        if hasattr(self.image_vit, "reset_with_pretrained_weights"):
            self.image_vit.reset_with_pretrained_weights()
        # 2) SimpleSegmenter weights from the seg pretraining ckpt. Pass `self`
        #    (the backbone) so the loader's `segmenter.<sub>` rekey lines up
        #    with `vision_backbone.segmenter.<sub>` in the parent state_dict.
        if self.config.pretrained_segmenter_path:
            from trajtok_trajvlm.checkpoints.load_trajvit_segmenter_full import load_full_segmenter
            load_full_segmenter(self, self.config.pretrained_segmenter_path)
        # 3) New random-init pieces (query_proj, cross_attn, projector) already
        #    initialized in __init__; no-op here.

    def apply_fsdp2(self, **kwargs):
        # SigLIP2 ViT: shard each resblock individually (Molmo2 standard).
        for block in self.image_vit.transformer.resblocks:
            fully_shard(block, **kwargs)
        fully_shard(self.image_vit, **kwargs)
        # Cross-attention + projector: shard at module level.
        fully_shard(self.cross_attn, **kwargs)
        fully_shard(self.image_projector, **kwargs)
        # SimpleSegmenter: do NOT shard (its internal LayerNorms aren't FSDP-friendly
        # — same issue we saw with DINOv3 in trajvit_vision_backbone). Replicated.

    def apply_activation_checkpointing(self):
        if hasattr(self.image_vit, "apply_activation_checkpointing"):
            self.image_vit.apply_activation_checkpointing()
        if self.config.connector_activation_checkpointing:
            try:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
                self.cross_attn = checkpoint_wrapper(self.cross_attn)
                self.image_projector = checkpoint_wrapper(self.image_projector)
            except Exception as e:
                log.warning(f"checkpoint_wrapper unavailable: {e}")

    def apply_compile(self, **kwargs):
        pass  # disabled; mirrors trajvit_vision_backbone

    def get_connector_parameters(self):
        vit_params = set(self.image_vit.parameters())
        return (p for p in self.parameters() if p not in vit_params)

    # ---- forward path ----

    def _siglip2_forward(self, images_378: torch.Tensor) -> torch.Tensor:
        """Run SigLIP2 ViT.
        images_378: (B*T, 3, 378, 378) RGB normalised.
        Returns: (B*T, num_patches=729, 1152) last-layer hidden states.
        """
        B_T, C, H, W = images_378.shape
        ps = self.config.vit.image_patch_size
        # Patch into (B*T, n_patches, n_pixels) for SiglipVisionTransformer
        # (it does a Linear on the per-patch pixels).
        # Rearrange (B*T, 3, H, W) -> (B*T, n_h*n_w, ps*ps*3)
        x = rearrange(
            images_378,
            "b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)",
            ph=ps, pw=ps,
        )
        out = self.image_vit(x)
        # SiglipVisionTransformer returns last hidden states (B*T, n_patches, d_vit).
        # Some implementations return a tuple/list per layer — handle both.
        if isinstance(out, (list, tuple)):
            out = out[-1]
        return out                              # (B*T, 729, 1152)

    def _segmenter_forward(
        self, video_224: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the segmenter, exposing both the assignment logits and the
        segmenter's internal trajectory queries.

        video_224: (B, T_chunk, 3, 224, 224)
        Returns:
          assignment_logits  (B, T_chunk*56*56, 128) scaled
          traj_features_512  (B, 128, 512)
        """
        bs, T, _, H, W = video_224.shape
        seg = self.segmenter
        video_frames = rearrange(video_224, "b t c h w -> (b t) c h w")
        patch_features = seg.patch_encoder(
            video_frames,
            pool=seg.backbone_pool,
            output_size=(seg.latent_res, seg.latent_res),
        )                                                       # (B*T, D=512, w, h)
        patch_features = rearrange(
            patch_features, "(b t) d w h -> b (t w h) d", b=bs, t=T
        )                                                       # (B, T*56*56, 512)
        # Initial trajectory queries — `.detach()` mirrors the segmenter's
        # training-time forward, but for SFT the user wants the segmenter
        # end-to-end trainable, so DO NOT detach here.
        traj_features_512 = seg.trajectory_perceiver(patch_features)   # (B, 128, 512)

        # Assignment logits via segmenter's heads
        traj_seg_feat = seg.traj_seg_head_low_res(traj_features_512)            # (B, 128, D/8)
        up_patch_feat = seg.patch_decoder_low_res(patch_features)               # (B, N, D/8)
        assign_logits = seg.compute_assignment_logits(
            up_patch_feat, traj_seg_feat
        )                                                                       # (B, N, 128)
        scale = F.softplus(seg.log_scale) + 1.0
        scale = torch.clamp(scale, max=20.0)
        scaled_logits = assign_logits * scale
        return scaled_logits, traj_features_512

    def _build_query_mask(self, assign_idx: torch.Tensor) -> torch.Tensor:
        """assign_idx: (B, N_after_interp).
        Returns key_padding_mask: (B, num_traj, N_after_interp), True = blocked.
        Mirrors SegmentTokenizer.make_query_mask with num_latent_per_traj=1.
        """
        B, N = assign_idx.shape
        M = self.config.num_traj
        device = assign_idx.device
        class_ids = torch.arange(M, device=device).view(1, M, 1)
        # True where the patch does NOT belong to this trajectory.
        mask = assign_idx.unsqueeze(1) != class_ids                # (B, M, N)
        # Hedge against fully-masked rows (every query softmax must see >=1 key,
        # otherwise the all-(-inf) softmax produces NaN). Always unmask the
        # last 2 patches — mirrors SegmentTokenizer.make_query_mask.
        mask[:, :, -2:] = False
        return mask

    def _pool_one_chunk(
        self,
        siglip_F: torch.Tensor,            # (B, T_chunk*729, 1152)
        assign_logits_full: torch.Tensor,  # (B, T_chunk*56*56, 128) (raw)
        traj_features_512: torch.Tensor,   # (B, 128, 512)
        T_chunk: int,
    ) -> torch.Tensor:
        """Produce 128 trajectory tokens for one (B, T_chunk, ...) chunk."""
        B = siglip_F.shape[0]
        # ---- Interpolate assignment masks from 56x56 to 27x27 (SigLIP2 grid) ----
        seg_res = self.config.segmenter_latent_res                  # 56
        vit_h, vit_w = self.vit_grid_h, self.vit_grid_w             # 27, 27
        x = rearrange(
            assign_logits_full,
            "b (t h w) k -> (b t) k h w",
            t=T_chunk, h=seg_res, w=seg_res,
        )
        x = F.interpolate(x, size=(vit_h, vit_w), mode="bilinear", align_corners=False)
        assign_at_vit = rearrange(
            x, "(b t) k h w -> b (t h w) k", b=B, t=T_chunk,
        )                                                            # (B, T*729, 128)

        # ---- argmax -> query_mask ----
        predicted_label = assign_at_vit.argmax(dim=-1)               # (B, T*729)
        query_mask = self._build_query_mask(predicted_label)         # (B, M=128, T*729)

        # ---- Up-project segmenter's 512-d initial queries to 1152 ----
        init_q_1152 = self.query_proj(traj_features_512)             # (B, 128, 1152)

        # ---- Cross-attention pooling ----
        # TrajPerceiver expects (x, T, H, W, attention_mask, input_latent).
        # We treat the SigLIP2 tokens as a single (T_chunk * vit_h * vit_w) sequence
        # for the rotary embedding spatial decomposition.
        out = self.cross_attn(
            siglip_F,
            T=T_chunk,
            H=vit_h,
            W=vit_w,
            attention_mask=(None, query_mask),       # SegmentTokenizer-style tuple
            input_latent=init_q_1152,
        )                                                            # (B, 128, 1152)
        return out

    def forward(
        self,
        images: torch.Tensor,
        image_masks: Optional[torch.Tensor] = None,
        token_pooling: Optional[torch.Tensor] = None,
        enable_cp: bool = False,
        cum_token_pooling_bounds=None,
        cum_image_bounds=None,
        image_shard_boundaries=None,
        **_,
    ) -> torch.Tensor:
        """
        images: (B, num_image, n_patches, H*W*3) — molmo2 data-pipeline form,
                pre-normalised at SigLIP2's image_res (378 by default).
                For images: (n_image=1, n_patches=1).
                For videos: (n_image=n_clips, n_patches=frames_per_clip).
        Returns: (total_image_tokens_across_batch, d_model) flat tensor.
        """
        cfg = self.config
        vit_res = cfg.vit.image_default_input_size[0]
        seg_res = cfg.segmenter_input_res

        # ---- skip padded batch slots (mirror trajvit backbone) ----
        B = images.shape[0]
        if token_pooling is not None:
            per_sample_valid = (token_pooling >= 0).any(dim=-1).sum(dim=-1)
            real_mask = per_sample_valid > 0
        else:
            real_mask = (images.reshape(B, -1).abs().sum(dim=-1) > 0)
        if not bool(real_mask.all()):
            keep_idx = real_mask.nonzero(as_tuple=True)[0]
            if keep_idx.numel() == 0:
                d_model = self.image_projector.w2.out_features
                return images.new_zeros((0, d_model))
            images = images.index_select(0, keep_idx)
            B = images.shape[0]

        # ---- Reshape molmo2's data-pipeline form into (B*T_total, 3, H, W) ----
        # images: (B, n_image, n_patches, H*W*3). For TrajVit preprocessor each
        # "patch" is the full frame, so n_patches doubles as frames_per_clip
        # (or 1 for image). T_total = n_image * n_patches = total frames per ex.
        Bdim, n_image, n_patches, _ = images.shape
        T_total = n_image * n_patches

        # ---- Decide chunking: each pool spans frames_per_pool frames ----
        # n_chunks * T_chunk = T_total (T_chunk == frames_per_pool except when
        # T_total is small — for T_total < frames_per_pool we use a single
        # chunk of size T_total, so images (T=1) still produce exactly 128 tokens.
        Fpp = cfg.frames_per_pool
        if T_total < Fpp:
            T_chunk = T_total
        else:
            assert T_total % Fpp == 0, (
                f"T_total={T_total} must be divisible by frames_per_pool={Fpp}; "
                f"adjust video preprocessor (num_frames, frames_per_clip)."
            )
            T_chunk = Fpp
        n_chunks = T_total // T_chunk

        # Reshape (B, T_total, 3, H, W) -> (B*n_chunks, T_chunk, 3, H, W) so each
        # chunk is an independent batch element for both SigLIP2 and segmenter.
        frames_rgb = images.reshape(Bdim, T_total, vit_res, vit_res, 3)
        frames_rgb = frames_rgb.permute(0, 1, 4, 2, 3).contiguous()    # (B, T_total, 3, H, W)
        frames_rgb = frames_rgb.view(Bdim * n_chunks, T_chunk, 3, vit_res, vit_res)
        B_eff = Bdim * n_chunks

        # ---- SigLIP2 forward on all (B_eff * T_chunk) frames ----
        frames_for_vit = frames_rgb.view(B_eff * T_chunk, 3, vit_res, vit_res)
        F_siglip = self._siglip2_forward(frames_for_vit)               # (B_eff*T_chunk, 729, 1152)
        F_siglip = F_siglip.view(B_eff, T_chunk * F_siglip.shape[1], F_siglip.shape[2])
                                                                       # (B_eff, T_chunk*729, 1152)

        # ---- Down-resize for segmenter (224x224) ----
        if vit_res != seg_res:
            frames_for_seg = F.interpolate(
                frames_for_vit, size=(seg_res, seg_res), mode="bilinear", align_corners=False,
            )
        else:
            frames_for_seg = frames_for_vit
        frames_for_seg = frames_for_seg.view(B_eff, T_chunk, 3, seg_res, seg_res)

        # ---- Segmenter forward (one batched call across all chunks) ----
        assign_logits, traj_q_512 = self._segmenter_forward(frames_for_seg)
        # assign_logits: (B_eff, T_chunk*56*56, 128); traj_q_512: (B_eff, 128, 512)

        # ---- Cross-attention trajectory pooling (one batched call) ----
        tokens = self._pool_one_chunk(
            F_siglip, assign_logits, traj_q_512, T_chunk=T_chunk,
        )                                                              # (B_eff, 128, 1152)

        # ---- Reshape back to (B, n_chunks*128, 1152) and project ----
        tokens = tokens.view(Bdim, n_chunks * cfg.num_traj, self.vit_emb_dim)
        out = self.image_projector(tokens)                             # (B, n_chunks*128, d_llm)

        return out.reshape(-1, out.shape[-1])
