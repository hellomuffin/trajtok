import torch
from torch import nn
from einops import rearrange, repeat
from easydict import EasyDict as edict
import logging
import torch.nn.functional as F
import math
import random
import os
import time
import torch.distributed

from torch.distributed.tensor import DTensor
from trajtok_segmenter.model.backbones.dinov3_convnext import CLIPResNetHierFeat, DINOResNetHierFeat, SigLipFeat
from trajtok_segmenter.model.perceiver_resampler import PerceiverResampler
from trajtok_segmenter.model.traj_perceiver import TrajPerceiver
from trajtok_segmenter.model.sam_mask_decoder import SamPatchDecoder, SamTrajectoryPerceiver, make_traj_seg_head
from trajtok_segmenter.model.traj_transformer import CustomTransformer
from trajtok_segmenter.train.hungarian import hungarian_per_sample, hungarian_per_batch, focal_loss, dice_loss
import omegaconf
import typing
import collections
import enum



logger = logging.getLogger(__name__)
# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)



    
def simple_load_pretrained(
    encoder,
    pretrained,
    checkpoint_key='model',
):
    if pretrained is None: 
        logger.info("pretrained model path is None, return")
        return encoder
    
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu', weights_only=False)
    
    pretrained_dict = checkpoint[checkpoint_key]

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

    m_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('vision_encoder.vision_encoder.'): k = k[len('vision_encoder.'):]
        elif k.startswith('text_encoder'): continue
        else: k = k.replace('vision_encoder.', '')
        m_pretrained_dict[k] = v
    pretrained_dict = m_pretrained_dict    
    

    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v

    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f'loaded pretrained encoder with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder


import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def make_orthonormal(rows: int, cols: int, device=None, dtype=torch.float32):
    """
    Returns M in R^{rows x cols}.
    - If rows >= cols: columns are orthonormal (M^T M = I_cols).
    - If rows <  cols: rows are    orthonormal (M M^T = I_rows).
    """
    if rows >= cols:
        A = torch.randn(rows, cols, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(A, mode='reduced')
        return Q
    else:
        A = torch.randn(cols, rows, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(A, mode='reduced')
        return Q.T

@torch.no_grad()
def init_latents_ring_vectorized(
    n: int, m: int, d: int,
    max_freq: int = 4,
    scale: float = 0.02,
    device=None,
    dtype=torch.float32,
):
    """
    Create latents of shape (n, m, d) with:
      - per-row evenly spaced angles (m points on a ring),
      - multi-frequency sin/cos features (1..max_freq),
      - shared orthonormal projection to R^d,
      - per-row random phase to avoid identical rows,
      - tiny noise + scale.
    """
    device = device or torch.device('cpu')
    Freq = max_freq
    B = 2 * Freq  # sin+cos per frequency

    # Even angles for m points on [0, 2π)
    base_angles = torch.linspace(0.0, 2*math.pi, steps=m+1, device=device, dtype=dtype)[:-1]  # (m,)
    # Per-row random phase offsets ∈ [0, 2π)
    # phases = torch.rand(n, 1, device=device, dtype=dtype) * (0.5*math.pi)                       # (n,1)
    theta = torch.stack([base_angles] * n)                                      # (n,m)

    ks = torch.arange(1, Freq + 1, device=device, dtype=dtype)                                 # (F,)
    ktheta = theta.unsqueeze(-1) * ks.view(1, 1, -1)                                           # (n,m,F)

    s = torch.sin(ktheta)                                                                      # (n,m,F)
    c = torch.cos(ktheta)                                                                      # (n,m,F)
    bank = torch.cat([s, c], dim=-1)                                                           # (n,m,2F)
    bank = F.normalize(bank, dim=-1)

    # Orthonormal map from R^{2F} -> R^{d} (shared across rows)
    W = make_orthonormal(B, d, device=device, dtype=dtype)                                     # (2F, d)

    latents = bank @ W                                                                          # (n,m,d)
    latents = latents + 1e-3 * torch.randn_like(latents)                                       # tiny jitter
    latents = latents * scale
    return latents.to(dtype)







class SimpleSegmenter(nn.Module):
    
    def __init__(self, config=None, backbone_config=None, perceiver_config=None, embed_dim=None, high_res=True):
        super(SimpleSegmenter, self).__init__()
        
        self.config = edict(config)
        self.backbone_config = edict(backbone_config)
        self.perceiver_config = edict(perceiver_config)
        
        # self.embed_dim = self.config.embed_dim if embed_dim is None else embed_dim
        self.embed_dim = self.segment_embed_dim = self.config.segment_embed_dim
        
        self.output_res = self.config.output_res
        self.latent_res = self.config.latent_res
        self.loss_func_type = self.config.loss_func.split("_") #TODO 'ce' or 'dice' or 'focal', combination should be separated by "_"
        self.rope_3d = self.config.rope_3d
        
        self.num_traj = self.config.num_traj
        
        self.backbone_model = self.backbone_config.backbone_model
        self.backbone_pretrained = self.backbone_config.backbone_pretrained
        self.backbone_output_hierarchy = self.backbone_config.backbone_output_hierarchy
        self.backbone_pool = self.backbone_config.backbone_pool
        
        
        self.patch_encoder = self.build_vision_backbone()
        
        self.traj_seg_head_low_res = nn.Sequential(
            nn.Linear(self.segment_embed_dim, self.segment_embed_dim//4),
            nn.ReLU(),
            nn.Linear(self.segment_embed_dim//4, self.segment_embed_dim//8),
        )
        
        updater_cls = TrajPerceiver if self.rope_3d else PerceiverResampler
        
        self.trajectory_perceiver =  updater_cls(
            dim=self.segment_embed_dim,
            depth=self.perceiver_config.depth, 
            dim_head=self.segment_embed_dim // 8,
            heads=8,
            num_latents=self.num_traj, 
            use_rotary=True,
            use_latent_transformer=False,
            update_x=False
        )
        
        self.patch_decoder_low_res = nn.Sequential(
            nn.Linear(self.segment_embed_dim, self.segment_embed_dim//4),
            nn.ReLU(),
            nn.Linear(self.segment_embed_dim//4, self.segment_embed_dim//8),
        )

        if high_res:
            self.patch_decoder_high_res = SamPatchDecoder(
                embed_dim = self.segment_embed_dim,
                latent_res = self.latent_res,
                output_res = self.output_res,   # 1× / 2× / 4×
            )
            self.traj_seg_head_high_res = nn.Sequential(
                nn.Linear(self.segment_embed_dim, self.segment_embed_dim//4),
                nn.ReLU(),
                nn.Linear(self.segment_embed_dim//4, self.segment_embed_dim//8),
            )
                    
        # in __init__
        self.log_scale = nn.Parameter(torch.tensor([math.log(16.0)]))
        
        
        
    def build_vision_backbone(self):
        if self.backbone_model == 'resnet50_clip':
            patch_encoder = CLIPResNetHierFeat(
                out_channel=self.embed_dim,
                upsample_first=True,
                pretrained=True
            )
        elif self.backbone_model == 'dinov3':
            patch_encoder = DINOResNetHierFeat(
                model_size='base',
                out_channel=self.embed_dim,
                upsample_first=True,
            )
        elif self.backbone_model == 'dinov3_small':
            patch_encoder = DINOResNetHierFeat(
                model_size='small',
                out_channel=self.embed_dim,
                upsample_first=True,
            )
        elif self.backbone_model == 'siglip2':
            patch_encoder = SigLipFeat(
                out_channel=self.embed_dim,
            )
            
        elif self.backbone_model is None:
            patch_encoder = None
        else:
            raise NotImplementedError
        return patch_encoder
    
            
    
    def reset_parameters(self):
        """Reset parameters for SimpleSegmenter and its submodules"""
        # Reset patch encoder - all backbone types should have reset_parameters now
        if self.patch_encoder is not None:
            self.patch_encoder.reset_parameters()
        
        # Reset trajectory perceiver
        self.trajectory_perceiver.reset_parameters()
        
        # Reset patch decoder high res
        self.patch_decoder_high_res.reset_parameters()
        
        # Reset segmentation heads and patch decoder low res (nn.Sequential modules)
        for module in [self.traj_seg_head_low_res, self.traj_seg_head_high_res, self.patch_decoder_low_res]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        
        # Reset log_scale parameter
        nn.init.constant_(self.log_scale, math.log(16.0))

    def compute_assignment_logits(self, x, queries):
        q = F.normalize(x, dim=-1)  # (B, N, D)
        p = F.normalize(queries, dim=-1)  # (B, M, D)
        logits = torch.matmul(q, p.transpose(1, 2)) # (B,N,M)
        return logits
    
    
    def forward(self, video=None, return_intermediate_feature = False, input_patch_feature=None, output_high_res=False):
        """segmenter forward
            video (torch.tensor, optional): shape [b, t, 3, w, h]. Defaults to None.
            input_patch_feature (torch.tensor, optional): shape [b, twh, d]. w and h should be the same with self.latent_res. Defaults to None.
        """
        
        assert (video is None or input_patch_feature is None), "can't provide both video input and patch feature input"
        if video is not None:
            if len(video.shape) == 5:
                bs, T = video.shape[0], video.shape[1]
                video_frames = rearrange(video, 'b t d w h -> (b t) d w h')
            elif len(video.shape) == 4:
                bs, T = video.shape[0], 1
                video_frames = video
            patch_features = self.patch_encoder(video_frames, pool=self.backbone_pool, output_size=(self.latent_res, self.latent_res))  
            patch_features = rearrange(patch_features, '(b t) d w h -> b (t w h) d', b=bs, t=T)
        elif input_patch_feature is not None:
            patch_features = input_patch_feature
        else:
            assert False, "either video input or patch feature input should be provided"

        if self.rope_3d: traj_features = self.trajectory_perceiver(patch_features.detach(), T, self.latent_res, self.latent_res)
        else: traj_features = self.trajectory_perceiver(patch_features.detach())
        
        traj_seg_feat_low_res = self.traj_seg_head_low_res(traj_features)
        
        # in forward (both low & high res)
        scale = F.softplus(self.log_scale) + 1.0            # >= 1
        scale = torch.clamp(scale, max=20.0)

        up_patch_feat_low_res = self.patch_decoder_low_res(patch_features)
        assignment_logits_low_res = self.compute_assignment_logits(up_patch_feat_low_res, traj_seg_feat_low_res)
        scaled_logits = scaled_logits_low_res = assignment_logits_low_res * scale

        if output_high_res:
            traj_seg_feat_high_res = self.traj_seg_head_high_res(traj_features)
            up_patch_feat_high_res = self.patch_decoder_high_res(patch_features, T=T)
            assignment_logits_high_res = self.compute_assignment_logits(up_patch_feat_high_res, traj_seg_feat_high_res)
            scaled_logits_high_res = assignment_logits_high_res * scale
            scaled_logits = [scaled_logits_low_res, scaled_logits_high_res]
            
        if return_intermediate_feature: return scaled_logits, patch_features
        return scaled_logits


    def compute_assignment_loss(
        self,
        logits: torch.Tensor,       # (B,N,M)
        mask: torch.LongTensor,     # (B,T,H,W)
        video_graph: torch.LongTensor,
        ignore_index: int = -1,
        low_res_variant = False,
        
    ) -> torch.Tensor:
        B,N,M = logits.shape
        
        labels, valid = self.pixel_labels(mask, video_graph, ignore_index, resize_res=self.latent_res if low_res_variant else None)
        # --- match & reorder for the whole batch in one shot
        perms = hungarian_per_batch(logits, labels, valid, ignore_index=ignore_index)  # (B, M)
        logits_aligned = torch.gather(
            logits, 2, perms.unsqueeze(1).expand(-1, logits.size(1), -1)
        )

        assert 'dice' in self.loss_func_type
        
        class_loss = dice_loss(logits_aligned, labels, ignore_index=ignore_index)
            
        if "ce" in self.loss_func_type:
            loss_func = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
            pixel_loss = loss_func(logits_aligned.transpose(1, 2), labels)
        elif "focal" in self.loss_func_type:
            pixel_loss = focal_loss(logits_aligned, labels, ignore_index=ignore_index, gamma=2)
        else:
            pixel_loss = torch.zeros_like(class_loss)
        
        return pixel_loss, class_loss


    def compute_assignment_loss_from_labels(
        self,
        logits: torch.Tensor,       # (B,N,M)
        labels: torch.LongTensor,   # (B,N)
        valid: torch.BoolTensor,    # (B,M) or Long/Byte mask
        ignore_index: int = -1,
    ) -> torch.Tensor:
        """
        Compute segmentation losses given precomputed pixel labels and valid traj mask.

        Args:
            logits: (B, N, M) assignment logits
            labels: (B, N) pixel-to-traj labels in [0..M-1] or ignore_index
            valid : (B, M) boolean mask where False denotes padding trajs
        Returns:
            pixel_loss, class_loss
        """
        B, N, M = logits.shape

        # ensure boolean for valid mask
        if valid.dtype != torch.bool:
            valid = valid.to(torch.bool)

        # batched Hungarian: single GPU->CPU sync for the whole batch
        perms = hungarian_per_batch(logits, labels, valid, ignore_index=ignore_index)
        logits_aligned = torch.gather(
            logits, 2, perms.unsqueeze(1).expand(-1, logits.size(1), -1)
        )

        assert 'dice' in self.loss_func_type
        class_loss = dice_loss(logits_aligned, labels, ignore_index=ignore_index)

        if "ce" in self.loss_func_type:
            loss_func = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
            pixel_loss = loss_func(logits_aligned.transpose(1, 2), labels)
        elif "focal" in self.loss_func_type:
            pixel_loss = focal_loss(logits_aligned, labels, ignore_index=ignore_index, gamma=2)
        else:
            pixel_loss = torch.zeros_like(class_loss)

        return pixel_loss, class_loss


    def pixel_labels(
        self,
        mask: torch.LongTensor,         # (B,T,H,W)  object-IDs per pixel
        video_graph: torch.LongTensor,  # (B,M,T)    object-IDs per traj & time
        ignore_index: int = -1,
        resize_res = None
    ):
        """
        Returns
        labels  : (B,N)  each entry ∈ {0..M-1} or ignore_index
        valid   : (B,M)  valid[b,m]=False if traj m is *padding* for batch-b
        """
        
        if resize_res is not None: mask = F.interpolate(mask, size=(resize_res, resize_res), mode='nearest')
        
        B,T,H,W = mask.shape
        B_,M,T_ = video_graph.shape
        assert (B,T) == (B_,T_), "shape mismatch"
        
        video_graph[video_graph==0] = ignore_index
        
        mask_f = mask.view(B, T, -1)                         # (B,T,HW)
        # (B,M,T,HW)  matches[b,m,t,n] == (traj m’s ID at t) == (pixel n’s ID at t)
        matches = video_graph.unsqueeze(-1).eq(
                mask_f.unsqueeze(1))                       # bool
        valid_traj = (video_graph != ignore_index).any(-1)   # (B,M) bool
        
        has_match = matches.any(1)                           # (B,T,HW)
        label_map = torch.where(
            has_match,
            matches.float().argmax(1),  # (B,T,HW), int64
            torch.full((B,T,H*W), ignore_index, device=video_graph.device, dtype=torch.long)
        )
        assert label_map.min() == ignore_index
        # shapes
        B, M, T, HW = matches.shape
        THW = T * HW
        dev = matches.device

        # --- compute first-hit index per traj (b,m) over THW with correct dtype ---
        matches_flat = matches.view(B, M, THW)                            # bool
        idx = torch.arange(THW, device=dev, dtype=torch.int64).view(1,1,-1)
        sentinel = torch.full((B, M, THW), THW + 1, device=dev, dtype=torch.int64)

        first_idx = torch.where(matches_flat, idx, sentinel).amin(dim=-1) # (B,M) int64
        has_any  = matches_flat.any(dim=-1)                               # (B,M)
        first_idx = torch.where(has_any, first_idx,
                                torch.full_like(first_idx, THW + 1))      # push never-hit to end

        # push padding/invalid trajs even further
        first_idx = torch.where(valid_traj, first_idx,
                                torch.full_like(first_idx, THW + 2))

        # --- sort trajs by earliest first-hit (stable) ---
        perm = first_idx.argsort(dim=1, stable=True)                      # (B,M) new->old

        # inverse map: invperm[b, old] = new
        invperm = torch.empty_like(perm)
        invperm.scatter_(1, perm, torch.arange(M, device=dev).view(1,-1).expand(B,-1))

        # --- remap labels to new indices (keep ignore_index intact) ---
        labels_flat = label_map.view(B, THW).to(torch.long)               # old indices
        is_bg = labels_flat.eq(ignore_index)
        labels_clamped = labels_flat.clamp_min(0)
        labels_new = invperm.gather(1, labels_clamped)                    # old -> new
        labels_new = torch.where(is_bg, torch.full_like(labels_new, ignore_index), labels_new)
        label_map = labels_new.view(B, T, HW)

        # (optional) reorder valid_traj to the new order
        valid_traj = valid_traj.gather(1, perm)
        return label_map.view(B, -1), valid_traj  # (B, N), (B, M)


    
import yaml
class SegmentTokenizer(nn.Module):
    def __init__(self, config_info,  use_external_patch_features=False, input_dim=None, output_dim=None, total_latent_level=0, use_latent=True, rearrange_latent=False, rope_3d=False, use_pos_branch=False, use_gt_seg=False, add_mean_traj=False):
        super().__init__()
       
        if type(config_info) == str: 
            with open(config_info) as f: config = yaml.safe_load(f)
            self.config = edict(config['traj_model'])
            self.backbone_config = edict(config['backbone'])
            self.perceiver_config=edict(config['perceiver'])
            self.segmenter_input_res = config['image_res']
        else: 
            self.config, self.backbone_config, self.perceiver_config = config_info
            self.segmenter_input_res = 224
        
        self.latent_res = self.config.latent_res
        self.total_latent_level = total_latent_level
        num_latent_per_traj = 2 ** total_latent_level
        
        self.config.pre_select_latents = True
        
        self.segmenter = SimpleSegmenter(
            config=self.config,
            backbone_config=self.backbone_config,
            perceiver_config=self.perceiver_config,  
            high_res=False
        )
        self.embed_dim = self.config.embed_dim if input_dim is None else input_dim
        self.output_dim = self.config.embed_dim if output_dim is None else output_dim
        self.use_external_patch_features = use_external_patch_features
        self.use_gt_seg = use_gt_seg

        # If using external patch features (trajectory video mode), the internal segmenter is still trainable
        # so that optimizer state includes it and training can finetune the segmentation head.
        # (Do NOT ce here.)
        
        self.rearrange_latent = rearrange_latent
        self.use_pos_branch = use_pos_branch
        
        fourier_init = init_latents_ring_vectorized(n=self.segmenter.num_traj, m=num_latent_per_traj, d=self.embed_dim, max_freq=2, )
        
        self.rope_3d = rope_3d
        updater_cls = TrajPerceiver if rope_3d else PerceiverResampler
        
        print("use_rope_3d?", rope_3d, "use_pos_branch?", use_pos_branch, "rearrange_latent?", rearrange_latent)
        self.traj_feature_updater = updater_cls(
            dim=self.embed_dim,
            depth=self.perceiver_config.depth, 
            dim_head=self.embed_dim // 8,
            heads=8,
            num_latents=self.segmenter.num_traj * num_latent_per_traj, 
            use_rotary=True,
            use_latent_transformer=False,
            update_x=False,
            external_latent=True
        )
        if self.use_pos_branch:
            self.pos_feature_updater = TrajPerceiver(
                dim=self.embed_dim,
                depth=2, 
                dim_head=self.embed_dim // 8,
                heads=8,
                num_latents=self.segmenter.num_traj * num_latent_per_traj, 
                use_rotary=True,
                use_latent_transformer=False,
                update_x=False,
                external_latent=True
            )
            self.pos_latents = nn.Parameter(fourier_init)
        
        if use_latent: self.latents = nn.Parameter(fourier_init)
        else: self.latents = None
        
        self.add_mean_traj = add_mean_traj
        
        self.output_projector = nn.Linear(self.embed_dim, self.output_dim)
        
    
    
    def load_pretrained_segmenter(self, ckpt_path):
        self.segmenter = simple_load_pretrained(self.segmenter, ckpt_path)
    
    
    def make_query_mask(
            self,
            assign_idx: torch.Tensor,     # (B, N)
            num_traj:    int,             # M  (trajectory / class count)
            num_latent_per_traj: int = 1, # K  (latents per trajectory)
            attend_other_latents: bool = False
    ):
        """
        Returns a boolean key‑padding mask.

            B  – batch
            M  – number of trajectories / classes
            K  – number of latent queries per trajectory
            N  – number of patches
            L  = M*K – total latent tokens

        Output shapes (B, L, L + N)
        """
        B, N = assign_idx.shape
        device = assign_idx.device
        L = num_traj * num_latent_per_traj                        # total latents

        class_ids   = torch.arange(num_traj, device=device).view(1, num_traj, 1)
        traj_patch_mask = assign_idx.unsqueeze(1) != class_ids    # (B, M, N)
        traj_patch_mask[:, :, -2:] = 0

        patch_mask = traj_patch_mask.repeat_interleave(num_latent_per_traj, dim=1)               # (B, L, N)

        if attend_other_latents:
            latent_mask = torch.zeros(B, L, L, dtype=torch.bool, device=device)
        else:
            # 1.  Row-index for every flattened position:  [0 0 …0  1 1 …1  … m-1 m-1 …]
            row_id = torch.arange(num_traj, device=device).repeat_interleave(num_latent_per_traj)         # shape (L,)
            # 2.  Build boolean mask: True  ⇒  *block* that attention score
            latent_mask = row_id.unsqueeze(0) != row_id.unsqueeze(1)  # (L, L)
            latent_mask = repeat(latent_mask, 'n m -> b n m', b = B)
            
        return latent_mask, patch_mask
    

    def resize_segmentation_logits(self, prob, num_frames):
        # prob: (b, twh, num_traj)
        scale_factor = self.latent_res / self.segmenter.output_res
        bs = prob.shape[0]
        # 1) move class channel to PyTorch's expected position (N,C,H,W)
        prob_4d = rearrange(prob, "b (t w h) d -> (b t) d w h", t=num_frames, w=self.segmenter.output_res, h=self.segmenter.output_res).contiguous()   # (T,C,W, H)

        # 2) treat T as batch, interpolate
        prob_resized_4d = F.interpolate(prob_4d,
                                scale_factor=scale_factor,
                                mode="bilinear",
                                align_corners=False)
        prob_resized = rearrange(prob_resized_4d, "(b t) d w h -> b (t w h) d", b=bs, t=num_frames)
        return prob_resized
    

    def resize_video(self, x: torch.Tensor, size: tuple[int, int], mode="bilinear") -> torch.Tensor:
        """
        Resize video tensor from (B, T, C, H, W) to (B, T, C, H', W').

        Args:
            x: Tensor (B, T, C, H, W)
            size: (H', W') target size
            mode: interpolation mode ("bilinear" good for RGB)

        Returns:
            Tensor (B, T, C, H', W')
        """
        b, t, c, h, w = x.shape
        # merge batch and time
        x = x.view(b * t, c, h, w)
        x = F.interpolate(x, size=size, mode=mode, align_corners=False)
        # reshape back
        x = x.view(b, t, c, size[0], size[1])
        return x

    
    
    def compute_traj_feat_grad_pass(self, assign_logits, patch_features):
        softmax_logits = F.softmax(assign_logits, dim=-1) # (b, n, m)
        a_out = softmax_logits.transpose(1,2)
        num_token_per_traj = torch.sum(a_out, dim=-1) + 1e-6  # (B,M)
        a_out = a_out / num_token_per_traj.unsqueeze(-1)
        x = torch.matmul(a_out, patch_features) # (B,M,N) (B,N,D) -> (B,M,D)
        return x 
        
    
    def sort_latents_and_query_mask(self, latents, query_mask_tuple):
        latent_mask, patch_mask = query_mask_tuple
        B, L, N = patch_mask.shape

        position_indices = torch.arange(N, device=patch_mask.device).unsqueeze(0).unsqueeze(0).expand(B, L, -1)
        
        # Mask out True positions with a large value so they don't interfere with argmin
        masked_positions = torch.where(patch_mask, N, position_indices)
        
        # Find the first False position (minimum valid position) for each latent
        first_false_pos = torch.min(masked_positions, dim=2)[0]  # (B, L)
        
        # Get sorting indices for each batch
        sort_indices = torch.argsort(first_false_pos, dim=1)  # (B, L)
        
        # Apply permutation to latents
        batch_indices = torch.arange(B, device=latents.device).unsqueeze(1).expand(-1, L)
        latents = latents[batch_indices, sort_indices]  # (B, L, D)
        
        # Apply permutation to patch_mask
        patch_mask = patch_mask[batch_indices, sort_indices]  # (B, L, N)
        
        # Update query_mask_tuple with the permuted masks
        query_mask_tuple = (latent_mask, patch_mask)
        
        return latents, query_mask_tuple
    
    def forward(self, video, gt_seg_prob=None, output_only_valid_token=False, external_patch_features=None, latent_level=None):
        
        if self.use_external_patch_features: 
            assert external_patch_features is not None
            assert external_patch_features.shape[-1] == self.embed_dim
        
        if gt_seg_prob is None:
            bs, T, _, h, w = video.shape
            assert h == w
            if h != self.segmenter_input_res: video = self.resize_video(video, size=(self.segmenter_input_res, self.segmenter_input_res))
        else: 
            bs, T = external_patch_features.shape[0], 16
        
        if self.use_external_patch_features: 
            if gt_seg_prob is not None:  ori_assign_logits = gt_seg_prob.to(external_patch_features.dtype).to(external_patch_features.device).detach()
            else: ori_assign_logits = self.segmenter(video, return_intermediate_feature=False)
            
            patch_features = external_patch_features.reshape(bs,-1,external_patch_features.shape[-1])
            self.latent_res = int(math.sqrt(patch_features.shape[1]//T))
        else: 
            ori_assign_logits, patch_features = self.segmenter(video, return_intermediate_feature=True)

        if self.latent_res != self.segmenter.output_res: 
            print("resizing logits!")
            assign_logits = self.resize_segmentation_logits(ori_assign_logits, num_frames=T)
        else: assign_logits = ori_assign_logits
        
        predicted_label_map = torch.argmax(assign_logits, dim=-1) # (b, n)
        
        
        cur_latent_level = random.randint(0, self.total_latent_level) if latent_level is None else latent_level
        select_num_latent_per_traj = 2 ** cur_latent_level
        
        if self.latents is None: 
            ori_latents = self.compute_traj_feat_grad_pass(assign_logits, external_patch_features).unsqueeze(2)
        else: ori_latents = repeat(self.latents, 'm k d -> b m k d', b = bs)
        
        if self.add_mean_traj: 
            ori_latents = ori_latents + self.compute_traj_feat_grad_pass(assign_logits, patch_features).unsqueeze(2)
            
        latents = rearrange(ori_latents[:, :, :select_num_latent_per_traj], "b m k d -> b (m k) d")
        
        if self.use_pos_branch:
            pos_latents = repeat(self.pos_latents, 'm k d -> b m k d', b = bs)
            pos_latents = rearrange(pos_latents[:, :, :select_num_latent_per_traj], "b m k d -> b (m k) d")
        
        query_mask_tuple = self.make_query_mask(
            predicted_label_map,
            num_traj=self.segmenter.num_traj, 
            num_latent_per_traj=select_num_latent_per_traj if self.config.pre_select_latents else 2**self.total_latent_level,
            attend_other_latents=self.config.attend_other_latents
        )
        if self.rearrange_latent: latents, query_mask_tuple = self.sort_latents_and_query_mask(latents, query_mask_tuple)
        
        try:
            if self.rope_3d: x = self.traj_feature_updater(patch_features, T, self.latent_res, self.latent_res, attention_mask=query_mask_tuple, input_latent=latents)
            else: x = self.traj_feature_updater(patch_features, attention_mask=query_mask_tuple, input_latent=latents)
        except Exception as e:
            # breakpoint()  # stripped for OSS release
            print(e)
        
        if self.use_pos_branch: x += self.pos_feature_updater(None, T, self.latent_res, self.latent_res, attention_mask=query_mask_tuple, input_latent=pos_latents)
        
        if not self.config.pre_select_latents:
            x_sep = rearrange(x, "b (m k) d -> b m k d", m=self.segmenter.num_traj, k=2**self.total_latent_level)                         # (B, M, K, D)
            x_select = x_sep[:, :, :select_num_latent_per_traj] 
            x = rearrange(x_select, "b m k d -> b (m k) d")                        
        
        x = self.output_projector(x)
        
        if gt_seg_prob is not None:
            class_counts = gt_seg_prob.sum(dim=1)           # (B, M)
            attn_mask = class_counts > 0
        else:
            attn_mask = torch.zeros(bs, self.segmenter.num_traj, dtype=torch.bool, device=predicted_label_map.device)
            rows = torch.arange(bs, device=predicted_label_map.device).unsqueeze(1)   # (b, 1)
            attn_mask[rows, predicted_label_map] = True   # -> (b, m) True where a class appears
        attn_mask = attn_mask.repeat_interleave(select_num_latent_per_traj, dim=1)
        
        if output_only_valid_token:
            output_tokens = [x[i][attn_mask[i]] for i in range(bs)]
            return output_tokens, ori_assign_logits
        else:
            return x, attn_mask, ori_assign_logits
    
    def apply_fsdp2(self, **kwargs):
        """
        Apply FSDP to SegmentTokenizer components individually to avoid
        parameter name transformation issues that occur when wrapping as single unit.
        
        This method mirrors the approach used in image_vit.apply_fsdp2() to ensure
        consistent parameter naming for distributed checkpointing.
        """
        from torch.distributed.fsdp import fully_shard
        
        # Apply FSDP to the main segmenter component (SimpleSegmenter)
        # Now using its own apply_fsdp2 method for granular wrapping
        self.segmenter.apply_fsdp2(**kwargs)
        
        # Apply FSDP to trajectory feature updater
        fully_shard(self.traj_feature_updater, **kwargs)
        
        # Apply FSDP to output projector
        fully_shard(self.output_projector, **kwargs)
        
        # Finally, wrap any remaining parameters in self (like latents)
        fully_shard(self, **kwargs)

    def freeze_segmenter(self):
        for p in self.segmenter.parameters():
            p.requires_grad = False
    
    def reset_parameters(self):
        """Reset parameters for SegmentTokenizer and its submodules"""
        # Reset the main segmenter
        # self.segmenter.reset_parameters()
        
        # Reset trajectory feature updater
        self.traj_feature_updater.reset_parameters()
        
        # Reset output projector
        nn.init.xavier_uniform_(self.output_projector.weight)
        if self.output_projector.bias is not None:
            nn.init.zeros_(self.output_projector.bias)
        
        # Reset latents parameter if it exists
        if self.latents is not None:
            # Re-initialize with fourier initialization as done in __init__
            fourier_init = init_latents_ring_vectorized(n=self.segmenter.num_traj, m=2 ** self.total_latent_level, d=self.embed_dim, max_freq=2, )
        
            with torch.no_grad():
                self.latents.copy_(fourier_init)
    
    def reset_with_pretrained_weights(self, ckpt_path):
        """Reset SegmentTokenizer parameters with pretrained weights from checkpoint"""
        # Reset the main segmenter with pretrained weights
        if not self.use_gt_seg: self.segmenter.reset_with_pretrained_weights(ckpt_path)
        
        # Reset other components to their default parameters
        self.traj_feature_updater.reset_parameters()
        
        # Reset output projector
        nn.init.xavier_uniform_(self.output_projector.weight)
        if self.output_projector.bias is not None:
            nn.init.zeros_(self.output_projector.bias)
        
        # Reset latents parameter if it exists
        if self.latents is not None:
            # Re-initialize with fourier initialization as done in __init__
            fourier_init = init_latents_ring_vectorized(n=self.segmenter.num_traj, m=2 ** self.total_latent_level, d=self.embed_dim, max_freq=2, )
        
            with torch.no_grad():
                self.latents.copy_(fourier_init)






            

class SegmentViT(nn.Module):
    def __init__(self,  config=None, backbone_config=None, perceiver_config=None, num_frames=16, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.config = edict(config)
        self.backbone_config = edict(backbone_config)
        self.perceiver_config=edict(perceiver_config)
        
        self.num_frames = num_frames

        self.traj_tokenizer = SegmentTokenizer((self.config, self.backbone_config, self.perceiver_config), total_latent_level=self.config.total_latent_level, add_mean_traj=self.config.add_mean_traj)
        self.vision_encoder, self.vision_layernorm = self.build_vision_encoder()  
        
        self.embed_dim = self.vision_encoder.embed_dim
        self.num_heads = self.vision_encoder.num_heads
        
        if norm_layer is not None:
            self.norm = norm_layer(self.embed_dim)
        else:
            self.norm = nn.Identity()
            

    def freeze_segmenter(self):
        logger.info("calling freeze segmenter")
        for param in self.traj_tokenizer.segmenter.parameters():
            param.requires_grad = False
            
    def unfreeze_segmenter(self):
        logger.info("calling UNfreeze segmenter")
        for param in self.traj_tokenizer.segmenter.parameters():
            param.requires_grad = True
            
    def build_vision_encoder(self):
        vision_encoder = CustomTransformer(
            model_name=self.config.model_name,
            pretrained=self.config.pretrained,
            pool=self.config.pool,
            embed_dim=self.config.embed_dim
        )
        return vision_encoder, None
        
    def forward(self, video, output_attention=False, gt_seg_prob=None, latent_level=None):
        x, attn_mask, ori_assign_logits = self.traj_tokenizer(video, gt_seg_prob=gt_seg_prob, latent_level=latent_level)

        # NOTE: previous code had a stray `return x` here that bypassed the
        # transformer entirely, so trajvitv2 was effectively == simplesegmenter.
        # Restoring the transformer pass so we actually train the trajectory ViT.
        if not output_attention:
            x, _ = self.vision_encoder(x, attn_mask, output_attention=output_attention)
        else:
            x, _, attn_score = self.vision_encoder(x, attn_mask, output_attention=output_attention)

        x = self.norm(x)

        if output_attention: return x, ori_assign_logits, attn_score
        else: return x, ori_assign_logits  # (bs, L, d)
        





def count_parameters(model, trainable_only: bool = True):
    """
    Args
    ----
    model : torch.nn.Module
    trainable_only : bool
        • True  → count only parameters with requires_grad = True  
        • False → count every parameter in the model

    Returns
    -------
    total_params : int
        Number of scalar parameters.
    """
    param_iter = (p for p in model.parameters()
                  if (p.requires_grad or not trainable_only))
    return sum(p.numel() for p in param_iter)





if __name__ == "__main__":
    traj_model = {
        "model_name": "vit-large",
        "embed_dim": 1024,
        "segment_embed_dim": 1024,
        "latent_res": 28,
        "output_res": 28,
        "loss_func": "dice",  # one of [dice, focal, ce], or combinations like dice_focal
        "num_traj": 128,
        "num_latent_per_traj_base_2": 0,
        "num_latent_encoder_use_base_2": -1,  # -1 means random select
        "pretrained": False,
        "pool": "cls",
        "pre_select_latents": False,
        "attend_other_latents": False,
        "rope_3d": False,
        "no_high_res": False,
        "total_latent_level": 0,
        "add_mean_traj": True
    }

    backbone = {
        "backbone_model": "dinov3_small",
        "backbone_pretrained": False,
        "backbone_output_hierarchy": False,
        "backbone_pool": "sum",
        "freeze": False
    }

    perceiver = {
        "depth": 2
    }

    # from calflops import calculate_flops


    model = SegmentViT(config=traj_model, backbone_config=backbone, perceiver_config=perceiver, num_frames=16).cuda()

    trainable = count_parameters(model, trainable_only=False)
    print(trainable)

    trainable = count_parameters(model.traj_tokenizer, trainable_only=False)
    print(trainable)
    
    B, H, W = 1, 224, 224  # Batch size, frames, height, width
    
    
    for T in [16, 32, 64, 128, 256]:
        input_shape = (B, T, 3, H, W)
        
        # print("trajvitv2 parameters (M)", trainable/ 1e6)

        timing_list = []
        for i in range(5):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)

        
            input = torch.zeros(input_shape).cuda()
            torch.cuda.synchronize()          # make sure previous work is done
            start_event.record()              # start GPU timer

            with torch.no_grad(): model(input)


            end_event.record()                # stop GPU timer
            torch.cuda.synchronize()    
            
            gpu_time_ms = start_event.elapsed_time(end_event)
            if i > 0: timing_list.append(gpu_time_ms)
        timing = torch.mean(torch.tensor(timing_list)).item()
        print("T", T, "gpu time", timing)

        # track_flops, macs, params = calculate_flops(
        #     model= model,
        #     input_shape=input_shape,
        #     output_as_string=False,
        #     output_precision=4,
        #     print_results=False,
        #     print_detailed=False
        # )
        # print("trajvitv2 flops", track_flops / 1e9)
            
        