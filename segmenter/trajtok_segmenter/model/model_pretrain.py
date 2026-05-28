"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see https://opensource.org/licenses/BSD-3-Clause
"""
import torch
from einops import rearrange, repeat
import logging
import torch.nn.functional as F
from typing import Tuple


from trajtok_segmenter.model.traj_transformer import VideoTokenViT
# NOTE: `share_models.vit3d.ViT3D` is omitted from this release. The
# `VideoViT` class below was an alternative 3D-ViT baseline kept for
# reproducibility of the paper's ablations; it is not exercised by the
# segmenter (`SegmentWrapper`) or the trajectory-CLIP (`SegmentCLIP`)
# training paths. If you call `VideoViT.build_vision_encoder` you will hit
# the NotImplementedError below; copy `share_models/vit3d.py` from the
# closed-source codebase to re-enable.
ViT3D = None  # placeholder; see note above


from trajtok_segmenter.model.model_retrieval_base import SingularityRetrievalBase
from trajtok_segmenter.model.checkpoint_utils import decompose_masks
from trajtok_segmenter.model.segmenter import SimpleSegmenter, SegmentViT

logger = logging.getLogger(__name__)


class Singularity(SingularityRetrievalBase):
    def __init__(self, config=None, tokenizer=None):
        super(Singularity, self).__init__(
            config=config, tokenizer=tokenizer, pretrain=False)
        self.mlm_prob = config.mlm_prob

    def get_mlm_loss(self, text, image_embeds, image_atts):
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        input_ids, labels = self.mask(
            input_ids, self.text_encoder.config.vocab_size, input_ids.device,
            targets=labels, probability_matrix=probability_matrix
        )        

        intermediate_mlm_output = self.text_encoder.bert(
            input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode="text"
        )

        text_embeds = intermediate_mlm_output.last_hidden_state

        mlm_output = self.text_encoder(
            encoder_embeds=text_embeds,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            labels=labels,
            soft_labels=None,
            mode="fusion"
        )
        return mlm_output.loss

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            # We only compute loss on masked tokens
            targets[~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
        
        
    

class VideoViT(SingularityRetrievalBase):
    def __init__(self, config=None, tokenizer=None):
        super(VideoViT, self).__init__(
            config=config, tokenizer=tokenizer, pretrain=False)
        
    def encode_image(self, vision_input, output_attention=False):
        if type(vision_input) == tuple: image = vision_input[0]
        else: image = vision_input
        
        output = self.vision_encoder(image, output_attention=output_attention,)
        
        if output_attention: image_embeds, attn_score = output
        else: image_embeds = output
        
        pooled_image_embeds = image_embeds[:,:1]
        
        if output_attention: return image_embeds, pooled_image_embeds, attn_score
        else: return image_embeds, pooled_image_embeds  # (bsz, #frm*L, d), (bsz, #frm, d)
    
    def build_vision_encoder(self):
        if ViT3D is None:
            raise NotImplementedError(
                "VideoViT requires ViT3D which is not bundled in this release. "
                "It is an ablation baseline not used by SegmentWrapper / SegmentCLIP."
            )
        logger.info("building ViT3D model")
        vision_encoder = ViT3D(
            img_size=self.config.image_res,
            patch_size=16,
            tubelet_size=2,
            num_frames=16,
            model_name=self.config.traj_model.model_name,
            pretrained=self.config.traj_model.pretrained,
            norm_layer=None,
            pool=self.config.traj_model.pool,
        )
        self.vision_width =  vision_encoder.embed_dim
        return vision_encoder, None
    
    def load_image_model(self, state_dict, load_only_vision=False):
        
        vision_state_dict = {k[len('vision_encoder.'):]:v for k, v in state_dict.items() if k.startswith('vision_encoder.')}
        msg = self.vision_encoder.load_image_model(vision_state_dict)
        if not load_only_vision:
            self.text_encoder.load_state_dict({k[len('text_encoder.'):]:v for k, v in state_dict.items() if k.startswith('text_encoder.')})
            self.vision_proj.load_state_dict({k[len('vision_proj.'):]:v for k, v in state_dict.items() if k.startswith('vision_proj.')})
            self.text_proj.load_state_dict({k[len('text_proj.'):]:v for k, v in state_dict.items() if k.startswith('text_proj.')})
            with torch.no_grad():  self.temp.copy_(state_dict["temp"])  # Copy the value
        return msg



    
class VideoTokCLIP(SingularityRetrievalBase):
    def __init__(self, config=None, tokenizer=None):
        super(VideoTokCLIP, self).__init__(
            config=config, tokenizer=tokenizer, pretrain=False)
        self.vision_proj = torch.nn.Linear(self.vision_width, self.embed_dim)

        
    def build_vision_encoder(self):
        logger.info("building ViTToken model")
        vision_encoder = VideoTokenViT(
            config=self.config.traj_model,
            pos_config=self.config.traj_pos,
            perceiver_config=self.config.perceiver,
            num_frames=16,  # fix: temporarily hard coded
            norm_layer=None,
        )
        self.vision_width = vision_encoder.embed_dim
        return vision_encoder, None

    
    def encode_image(self, vision_input, output_attention=False):
        if len(vision_input) == 4:
            video, mask, graph, num_tokens = vision_input
        else:
            video, mask, graph = vision_input
        
        demasks = decompose_masks(mask)
        
        output = self.vision_encoder(
            video, 
            segmask=demasks, 
            video_graph=graph, 
            output_attention=output_attention,
        )
        if output_attention: image_embeds, attn_score = output
        else: image_embeds = output
        
        pooled_image_embeds = image_embeds[:,:1]
        
        if output_attention: return image_embeds, pooled_image_embeds, attn_score
        else: return image_embeds, pooled_image_embeds  # (bsz, #frm*L, d), (bsz, #frm, d)

        
        
    def load_image_model(self, state_dict, load_only_vision=False):
        if load_only_vision:
            logger.info("loading vision module only")
            vision_state_dict = {k[len('vision_encoder.'):]:v for k, v in state_dict.items() if k.startswith('vision_encoder.')}
            msg = self.vision_encoder.load_state_dict(vision_state_dict)
        else:
            msg = self.load_state_dict(state_dict, strict=False)
        return msg
        
        
        
        

    
class SegmentWrapper(VideoTokCLIP):
    def build_vision_encoder(self):
        logger.info("building segmenter model")
        self.config.traj_model.output_res = self.config.image_res // self.config.mask_down_factor
        logger.info(f"output res: {self.config.traj_model.output_res}")
        
        vision_encoder = SimpleSegmenter(
            config=self.config.traj_model,
            backbone_config=self.config.backbone,
            perceiver_config=self.config.perceiver, 
            high_res=not self.config.traj_model.no_high_res
        )
        
        self.vision_width = self.config.traj_model.embed_dim
        return vision_encoder, None


    def forward(self, vision_input, text, idx, training_mask=None):
        
        if len(vision_input) == 4:
            video, mask, graph, num_tokens = vision_input
        else:
            video, mask, graph = vision_input
        # graph: (B,M,T)
        if graph.shape[1] > self.config.traj_model.num_traj: graph = graph[:, :self.config.traj_model.num_traj]
        
        # print("input image shape:", image.shape) # (B, T, 3, 224, 224)
        assignment_logits = self.vision_encoder(video, output_high_res=not self.config.traj_model.no_high_res)

       
        if self.config.traj_model.no_high_res:
            low_res_pixel, low_res_class = self.vision_encoder.compute_assignment_loss(assignment_logits, mask, graph, ignore_index=-1, low_res_variant = True)
            high_res_class, high_res_pixel = torch.zeros_like(low_res_class), torch.zeros_like(low_res_pixel)
        else:
            low_res_pixel, low_res_class = self.vision_encoder.compute_assignment_loss(assignment_logits[0], mask, graph, ignore_index=-1, low_res_variant = True)
            high_res_pixel, high_res_class = self.vision_encoder.compute_assignment_loss(assignment_logits[1], mask, graph, ignore_index=-1, low_res_variant = False)
        
        return_dict = dict(
            loss_low_res_pixel = low_res_pixel  * self.config.loss_weight.icl,
            loss_low_res_class =  2 * low_res_class  * self.config.loss_weight.icl,
            loss_high_res_pixel = high_res_pixel  * self.config.loss_weight.icl,
            loss_high_res_class = 2 * high_res_class  * self.config.loss_weight.icl,
            loss_ita = torch.zeros_like(high_res_class),
        )
        return return_dict
    
    
    
    
class SegmentCLIP(VideoTokCLIP):
    def build_vision_encoder(self):
        logger.info("building segmenter model")
        self.config.traj_model.output_res = self.config.traj_model.latent_res
        logger.info(f"output res: {self.config.traj_model.output_res}")
        
        vision_encoder = SegmentViT(
            config=self.config.traj_model,
            backbone_config=self.config.backbone,
            perceiver_config=self.config.perceiver, 
        )
        self.use_gt_mask = self.config.use_gt_mask
        self.vision_width = self.config.traj_model.embed_dim
        return vision_encoder, None
    
    
    
    def encode_image(self, vision_input, output_assignment=False, latent_level=None):
        if len(vision_input) == 4:
            video, mask, graph, num_tokens = vision_input
        else:
            video, mask, graph = vision_input
            
        # graph: (B,M,T)
        if graph.shape[1] > self.config.traj_model.num_traj: graph = graph[:, :self.config.traj_model.num_traj]
        
        # print("input image shape:", image.shape) # (B, T, 3, 224, 224)
        if self.use_gt_mask:
            labels, valid = self.vision_encoder.traj_tokenizer.segmenter.pixel_labels(mask, graph, ignore_index=-1)  # (B,N), (B,M)
            labels = labels + 1 # 0 become the dummy class
            gt_seg_prob  = torch.nn.functional.one_hot(labels, self.vision_encoder.traj_tokenizer.segmenter.num_traj+1)           # (B, N, M)
            gt_seg_prob = gt_seg_prob[..., 1:]
        else:
            gt_seg_prob = None
        
        image_embeds, assignment_logits = self.vision_encoder(video, gt_seg_prob=gt_seg_prob, latent_level=latent_level)
        pooled_image_embeds = image_embeds[:,:1]
        
        
        if output_assignment: return image_embeds, pooled_image_embeds, assignment_logits
        else: return image_embeds, pooled_image_embeds
        
        
    def forward(self, vision_input, text, idx, training_mask=None):
        if len(vision_input) == 4:
            video, mask, graph, num_tokens = vision_input
        else:
            video, mask, graph = vision_input
            
        # graph: (B,M,T)
        if graph.shape[1] > self.config.traj_model.num_traj: graph = graph[:, :self.config.traj_model.num_traj]
        
        image_embeds, pooled_image_embeds, assignment = self.encode_image(vision_input, output_assignment=True)
        
        low_res_pixel, low_res_class = self.vision_encoder.traj_tokenizer.segmenter.compute_assignment_loss(assignment, mask, graph, ignore_index=-1, low_res_variant = True)
        
        text_embeds, pooled_text_embeds = self.encode_text(text)

        loss_ita, sim_i2t, sim_t2i, accuracy_ita = self.get_contrastive_loss(
            pooled_image_embeds, pooled_text_embeds, idx)
            
        return_dict = dict(
            loss_ita=loss_ita * self.config.loss_weight.itc,
            loss_low_res_pixel = 0.5 * low_res_pixel  * self.config.loss_weight.icl,
            loss_low_res_class = low_res_class  * self.config.loss_weight.icl,
            loss_high_res_pixel = torch.zeros_like(loss_ita),
            loss_high_res_class = torch.zeros_like(loss_ita),
            accuracy_ita = accuracy_ita
        )
        return return_dict
    

    def load_segmenter_model(self, state_dict):
        rename_state_dict = {k[len('vision_encoder.'):]:v for k, v in state_dict.items() if k.startswith('vision_encoder.')}
        msg = self.vision_encoder.traj_tokenizer.segmenter.load_state_dict(rename_state_dict, strict=False)
        return msg

    def load_image_model(self, state_dict, load_only_vision=False):
        # Warm-start: the `image_pretrained_path` checkpoint is a SimpleSegmenter
        # (SegmentWrapper) ckpt — we only restore the SimpleSegmenter portion that
        # lives at `vision_encoder.traj_tokenizer.segmenter` here. The trajectory
        # ViT, text encoder, and projections stay random.
        return self.load_segmenter_model(state_dict)
    
    
    
