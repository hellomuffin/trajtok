import argparse
import os
import shutil
import logging
import json
import gc
from typing import Dict, Any, Optional

import torch
from transformers import GenerationConfig
from transformers.image_utils import (
    PILImageResampling,
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
)

from olmo.models.molmo2.molmo2 import Molmo2Config as ModelConfig
from olmo.train.checkpointer import load_model_state
from olmo.util import (
    prepare_cli_environment,
    resource_path,
    select_checkpoint
)

from .configuration_molmo2 import Molmo2Config, Molmo2VitConfig, Molmo2AdapterConfig, Molmo2TextConfig
from .modeling_molmo2 import Molmo2ForConditionalGeneration
from .processing_molmo2 import Molmo2Processor
from .image_processing_molmo2 import Molmo2ImageProcessor
from .video_processing_molmo2 import Molmo2VideoProcessor


logger = logging.getLogger(__name__)


CHAT_TEMPLATE = (
    "{% set DEMO_STYLES = ["
    "'point_count','pointing','cosyn_point','user_qa','long_caption','short_caption',"
    "'video_long_caption','video_short_caption','video_point_track_per_frame',"
    "'video_point_track_start_end','video_point_track_all_frames','video_single_point_track_start_end',"
    "'video_transcript','video_clip_caption_start_end','video_clip_caption_start_end_in_seconds',"
    "'video_clip_transcript_start_end','video_clip_transcript_start_end_in_seconds',"
    "'video_frame_caption_timestamp','video_frame_caption_timestamp_in_seconds',"
    "'correction_qa','text_sft','video_point','video_point_count','video_count','video_count_point',"
    "'multi_image_pointing','multi_image_counting','multi_image_point_then_count','multi_image_count_then_point','demo',"
    "'a_okvqa_mc','ai2_diagram_no_letter','ai2_diagram','science_qa',"
    "'multi_image_mc','multi_image_mc_exp','mantis_instruct_mc',"
    "'video_multiple_choice','video_multiple_choice_count_without_pointing',"
    "'video_multiple_choice_multiple_correct','video_multiple_choice_w_subtitle'"
    "] %}"

    "{% set image_count = namespace(value=0) %}"
    "{% set video_count = namespace(value=0) %}"

    "{% set has_subtitle = messages and messages[0]['role'].lower() == 'subtitle' %}"

    "{% for message in messages %}"
        "{% if message['content'] is not string %}"
            "{% for content in message['content'] %}"
                "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
                    "{% set image_count.value = image_count.value + 1 %}"
                "{% elif content['type'] == 'video' or 'video' in content or 'video_url' in content %}"
                    "{% set video_count.value = video_count.value + 1 %}"
                "{% endif %}"
            "{% endfor %}"
        "{% endif %}"
    "{% endfor %}"

    "{% if image_count.value == 1 %}"
        "{{ '<|image|>' }}"
    "{% elif image_count.value > 1 %}"
        "{% for i in range(image_count.value) %}"
            "{{ 'Image ' ~ (i + 1) ~ '<|image|>' }}"
        "{% endfor %}"
    "{% endif %}"

    "{% for _ in range(video_count.value) %}"
        "{{ '<|video|>' }}"
    "{% endfor %}"

    "{% if has_subtitle %}"
        "{{ messages[0]['content'] }}"
    "{% endif %}"

    "{% for message in messages %}"
        "{% set role = message['role'].lower() %}"

        "{% if role == 'subtitle' %}"
            "{% continue %}"
        "{% endif %}"

        "{% set conv_index = loop.index - (1 if has_subtitle else 0) %}"

        "{%- if (conv_index % 2 == 1 and role != 'user') "
        "or (conv_index % 2 == 0 and role != 'assistant') -%}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
        "{%- endif -%}"

        "{% if message['content'] is string %}"
            "{% set text_content = message['content'] %}"
        "{% else %}"
            "{% set m = namespace(text='') %}"
            "{% for content in message['content'] %}"
                "{% if content['type'] == 'text' %}"
                    "{% if content['style'] is defined and content['style'] not in DEMO_STYLES %}"
                        "{% set seg = content['style'] ~ ': ' ~ content['text'] %}"
                    "{% else %}"
                        "{% set seg = content['text'] %}"
                    "{% endif %}"
                    "{% set m.text = m.text ~ ('' if not m.text else ' ') ~ seg %}"
                "{% endif %}"
            "{% endfor %}"
            "{% set text_content = m.text %}"
        "{% endif %}"

        "{% if role == 'user' %}"
            "{% if not (has_subtitle and loop.index == 2) and not (not has_subtitle and loop.first) %}{{ '<|im_end|>\\n' }}{% endif %}"
            "{{ '<|im_start|>user\\n' }}"
            "{{ text_content }}"
            "{{ '<|im_end|>\\n' }}"
        "{% else %} {# assistant #}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{{ text_content }}"
        "{% endif %}"
    "{% endfor %}"

    "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)


def convert_config(
    model_config: ModelConfig,
    attn_implementation: str,
    override_max_model_len: Optional[int],
) -> Molmo2Config:
    """Convert config to HF-compatible config"""
    vision_backbone_cfg = model_config.vision_backbone
    vit_config = vision_backbone_cfg.vit
    llm_config = model_config.llm

    molmo2_vit_config = Molmo2VitConfig(
        hidden_size=vit_config.image_emb_dim,
        intermediate_size=vit_config.image_mlp_dim,
        num_hidden_layers=vit_config.image_num_layers,
        num_attention_heads=vit_config.image_num_heads,
        num_key_value_heads=vit_config.image_num_key_value_heads,
        head_dim=vit_config.image_head_dim,
        hidden_act=vit_config.image_mlp_activations,
        layer_norm_eps=vit_config.image_norm_eps,
        image_default_input_size=vit_config.image_default_input_size,
        image_patch_size=vit_config.image_patch_size,
        image_num_pos=vit_config.image_num_pos,
        attention_dropout=0.0,
        residual_dropout=0.0,
        initializer_range=vit_config.initializer_range,
        float32_attention=vit_config.float32_attention,
        attn_implementation=attn_implementation,
    )
    adapter_hidden_act = "silu" if llm_config.activation_type == "swiglu" else llm_config.activation_type
    adapter_intermediate_size = (
        llm_config.mlp_hidden_size if llm_config.mlp_hidden_size is not None
        else llm_config.mlp_ratio * llm_config.d_model
    ) // 2
    molmo2_adapter_config = Molmo2AdapterConfig(
        vit_layers=vision_backbone_cfg.vit_layers,
        pooling_attention_mask=vision_backbone_cfg.pooling_attention_mask,
        hidden_size=vit_config.image_emb_dim,
        num_attention_heads=vit_config.image_num_heads,
        num_key_value_heads=vit_config.image_num_key_value_heads,
        head_dim=vit_config.image_head_dim,
        float32_attention=vit_config.float32_attention,
        attention_dropout=0.0,
        residual_dropout=0.0,
        hidden_act=adapter_hidden_act,
        intermediate_size=adapter_intermediate_size,
        text_hidden_size=llm_config.d_model,
        image_feature_dropout=vision_backbone_cfg.image_feature_dropout,
        initializer_range=llm_config.initializer_range,
        attn_implementation=attn_implementation,
    )
    llm_head_dim = llm_config.d_model // llm_config.n_heads if llm_config.head_dim is None else llm_config.head_dim
    llm_intermediate_size = (
        llm_config.mlp_hidden_size if llm_config.mlp_hidden_size is not None
        else llm_config.mlp_ratio * llm_config.d_model
    ) // 2
    llm_hidden_act = "silu" if llm_config.activation_type == "swiglu" else llm_config.activation_type
    rope_scaling: Optional[Dict[str, Any]] = None
    if llm_config.rope_type != "default":
        rope_scaling = dict(rope_type=llm_config.rope_type)
        for key in [
            "rope_factor",
            "rope_high_freq_factor",
            "rope_low_freq_factor",
            "rope_attention_factor",
            "rope_original_max_position_embeddings",
            "rope_beta_fast",
            "rope_beta_slow",
            "rope_mscale",
            "rope_mscale_all_dim",
            "rope_truncate",
        ]:
            if getattr(llm_config, key) is not None:
                rope_scaling[key[len("rope_"):]] = getattr(llm_config, key)
    
    max_position_embeddings = llm_config.max_position_embeddings or llm_config.max_sequence_length
    if override_max_model_len is not None:
        max_position_embeddings = override_max_model_len
    rope_scaling_layers: list[int] | None = None
    if llm_config.full_attention_layers is not None:
        # HACK: The original Olmo3 applies scaling to full attention layers,
        # while we applies scaling to slinding attention layers.
        if llm_config.sliding_attention_rope_scaling:
            rope_scaling_layers = [idx for idx in range(llm_config.n_layers) if idx not in llm_config.full_attention_layers]
        else:
            rope_scaling_layers = list(llm_config.full_attention_layers)
    molmo2_text_config = Molmo2TextConfig(
        hidden_size=llm_config.d_model,
        num_attention_heads=llm_config.n_heads,
        num_key_value_heads=llm_config.effective_n_kv_heads,
        head_dim=llm_head_dim,
        vocab_size=llm_config.embedding_size or llm_config.vocab_size,
        additional_vocab_size=llm_config.additional_vocab_size,
        qkv_bias=llm_config.qkv_bias,
        num_hidden_layers=llm_config.n_layers,
        intermediate_size=llm_intermediate_size,
        hidden_act=llm_hidden_act,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        max_position_embeddings=max_position_embeddings,
        rope_theta=llm_config.rope_theta,
        rope_scaling=rope_scaling,
        rope_scaling_layers=rope_scaling_layers,
        use_qk_norm=llm_config.attention_layer_norm,
        qk_norm_type=llm_config.attention_layer_norm_type,
        layer_norm_eps=llm_config.layer_norm_eps,
        norm_after=llm_config.norm_after,
        initializer_range=llm_config.initializer_range,
        attn_implementation=attn_implementation,
    )

    tokenizer = model_config.build_tokenizer()
    image_start_token_id = tokenizer.image_start_token_id
    image_end_token_id = tokenizer.image_end_token_id
    low_res_image_start_token_id = tokenizer.low_res_image_start_token_id
    image_low_res_id = tokenizer.image_low_res_token_id
    image_patch_id = tokenizer.image_patch_token_id
    image_col_id = tokenizer.image_col_token_id
    frame_start_token_id = tokenizer.frame_start_token_id
    frame_end_token_id = tokenizer.frame_end_token_id

    use_frame_special_tokens = getattr(model_config.mm_preprocessor, "use_frame_special_tokens", False)

    molmo2_config = Molmo2Config(
        vit_config=molmo2_vit_config,
        adapter_config=molmo2_adapter_config,
        text_config=molmo2_text_config,
        image_start_token_id=image_start_token_id,
        low_res_image_start_token_id=low_res_image_start_token_id,
        image_end_token_id=image_end_token_id,
        image_low_res_id=image_low_res_id,
        image_patch_id=image_patch_id,
        image_col_id=image_col_id,
        frame_start_token_id=frame_start_token_id,
        frame_end_token_id=frame_end_token_id,
        use_frame_special_tokens=use_frame_special_tokens,
        initializer_range=llm_config.initializer_range,
        use_cache=True,
        tie_word_embeddings=False,  # Always false for Molmo2
    )
    return molmo2_config


def convert_lm_head_and_prefix(
    state_dict: dict[str, Any],
    base_model_prefix: str,
    weight_tying: bool
) -> dict[str, Any]:
    new_state_dict = {}
    for key, val in state_dict.items():
        if key == "transformer.ff_out.weight":
            new_key = "lm_head.weight"
        else:
            new_key = f"{base_model_prefix}.{key}"
        new_state_dict[new_key] = val
    
    if weight_tying:
        new_state_dict["lm_head.weight"] = state_dict["transformer.wte.embedding"]
    
    return new_state_dict


def convert_molmo2(
    state_dict: dict[str, Any],
    config: Molmo2Config,
    weight_tying: bool,
) -> dict[str, Any]:
    base_model_prefix = Molmo2ForConditionalGeneration.base_model_prefix
    new_state_dict = convert_lm_head_and_prefix(state_dict, base_model_prefix, weight_tying)
    model_prefix = f"{base_model_prefix}.transformer"
    qkv_bias = config.qkv_bias if isinstance(config, Molmo2TextConfig) else config.text_config.qkv_bias
    use_qk_norm = config.use_qk_norm if isinstance(config, Molmo2TextConfig) else config.text_config.use_qk_norm
    for layer_i in range(config.num_hidden_layers):
        prefix = f"{model_prefix}.blocks.{layer_i}"

        move_to_attn = ["att_proj.weight", "attn_out.weight"]
        if qkv_bias:
            move_to_attn.append("att_proj.bias")
        if use_qk_norm:
            move_to_attn += ["q_norm.weight", "k_norm.weight"]
        
        for k in move_to_attn:
            assert f"{prefix}.self_attn.{k}" not in new_state_dict
            new_state_dict[f"{prefix}.self_attn.{k}"] = new_state_dict.pop(f"{prefix}.{k}")
        
        move_to_mlp = ["ff_proj.weight", "ff_out.weight"]
        for k in move_to_mlp:
            assert f"{prefix}.mlp.{k}" not in new_state_dict
            new_state_dict[f"{prefix}.mlp.{k}"] = new_state_dict.pop(f"{prefix}.{k}")
    
    return new_state_dict


def convert_model(
    checkpoint_dir: str,
    model_config: ModelConfig,
    hf_config: Molmo2Config,
    use_bfloat16: bool,
) -> Molmo2ForConditionalGeneration:
    """Convert model to HF-compatible model"""
    with torch.device("meta"):
        model = model_config.build_model()
        hf_model = Molmo2ForConditionalGeneration(hf_config)
    model.to_empty(device=torch.device("cpu"))
    hf_model.to_empty(device=torch.device("cpu"))

    load_model_state(checkpoint_dir, model)
    model.eval()
    model = model.to(torch.float32)
    state_dict = model.state_dict()

    new_state_dict = convert_molmo2(state_dict, hf_config, model_config.llm.weight_tying)
    hf_model.eval()
    hf_model = hf_model.to(torch.bfloat16 if use_bfloat16 else torch.float32)
    hf_model.load_state_dict(new_state_dict)
    return hf_model


def save(
    checkpoint_dir: str,
    output_dir: str,
    use_bfloat16: bool,
    attn_implementation: str,
    override_max_model_len: Optional[int],
) -> None:
    logger.info(f"Loading model config from {checkpoint_dir}")
    config_path = resource_path(select_checkpoint(checkpoint_dir), "config.yaml")
    model_config: ModelConfig = ModelConfig.load(config_path, key="model", validate_paths=False)

    hf_config = convert_config(model_config, attn_implementation, override_max_model_len)

    logger.info(f"Save HF-compatible model config and checkpoint to {output_dir}")
    logger.info(f"Save HF-compatible model config and checkpoint to {output_dir}")
    hf_model = convert_model(checkpoint_dir, model_config, hf_config, use_bfloat16)

    hf_model.save_pretrained(output_dir)

    gc.collect()

    model_file = os.path.join(output_dir, "modeling_molmo2.py")
    if not os.path.exists(model_file):
        logger.warning(f"Copying model file to {model_file} manually")
        shutil.copyfile(
            "olmo/hf_model/modeling_molmo2.py",
            model_file,
        )

    with open(os.path.join(output_dir, "config.json")) as f:
        config = json.load(f)

    auto_map = config.get("auto_map", None)
    if auto_map is None:
        auto_map = {}
    if "AutoModelForImageTextToText" not in auto_map:
        logger.warning("Add AutoModelForImageTextToText to auto_map")
        auto_map["AutoModelForImageTextToText"] = "modeling_molmo2.Molmo2ForConditionalGeneration"
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    tokenizer = model_config.build_tokenizer().tokenizer
    if not tokenizer.bos_token:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    tokenizer.chat_template = CHAT_TEMPLATE

    logger.info(f"Save tokenizer and processor to {output_dir}")

    mm_cfg = model_config.mm_preprocessor
    vit_cfg = model_config.vision_backbone.vit

    img_cfg = mm_cfg.image
    video_cfg = mm_cfg.video

    assert vit_cfg.resize_mode == "siglip", "Only siglip resize is supported for now"
    assert vit_cfg.normalize == "siglip", "Only siglip normalization is supported for now"
    assert img_cfg.crop_mode == "overlap-and-resize-c2", "Only overlap-and-resize-c2 crop mode is supported for now"
    assert img_cfg.max_crops == img_cfg.max_multi_image_crops, "max_crops and max_multi_image_crops must be the same"
    assert img_cfg.pooling_w == img_cfg.multi_image_pooling_w, "pooling_w and multi_image_pooling_w must be the same"
    assert img_cfg.pooling_h == img_cfg.multi_image_pooling_h, "pooling_h and multi_image_pooling_h must be the same"

    image_processor = Molmo2ImageProcessor(
        size={"height": vit_cfg.image_default_input_size[0], "width": vit_cfg.image_default_input_size[1]},
        resample=PILImageResampling.BILINEAR,
        image_mean=IMAGENET_STANDARD_MEAN,
        image_std=IMAGENET_STANDARD_STD,
        do_convert_rgb=True,
        max_crops=img_cfg.max_crops,
        overlap_margins=img_cfg.overlap_margins,
        patch_size=vit_cfg.image_patch_size,
        pooling_size=[img_cfg.pooling_h, img_cfg.pooling_w],
    )

    image_use_col_tokens = img_cfg.use_col_tokens
    use_single_crop_col_tokens = img_cfg.use_single_crop_col_tokens
    use_single_crop_start_token = img_cfg.use_single_crop_start_token

    assert vit_cfg.resize_mode == "siglip", "Only siglip resize is supported for now"
    assert vit_cfg.normalize == "siglip", "Only siglip normalization is supported for now"
    assert video_cfg.time_mode == "per-frame-compact", "Only per-frame-compact time mode is supported for now"

    max_fps = video_cfg.max_fps
    if isinstance(max_fps, (tuple, list)):
        assert len(max_fps) == 1, "Only one max_fps is supported for now"
        max_fps = max_fps[0]
    video_processor = Molmo2VideoProcessor(
        size={"height": vit_cfg.image_default_input_size[0], "width": vit_cfg.image_default_input_size[1]},
        resample=PILImageResampling.BILINEAR,
        image_mean=IMAGENET_STANDARD_MEAN,
        image_std=IMAGENET_STANDARD_STD,
        do_convert_rgb=True,
        patch_size=vit_cfg.image_patch_size,
        pooling_size=[video_cfg.pooling_h, video_cfg.pooling_w],
        frame_sample_mode=video_cfg.frame_sample_mode,
        num_frames=video_cfg.max_frames,
        max_fps=max_fps,
        sampling_fps=2,
    )

    video_use_col_tokens = False
    use_frame_special_tokens = video_cfg.use_frame_special_tokens

    processor = Molmo2Processor(
        image_processor,
        video_processor,
        tokenizer,
        chat_template=CHAT_TEMPLATE,
        image_use_col_tokens=image_use_col_tokens,
        use_single_crop_col_tokens=use_single_crop_col_tokens,
        use_single_crop_start_token=use_single_crop_start_token,
        video_use_col_tokens=video_use_col_tokens,
        use_frame_special_tokens=use_frame_special_tokens,
    )
    processor.audio_tokenizer = None
    processor.save_pretrained(output_dir)

    logger.info(f"Save generation config to {output_dir}")
    generation_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generation_config.save_pretrained(output_dir)

    del hf_model, processor, tokenizer, generation_config
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Convert Molmo checkpoint to HuggingFace format."
    )
    parser.add_argument("checkpoint_dir", help="Location of Molmo2 checkpoint.")
    parser.add_argument("output_dir", help="Location to save the converted checkpoint.", default="./hf-ckpt")
    parser.add_argument("--use_bfloat16", action="store_true", help="Use bfloat16 weights")
    parser.add_argument(
        "--attn_implementation", type=str, default="sdpa", help="Attention type",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument(
        "--override_max_model_len",
        type=int,
        default=None,
        help="Override the max model length",
    )
    args = parser.parse_args()
    prepare_cli_environment()

    save(
        args.checkpoint_dir,
        args.output_dir,
        args.use_bfloat16,
        args.attn_implementation,
        args.override_max_model_len,
    )


if __name__ == "__main__":
    main()