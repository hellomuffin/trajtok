import logging
from typing import Dict
from dataclasses import replace

from olmo.models.molmo2.molmo2_preprocessor import Molmo2PreprocessorConfig
from olmo.preprocessing.data_formatter import DataFormatter
from olmo.models.molmo.molmo_preprocessor import MolmoPreprocessorConfig
from olmo.models.molmo2.molmo2 import Molmo2Config
from olmo.nn.image_vit import VitConfig
from olmo.nn.llm import LlmConfig, AttentionType, LayerNormType, AttentionLayerNormType, RopeType
from olmo.models.molmo.molmo import MolmoConfig
from olmo.preprocessing.video_preprocessor import VideoPreprocessorConfig
from olmo.tokenizer import TokenizerConfig
from olmo.nn.vision_backbone import MolmoVisionBackboneConfig

log = logging.getLogger(__name__)


DEBUG_LLM = LlmConfig(
    d_model=128,
    n_heads=2,
    n_layers=3,
    max_sequence_length=4096,
    additional_vocab_size=128,
    vocab_size=152064,
    rope=True,
    embedding_size=None,
    weight_tying=False,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2-7B",
    )
)

DEBUG_VIT = VitConfig(
    image_num_layers=1,
    image_model_type="siglip",
    image_default_input_size=(378, 378),
    image_emb_dim=1152,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_head_dim=72,
    image_mlp_dim=4304,
    image_mlp_activations="gelu_pytorch_tanh",
    image_num_pos=729,  # no CLS token
    resize_mode="siglip",
)


DEBUG_MOLMO = MolmoConfig(
    llm=DEBUG_LLM,
    vision_backbone=MolmoVisionBackboneConfig(
        vit=DEBUG_VIT
    ),
    data_formatter=DataFormatter(),
    mm_preprocessor=MolmoPreprocessorConfig(crop_mode="resize", max_crops=1)
)


DEBUG_MOLMO2 = Molmo2Config(
    llm=DEBUG_LLM,
    vision_backbone=MolmoVisionBackboneConfig(
        vit=DEBUG_VIT
    ),
    data_formatter=DEBUG_MOLMO.data_formatter,
    mm_preprocessor=Molmo2PreprocessorConfig(
        video=VideoPreprocessorConfig(
            pooling_h=3,
            pooling_w=3,
            max_frames=4,
        ),
    )
)


OPENAICLIP_VISION_BACKBONE = VitConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/vit-l-14-336.pt",
    image_model_type="openai",
    image_default_input_size=(336, 336),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1024,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=23,
    image_head_dim=64,
    image_mlp_dim=4096,
    image_mlp_activations="quick_gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-5,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
)


SIGLIP_VISION_BACKBONE = VitConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/siglip-so400m-14-384.pt",
    image_model_type="siglip",
    image_default_input_size=(378, 378),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1152,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=27,
    image_head_dim=72,
    image_mlp_dim=4304,
    image_mlp_activations="gelu_pytorch_tanh",
    image_dropout_rate=0.0,
    image_num_pos=729, # no CLS token
    image_norm_eps=1e-6,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="siglip",
    normalize="siglip"
)


SIGLIP2_VISION_BACKBONE = replace(
    SIGLIP_VISION_BACKBONE,
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/siglip2-so400m-14-384.pt",
)


DINOV2_LARGE_336_VISION_BACKBONE = VitConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/dinov2-large-336.pt",
    image_model_type="dino",
    image_default_input_size=(336, 336),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1024,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=24,
    image_head_dim=64,
    image_mlp_dim=4096,
    image_mlp_activations="gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-6,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="dino",
    normalize="dino",
)


METACLIP_L14_336_VISION_BACKBONE = VitConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/metaclip-l14-336.pt",
    image_model_type="openai",
    image_default_input_size=(336, 336),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1024,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=24,
    image_head_dim=64,
    image_mlp_dim=4096,
    image_mlp_activations="quick_gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-5,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="metaclip",
)


METACLIP_B16_224_VISION_BACKBONE = VitConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/metaclip-b16-224.pt",
    image_model_type="openai",
    image_default_input_size=(224, 224),
    image_patch_size=16,
    image_pos_patch_size=16,
    image_emb_dim=768,
    image_num_heads=12,
    image_num_key_value_heads=12,
    image_num_layers=12,
    image_head_dim=64,
    image_mlp_dim=3072,
    image_mlp_activations="quick_gelu",
    image_dropout_rate=0.0,
    image_num_pos=197,
    image_norm_eps=1e-5,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="metaclip",
)


METACLIP_400M_B16_224_VISION_BACKBONE = replace(
    METACLIP_B16_224_VISION_BACKBONE,
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/metaclip-400m-b16-224.pt",
)


OLMO3_7B_INSTRUCT = LlmConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/olmo3-7b-instruct.pt",
    d_model=4096,
    n_heads=32,
    n_kv_heads=None,
    clip_qkv=None,
    n_layers=32,
    mlp_ratio=4,
    mlp_hidden_size=22016,
    activation_type="swiglu",
    block_type="sequential",
    rope=True,
    rope_full_precision=True,
    rope_theta=500000,
    rope_type=RopeType.yarn,
    rope_factor=8.0,
    rope_attention_factor=1.2079441541679836,
    rope_beta_fast=32,
    rope_beta_slow=1,
    rope_original_max_position_embeddings=8192,
    full_attention_layers=(3, 7, 11, 15, 19, 23, 27, 31),
    attention_dropout=0.0,
    attention_layer_norm=True,
    layer_norm_type="rms",
    layer_norm_with_affine=True,
    layer_norm_eps=1.0e-06,
    attention_layer_norm_with_affine=True,
    max_sequence_length=4096,
    include_bias=False,
    bias_for_layer_norm=False,
    scale_logits=False,
    vocab_size=100278,
    embedding_size=100278,
    additional_vocab_size=128,
    weight_tying=False,
    attention_type=AttentionType.sdpa,
    norm_after=True,
    tokenizer=TokenizerConfig(
        identifier="allenai/Olmo-3-7B-Instruct",
    ),
    embedding_dropout=0,
    fix_pad_tokenizer=True,
)


QWEN3_4B = LlmConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen3-4b.pt",
    vocab_size=151936,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    attention_layer_norm=True,
    attention_layer_norm_type=AttentionLayerNormType.qwen3,
    rope=True,
    qkv_bias=False,
    weight_tying=True,
    include_bias=False,
    embedding_size=151936,
    d_model=2560,
    mlp_hidden_size=9728*2,
    n_layers=36,
    additional_vocab_size=128,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen3-4B",
    ),
)


QWEN3_4B_INSTRUCT = LlmConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen3-4b-instruct.pt",
    vocab_size=151936,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    attention_layer_norm=True,
    attention_layer_norm_type=AttentionLayerNormType.qwen3,
    rope=True,
    qkv_bias=False,
    weight_tying=True,
    include_bias=False,
    embedding_size=151936,
    d_model=2560,
    mlp_hidden_size=9728*2,
    n_layers=36,
    additional_vocab_size=128,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    rope_theta=5000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen3-4B-Instruct-2507",
    ),
)


QWEN3_8B_BASE = LlmConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen3-8b-base.pt",
    vocab_size=151936,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    attention_layer_norm=True,
    attention_layer_norm_type=AttentionLayerNormType.qwen3,
    rope=True,
    qkv_bias=False,
    weight_tying=False,
    include_bias=False,
    embedding_size=151936,
    d_model=4096,
    mlp_hidden_size=12288*2,
    n_layers=36,
    additional_vocab_size=128,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen3-8B-Base",
    ),
)


QWEN3_8B = replace(
    QWEN3_8B_BASE,
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen3-8b.pt",
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen3-8B",
    ),
)


VISION_BACKBONES: Dict[str, VitConfig] = {
    "openai": OPENAICLIP_VISION_BACKBONE,
    "siglip": SIGLIP_VISION_BACKBONE,
    "siglip2": SIGLIP2_VISION_BACKBONE,
    "dinov2_large_336": DINOV2_LARGE_336_VISION_BACKBONE,
    "metaclip_l14_336": METACLIP_L14_336_VISION_BACKBONE,
    "metaclip_b16_224": METACLIP_B16_224_VISION_BACKBONE,
    "metaclip_400m_b16_224": METACLIP_400M_B16_224_VISION_BACKBONE,
}


LLMS: Dict[str, LlmConfig] = {
    "olmo3_7b_instruct": OLMO3_7B_INSTRUCT,
    "qwen3_8b_base": QWEN3_8B_BASE,
    "qwen3_8b": QWEN3_8B,
    "qwen3_4b": QWEN3_4B,
    "qwen3_4b_instruct": QWEN3_4B_INSTRUCT,
}
