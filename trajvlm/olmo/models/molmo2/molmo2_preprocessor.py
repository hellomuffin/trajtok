import dataclasses
from omegaconf import OmegaConf

from olmo.config import D
from olmo.preprocessing.multicrop_preprocessor import MultiCropConfig
from olmo.preprocessing.multimodal_preprocessor import MultimodalPreprocessor
from olmo.preprocessing.text_preprocessor import TextPreprocessorConfig
from olmo.preprocessing.video_preprocessor import VideoPreprocessorConfig


@dataclasses.dataclass
class Molmo2PreprocessorConfig(TextPreprocessorConfig):
    """Molmo2 preprocessor configuration"""

    video: VideoPreprocessorConfig = dataclasses.field(default_factory=VideoPreprocessorConfig)
    image: MultiCropConfig = dataclasses.field(default_factory=MultiCropConfig)

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        if not hasattr(config, "video"):
            # Old style config that didn't separate the video config parameters
            return OmegaConf.structured(Molmo2PreprocessorConfig(
                image=config.image,
                video=VideoPreprocessorConfig.build_from_legacy_config(config),
                max_answer_len=config.max_answer_len,
                last_message_loss_only=config.last_message_loss_only,
                max_text_tokens=config.max_text_tokens,
                loss_token_weighting=config.loss_token_weighting,
            ))
        return config

    def build(self, tokenizer, image_preprocessor, text_seq_len=None, max_sequence_length=None) -> MultimodalPreprocessor:
        if self.image is not None:
            image, multi_image = self.image.build_image_preprocessor(
                tokenizer, image_preprocessor)
        else:
            image, multi_image = None, None
        return MultimodalPreprocessor.build(
            text_preprocessor=self.build_text_preprocessor(tokenizer, max_sequence_length),
            image_preprocessor=image,
            multi_image_preprocessor=multi_image,
            video_preprocessor=self.video.build_video_preprocessor(tokenizer, image_preprocessor),
            text_seq_len=text_seq_len
        )
