import dataclasses
from typing import Optional, Tuple, Union

from olmo.config import D
from olmo.preprocessing.text_preprocessor import TextPreprocessorConfig
from olmo.preprocessing.image_preprocessor import ImagePreprocessor
from olmo.preprocessing.multicrop_preprocessor import MultiCropImagePreprocessor, \
    MultiImagePreprocessor, MultiCropConfig
from olmo.preprocessing.multimodal_preprocessor import MultimodalPreprocessor


@dataclasses.dataclass
class MolmoPreprocessorConfig(TextPreprocessorConfig, MultiCropConfig):
    image_padding_mask: Union[bool, int] = False
    legacy_image_mask: bool = False

    def build(self, tokenizer, image_preprocessor: ImagePreprocessor, text_seq_len, max_seq_len):
        image, multi_image = self.build_image_preprocessor(
            tokenizer,
            image_preprocessor,
            self.image_padding_mask,
            self.legacy_image_mask
        )
        return MultimodalPreprocessor.build(
            text_preprocessor=self.build_text_preprocessor(tokenizer, max_seq_len),
            multi_image_preprocessor=multi_image,
            image_preprocessor=image,
            text_seq_len=text_seq_len,
        )
