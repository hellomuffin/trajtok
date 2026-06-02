import dataclasses
from enum import IntEnum
from typing import Any, Optional, Dict, List
import numpy as np

from olmo.data.video_loader import VideoFrames
from olmo.preprocessing.image_preprocessor import load_image
from olmo.preprocessing.text_preprocessor import InterleavedTextPreprocessor
from olmo.preprocessing.data_formatter import DataFormatter
from olmo.preprocessing.preprocessor_utils import TokenizedVisionData

from olmo.preprocessing.preprocessor_utils import TensorSpec


class MultimodalTypes(IntEnum):
    TEXT_ONLY = 0
    IMAGE = 1
    VIDEO = 2
    MULTI_IMAGE = 3


@dataclasses.dataclass
class MultimodalPreprocessor:
    """Preprocessor that combines text with various types of visual input """
    text_preprocessor: InterleavedTextPreprocessor
    image_preprocessor: Any = None
    video_preprocessor: Any = None
    multi_image_preprocessor: Any  = None
    include_image_metadata: bool = False
    _output_shapes: Optional[Dict[str, TensorSpec]] = None


    @classmethod
    def build(cls, *args, text_seq_len=None, **kwargs) -> 'MultimodalPreprocessor':
        """
        Build a `MultimodalPreprocessor` with max_sequence_length defaulting to
        `text_len` + max number of multi-modal vision tokens if text_len is set
        """
        preprocessor = cls(*args, **kwargs)
        if text_seq_len is not None and preprocessor.text_preprocessor.max_sequence_length is None:
            mm_text_len = max(
                x.get_output_shapes()["tokens"].shape[0] for x in
                [preprocessor.image_preprocessor, preprocessor.video_preprocessor, preprocessor.multi_image_preprocessor]
                if x is not None
            )
            max_seq_len = mm_text_len + text_seq_len
            preprocessor.text_preprocessor.max_sequence_length = max_seq_len
        return preprocessor

    def __call__(
        self,
        messages,
        is_training=False,
        rng=None,
        image=None,
        video=None,
        image_group=None,
        weight=None,
        metadata=None
    ):
        if sum([
            video is not None,
            image is not None,
            image_group is not None
        ]) > 1:
            raise NotImplementedError("Multiple kinds of visual input")

        tokenized_data: Optional[TokenizedVisionData]
        if image is not None:
            if self.image_preprocessor is None:
                raise ValueError("This preprocessor does not support images")
            tokenized_data = self.image_preprocessor(image, is_training=is_training, rng=rng)
        elif video is not None:
            if self.video_preprocessor is None:
                raise ValueError("This preprocessor does not support video")
            tokenized_data = self.video_preprocessor(
                video, is_training=is_training, rng=rng, metadata=metadata)
        elif image_group is not None:
            if self.multi_image_preprocessor is None:
                raise ValueError("This preprocessor does not support multi-image")
            tokenized_data = self.multi_image_preprocessor(
                image_group, is_training=is_training, rng=rng)
        else:
            tokenized_data = None

        if tokenized_data is None:
            example = self.text_preprocessor.tokenize_and_interleave(messages, [], weight=weight)
        elif isinstance(tokenized_data, (list, tuple)):
            assert image_group is not None
            multi_model_pos_ids = (
                None if tokenized_data[0].position_ids is None
                else [tokenized_data[i].position_ids for i in range(len(tokenized_data))]
            )
            example = self.text_preprocessor.tokenize_and_interleave(
                messages,
                [tokenized_data[i].tokens for i in range(len(tokenized_data))],
                multi_model_pos_ids,
                weight=weight
            )
            if tokenized_data[0].image_masks is not None:
                example["image_masks"] = np.concatenate(
                    [tokenized_data[i].image_masks for i in range(len(tokenized_data))]
                )
            if tokenized_data[0].images is not None:
                all_crops = []
                pooled_patches_idx = []
                num_starts = 0
                for i in range(len(tokenized_data)):
                    offset = sum(np.prod(x.shape[:2]) for x in all_crops)
                    pooled_idx_with_offset = np.where(
                        tokenized_data[i].token_pooling >= 0,
                        tokenized_data[i].token_pooling + offset,
                        tokenized_data[i].token_pooling,
                    )
                    pooled_patches_idx.append(pooled_idx_with_offset)
                    all_crops.append(tokenized_data[i].images)
                    num_starts += (tokenized_data[i].tokens == self.image_preprocessor.tokenizer.image_start_token_id).sum()
                    num_starts += (tokenized_data[i].tokens == self.image_preprocessor.tokenizer.low_res_image_start_token_id).sum()
                    num_starts += (tokenized_data[i].tokens == self.image_preprocessor.tokenizer.frame_start_token_id).sum()
                example["images"] = np.concatenate(all_crops)
                example["token_pooling"] = np.concatenate(pooled_patches_idx)
                if self.include_image_metadata:
                    example["num_images"] = np.array([example["images"].shape[0]], dtype=np.int64)
                    example["num_image_starts"] = np.array([num_starts], dtype=np.int64)
        else:
            example = self.text_preprocessor.tokenize_and_interleave(
                messages,
                [tokenized_data.tokens],
                None if tokenized_data.position_ids is None else [tokenized_data.position_ids],
                weight
            )
            if tokenized_data.images is not None:
                example["images"] = tokenized_data.images
                # example["num_images"] = np.array([tokenized_data.images.shape[0]], dtype=np.int64)
            if tokenized_data.image_masks is not None:
                example["image_masks"] = tokenized_data.image_masks
            if tokenized_data.token_pooling is not None:
                example["token_pooling"] = tokenized_data.token_pooling
            if tokenized_data.low_res_token_pooling is not None:
                example["low_res_token_pooling"] = tokenized_data.low_res_token_pooling
            if tokenized_data.cum_token_pooling_bounds is not None:
                example["cum_token_pooling_bounds"] = tokenized_data.cum_token_pooling_bounds
            if tokenized_data.cum_image_bounds is not None:
                example["cum_image_bounds"] = tokenized_data.cum_image_bounds
            if tokenized_data.other_data is not None:
                example.update(tokenized_data.other_data)
            if self.include_image_metadata:
                num_starts =  (tokenized_data.tokens == self.image_preprocessor.tokenizer.image_start_token_id).sum()
                num_starts += (tokenized_data.tokens == self.image_preprocessor.tokenizer.low_res_image_start_token_id).sum()
                num_starts += (tokenized_data.tokens == self.image_preprocessor.tokenizer.frame_start_token_id).sum()
                example["num_image_starts"] = np.array([num_starts], dtype=np.int64)

        multimodal_type = MultimodalTypes.IMAGE if image is not None else (
            MultimodalTypes.VIDEO if video is not None else MultimodalTypes.MULTI_IMAGE if image_group is not None else MultimodalTypes.TEXT_ONLY
        )
        if self.include_image_metadata:
            example["multimodal_type"] = np.array([multimodal_type], dtype=np.int64)
            if "images" not in example:
                example["num_images"] = np.array([0], dtype=np.int64)
                example["num_image_starts"] = np.array([0], dtype=np.int64)
        return example

    def get_output_shapes(self) -> Dict[str, TensorSpec]:
        if self._output_shapes is not None:
            return self._output_shapes
        specs = [
            x.get_output_shapes() for x in
            [self.image_preprocessor, self.video_preprocessor, self.multi_image_preprocessor]
            if x is not None
        ]
        spec = TensorSpec.max_dictionaries(*specs)
        max_seq_len = self.text_preprocessor.max_sequence_length
        if max_seq_len:
            if spec["tokens"].shape[0] > max_seq_len:
                raise ValueError(f"Max sequence length {spec['tokens'].shape[0]} is greater than preprocessor max token length {max_seq_len}")
            spec["tokens"] = TensorSpec([max_seq_len], np.int64)
        else:
            # Unknown since we don't have a bound on the number of tokens
            spec["tokens"] = TensorSpec([None], np.int64)
        if self.include_image_metadata:
            spec["multimodal_type"] = TensorSpec([1], np.int64)
            spec["num_images"] = TensorSpec([1], np.int64)
            spec["num_image_starts"] = TensorSpec([1], np.int64)
        self._output_shapes = spec
        return spec


@dataclasses.dataclass
class ExamplePreprocessor:
    """Preprocesses examples dictionaries as returned by our data loaders

    Includes loading the multi-modal data and formatting the text for the LLM
    """
    formatter: DataFormatter
    preprocessor: MultimodalPreprocessor
    for_inference: bool = False
    is_training: bool = False
    include_image: bool = False

    def get_output_shapes(self) -> Dict[str, TensorSpec]:
        return self.preprocessor.get_output_shapes()

    @property
    def tokenizer(self):
        return self.preprocessor.text_preprocessor.tokenizer

    def __call__(self, example, rng=np.random):
        example = dict(example)
        image: Optional[np.ndarray] = None
        image_group: Optional[List[np.ndarray]] = None
        video: Optional[VideoFrames] = None
        if "image" in example:
            is_image_group = isinstance(example["image"], (list, tuple))
            try:
                if is_image_group:
                    image_group = [load_image(x) for x in example["image"]]
                else:
                    image = load_image(example["image"])
            except Exception as e:
                # Python 3.10 compat: e.add_note() doesn't exist; let outer handler skip.
                raise
            if not is_image_group:
                # So the formatter can know the height/weight of the video
                example["image"] = image
            image_to_video_metadata = None
        if "images" in example:
            example["images"] = [load_image(x) for x in example["images"]]

        if "video" in example:
            if isinstance(example["video"], VideoFrames):
                video_path = None
                video = example["video"]
            else:
                video_path = example["video"]
                try:
                    decode_method = None
                    if "metadata" in example and "decode_method" in example["metadata"]:
                        decode_method = example["metadata"]["decode_method"]
                    clip = None
                    if "metadata" in example and "clip_start_time" in example["metadata"]:
                        if example["metadata"]["clip_start_time"] is not None:
                            clip = (example["metadata"]["clip_start_time"], example["metadata"]["clip_end_time"])
                    subtitle = None
                    if 'subtitle' in example:
                        subtitle = example['subtitle']
                    sampler_overrides = {}
                    if "metadata" in example and "sampler_overrides" in example["metadata"]:
                        sampler_overrides = example["metadata"]["sampler_overrides"]
                    fake_timestamp_fps = None
                    if "metadata" in example and "fake_timestamp_fps" in example["metadata"]:
                        fake_timestamp_fps = example["metadata"]["fake_timestamp_fps"]
                    video = self.preprocessor.video_preprocessor.load_video(example["video"], clip, subtitle=subtitle, decode_method=decode_method, is_training=self.is_training,
                                                                            fake_timestamp_fps=fake_timestamp_fps, **sampler_overrides)

                except Exception as e:
                    # e.add_note requires Python 3.11+; we're on 3.10, and a
                    # crash here kills the dataloader worker → kills the rank →
                    # NCCL timeout cascades across the whole 32-rank job for
                    # ONE bad video. Re-raise the original exception so the
                    # outer iterable_dataset_mixture try/except can skip it.
                    raise
                # So the formatter can know the details of the video
                example["video"] = video

        try:
            messages, formatter_metadata = self.formatter(example, self.is_training, self.for_inference, rng)
        except Exception as e:
            # Python 3.10 compat: e.add_note() doesn't exist; let outer handler skip.
            raise

        if isinstance(messages[0], list):
            # If there are multiple conversations for this example, shuffle their order
            # This might matter if we truncate the tokens to a max sequence length
            rng.shuffle(messages)

        try:
            out = self.preprocessor(
                messages, video=video, image=image,
                image_group=image_group, weight=example.get("weight"),
                metadata=example.get("metadata"),
                rng=rng,
                is_training=self.is_training
            )
        except Exception as e:
            # Python 3.10 compat: e.add_note() doesn't exist; let outer handler skip.
            raise

        if formatter_metadata is None:
            formatter_metadata = {}
        if video is not None:
            h, w = video.frames[0].shape[:2]
            formatter_metadata["image_size"] = (w, h)
        elif image_group is not None:
            image_sizes = [(x.shape[1], x.shape[0]) for x in image_group]
            formatter_metadata["image_group_size"] = image_sizes
        elif image is not None:
            h, w = image.shape[:2]
            formatter_metadata["image_size"] = (w, h)
        else:
            sz = None

        if self.include_image:
            if video is not None:
                formatter_metadata["video_path"] = video_path
                formatter_metadata["video_frames"] = video
            elif image_group is not None:
                formatter_metadata["images"] = example["image"]
            elif image is not None:
                formatter_metadata["image"] = example["image"]
        if "metadata" in example or formatter_metadata:
            metadata = example.get("metadata", {})
            if formatter_metadata:
                metadata.update(formatter_metadata)
            out["metadata"] = metadata
        return out
