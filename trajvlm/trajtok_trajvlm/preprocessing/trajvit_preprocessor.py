"""TrajViT image + video preprocessor.

The TrajViT vision backbone (DINOv3-small + PerceiverResampler) emits a
*fixed* number of trajectory tokens per image / video clip (default 128),
unlike Molmo2's standard ViT path which produces a variable patch-grid
that depends on resize+pool. This preprocessor therefore bypasses
Molmo2's multi-crop logic entirely:

  Input image          → resize to (image_res, image_res) → normalise
                       → emit a fixed token sequence
                         [image_start, <num_traj × image_patch_token_id>, image_end]
                       → `images` shape (1, 1, image_res * image_res * 3)
                         (a single "patch" of size image_res, which our
                         backbone reshapes back to (3, image_res, image_res))

  Input video frames   → uniformly sample `num_frames` per clip
                       → split into clips of `frames_per_clip` (default 16)
                       → per clip: resize + normalise frames, emit
                         [frame_start | image_start, <num_traj × image_patch_token_id>, image_end]
                       → `images` shape (n_clips, frames_per_clip, image_res * image_res * 3)

The placeholder `token_pooling` array is a trivial identity (size num_traj × 1)
so the existing Molmo2 collator code is happy; our backbone forward ignores
it because we don't do per-patch pooling.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import numpy as np
import PIL.Image

from olmo.preprocessing.preprocessor_utils import (
    TensorSpec,
    TokenizedVisionData,
    batch_pixels_to_patches,
)
from olmo.config import BaseConfig
from olmo.preprocessing.multicrop_preprocessor import MultiCropConfig
from olmo.preprocessing.video_preprocessor import VideoPreprocessorConfig


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize_and_normalise(img: np.ndarray, image_res: int, normalize: bool = True) -> np.ndarray:
    """Bicubic-resize a uint8 (H, W, 3) array to (image_res, image_res, 3); optionally normalise.
    Returns float32 (H, W, 3)."""
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    pil = PIL.Image.fromarray(img).resize((image_res, image_res), PIL.Image.BICUBIC)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    if normalize:
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return arr


def _build_image_tokens(num_traj: int, tokenizer) -> np.ndarray:
    """[<im_start>, num_traj × <im_patch>, <im_end>]."""
    pad = np.full((num_traj,), tokenizer.image_patch_token_id, dtype=np.int32)
    return np.concatenate([
        [tokenizer.image_start_token_id],
        pad,
        [tokenizer.image_end_token_id],
    ], 0).astype(np.int32)


def _build_video_tokens(num_clips: int, num_traj: int, tokenizer) -> np.ndarray:
    """Per clip: [<frame_start?>, <im_start>, num_traj × <im_patch>, <im_end>].

    We use `frame_start_token_id` between clips if the tokenizer provides one,
    else just <im_start>. This matches Molmo2's `within_image` bi-directional
    attention scoping (it splits images by `_frame_start` / `_low_res_image_start`).
    """
    chunks = []
    frame_start = getattr(tokenizer, "frame_start_token_id", None)
    for c in range(num_clips):
        if frame_start is not None and c == 0:
            chunks.append([frame_start])
        chunks.extend([
            [tokenizer.image_start_token_id],
            np.full((num_traj,), tokenizer.image_patch_token_id, dtype=np.int32),
            [tokenizer.image_end_token_id],
        ])
    return np.concatenate(chunks, 0).astype(np.int32)


# ---------------------------------------------------------------------------
# Image preprocessor
# ---------------------------------------------------------------------------

class TrajVitImagePreprocessor:
    """Emits exactly `num_traj` <im_patch> placeholders per input image."""

    def __init__(
        self,
        tokenizer,
        image_res: int = 224,
        num_traj: int = 128,
        normalize_on_gpu: bool = False,
    ):
        self.tokenizer = tokenizer
        self.image_res = int(image_res)
        self.num_traj = int(num_traj)
        # Each "patch" we hand back to Molmo2 is the *entire* image (image_res^2 pixels).
        # The vision backbone reshapes (B, T, 1, image_res*image_res*3) back to (B*T, 3, image_res, image_res).
        self.image_patch_size = self.image_res
        self.normalize_on_gpu = bool(normalize_on_gpu)

    def get_output_shapes(self) -> dict:
        sample = np.zeros([self.image_res, self.image_res, 3], dtype=np.uint8)
        td = self(sample, is_training=False)
        return TensorSpec.get_spec(td)

    def __call__(self, image, is_training: bool = False, rng=None) -> TokenizedVisionData:
        if isinstance(image, PIL.Image.Image):
            image = np.asarray(image.convert("RGB"))
        arr = _resize_and_normalise(image, self.image_res, normalize=not self.normalize_on_gpu)
        # Shape into the (n_images, n_patches, pixels) layout Molmo2 expects.
        # n_images=1, n_patches=1, pixels=image_res*image_res*3
        crops = arr[None]                                                  # (1, H, W, 3) float32
        images = batch_pixels_to_patches(crops, self.image_patch_size)      # (1, 1, H*W*3)
        tokens = _build_image_tokens(self.num_traj, self.tokenizer)
        # Trivial token_pooling: not used by our backbone, but the collator expects
        # a (num_image_tokens, pool_dim) array of patch indices.
        token_pooling = np.zeros((self.num_traj, 1), dtype=np.int32)
        return TokenizedVisionData(
            tokens=tokens,
            images=images.astype(np.float32),
            image_masks=None,
            token_pooling=token_pooling,
            cum_image_bounds=np.array([1], dtype=np.int64),
            cum_token_pooling_bounds=np.array([self.num_traj], dtype=np.int64),
        )


# ---------------------------------------------------------------------------
# Video preprocessor
# ---------------------------------------------------------------------------

class TrajVitVideoPreprocessor:
    """Sample `num_frames`, split into `frames_per_clip`-frame clips, emit
    `num_traj` placeholders per clip."""

    def __init__(
        self,
        tokenizer,
        image_res: int = 224,
        num_traj: int = 128,
        num_frames: int = 128,
        frames_per_clip: int = 16,
        normalize_on_gpu: bool = False,
    ):
        self.tokenizer = tokenizer
        self.image_res = int(image_res)
        self.num_traj = int(num_traj)
        self.num_frames = int(num_frames)
        self.frames_per_clip = int(frames_per_clip)
        self.image_patch_size = self.image_res
        self.normalize_on_gpu = bool(normalize_on_gpu)

    @property
    def max_frames(self) -> int:
        return self.num_frames

    def get_output_shapes(self) -> dict:
        from collections import namedtuple
        DummyFrames = namedtuple("DummyFrames", ["frames", "timestamps", "subtitle", "n_frames"])
        sample = DummyFrames(
            frames=np.zeros([self.num_frames, self.image_res, self.image_res, 3], dtype=np.uint8),
            timestamps=np.arange(self.num_frames).astype(np.float32),
            subtitle=None,
            n_frames=self.num_frames,
        )
        td = self(sample, is_training=False)
        return TensorSpec.get_spec(td)

    def load_video(self, *args, **kwargs):
        # The TrainerConfig invokes preprocessor.video_to_patches_and_tokens(video_frames,...)
        # via VideoPreprocessor.load_video. We don't load the video here — we expect
        # to receive raw frames. The data loader is responsible for sampling.
        raise NotImplementedError(
            "TrajVitVideoPreprocessor.load_video is not implemented; "
            "the video data loader should pass pre-loaded frames."
        )

    def __call__(self, video_frames, is_training: bool = False, rng=None, metadata=None) -> TokenizedVisionData:
        # `video_frames` is expected to have a `.frames` attribute (T, H, W, 3) uint8
        if hasattr(video_frames, "frames"):
            frames = video_frames.frames
        else:
            frames = video_frames
        T_in = frames.shape[0]
        # Uniformly sample up to num_frames (or replicate last frame if too few)
        if T_in == 0:
            raise ValueError("TrajVitVideoPreprocessor got zero frames")
        if T_in >= self.num_frames:
            idx = np.linspace(0, T_in - 1, self.num_frames).round().astype(int)
            sampled = frames[idx]
        else:
            # pad with the last frame
            sampled = np.concatenate([
                frames,
                np.tile(frames[-1:], (self.num_frames - T_in, 1, 1, 1)),
            ], axis=0)

        # Resize+normalise per-frame
        resized = np.stack(
            [_resize_and_normalise(sampled[t], self.image_res, normalize=not self.normalize_on_gpu)
             for t in range(self.num_frames)],
            axis=0,
        )

        # Split into clips. Each clip has `frames_per_clip` frames; final clip may be padded.
        nc = (self.num_frames + self.frames_per_clip - 1) // self.frames_per_clip
        clip_arr = np.zeros((nc, self.frames_per_clip, self.image_res, self.image_res, 3), dtype=np.float32)
        for c in range(nc):
            s = c * self.frames_per_clip
            e = min(s + self.frames_per_clip, self.num_frames)
            clip_arr[c, : e - s] = resized[s:e]
            if e - s < self.frames_per_clip:
                clip_arr[c, e - s:] = resized[e - 1]  # repeat last frame

        # Layout to (n_clips, frames_per_clip, H*W*3) — Molmo2 expects (B, num_image, n_patches, pixels)
        # Treat each frame as a "patch" — backbone reshapes back.
        images = clip_arr.reshape(nc * self.frames_per_clip, self.image_res, self.image_res, 3)
        images = batch_pixels_to_patches(images, self.image_patch_size)      # (nc*Tpc, 1, H*W*3)
        # Reshape to (nc, frames_per_clip, H*W*3) so the backbone gets (B, T, ...)
        images = images.reshape(nc, self.frames_per_clip, -1).astype(np.float32)

        tokens = _build_video_tokens(nc, self.num_traj, self.tokenizer)
        token_pooling = np.zeros((nc * self.num_traj, 1), dtype=np.int32)
        return TokenizedVisionData(
            tokens=tokens,
            images=images,
            image_masks=None,
            token_pooling=token_pooling,
            cum_image_bounds=np.array([nc], dtype=np.int64),
            cum_token_pooling_bounds=np.array([nc * self.num_traj], dtype=np.int64),
        )


# ---------------------------------------------------------------------------
# Config wrapper
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrajVitImageConfig(MultiCropConfig):
    """TrajViT image preprocessor config.

    Subclasses `MultiCropConfig` so OmegaConf's strict type check on
    `Molmo2PreprocessorConfig.image` passes. Inherited fields are unused — our
    `build_image_preprocessor` ignores them.
    """
    image_res: int = 224
    num_traj: int = 128
    normalize_on_gpu: bool = False

    def build_image_preprocessor(self, tokenizer, *_, **__) -> Tuple[TrajVitImagePreprocessor, None]:
        ip = TrajVitImagePreprocessor(
            tokenizer,
            image_res=self.image_res,
            num_traj=self.num_traj,
            normalize_on_gpu=self.normalize_on_gpu,
        )
        return ip, None  # we do not support multi-image yet


@dataclasses.dataclass
class TrajVitVideoConfig(VideoPreprocessorConfig):
    """TrajViT video preprocessor config — subclasses VideoPreprocessorConfig
    for OmegaConf strict-type checks on `Molmo2PreprocessorConfig.video`."""
    image_res: int = 224
    num_traj: int = 128
    num_frames: int = 128
    frames_per_clip: int = 16
    normalize_on_gpu: bool = False

    def build_video_preprocessor(self, tokenizer, *_, **__) -> TrajVitVideoPreprocessor:
        return TrajVitVideoPreprocessor(
            tokenizer,
            image_res=self.image_res,
            num_traj=self.num_traj,
            num_frames=self.num_frames,
            frames_per_clip=self.frames_per_clip,
            normalize_on_gpu=self.normalize_on_gpu,
        )


# Back-compat alias — callers passing TrajVitPreprocessorConfig keep working.
TrajVitPreprocessorConfig = TrajVitImageConfig
