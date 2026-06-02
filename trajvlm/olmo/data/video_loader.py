import dataclasses
import functools
import io
import logging
import math
import re
from collections import Counter
from os.path import basename, dirname
from typing import List, Tuple, Optional, Union, Dict
import decord
import av
import numpy as np
from decord import VideoReader, cpu

from olmo.config import BaseConfig
from olmo.io import resource_path, read_file, is_url
from olmo.torch_util import get_global_rank

decord.logging.set_level(2)
av.logging.set_level(av.logging.PANIC)


@dataclasses.dataclass
class VideoFrames:
    "Frames from a video and frame metadata"

    frames: np.ndarray
    """Frames as RGB images"""

    timestamps: np.ndarray
    """Timestamps for each frame"""

    target_fps: Optional[float]
    """Target FPS used to sample the frames, if there was one"""

    sampling_augmentation: Optional[str] = None
    """Augmentation used"""

    subtitle: Optional[str] = None
    """Subtitle text associated with the video"""

    def __post_init__(self):
        assert len(self.timestamps) == len(self.frames)
        assert len(self.frames.shape) == 4
        assert self.frames.shape[-1] == 3
        if self.target_fps is not None:
            self.target_fps = float(self.target_fps)

    def __len__(self):
        return len(self.frames)

    @property
    def sampled_fps(self) -> float:
        if self.target_fps is None:
            return 1 / (self.timestamps[1:] - self.timestamps[:-1]).mean()
        else:
            return self.target_fps


def get_sampling_fps(
    video_fps: float,
    max_frames: int,
    total_frames: int,
    frame_sample_mode: str,
    candidate_sampling_fps: Tuple[float],
) -> float:
    """
    Get the sampling fps that best spans the video and has the most frames sampled
    """
    num_frames_sampled = 0
    selected_sampling_fps = None
    for sampling_fps in candidate_sampling_fps:
        step_size = max(int(video_fps / sampling_fps), 1)
        num_frames_sampled_at_fps = int(total_frames / step_size)
        if num_frames_sampled == 0:
            if "uniform" in frame_sample_mode:
                if num_frames_sampled_at_fps > max_frames:
                    break
            selected_sampling_fps = sampling_fps
            num_frames_sampled = num_frames_sampled_at_fps

        else:
            # the candidate sampling fps increases so frame count can't decrease
            assert num_frames_sampled <= num_frames_sampled_at_fps
            if num_frames_sampled_at_fps > max_frames:
                # choose the sampling fps that spans the video
                continue

            elif num_frames_sampled_at_fps > num_frames_sampled:
                # both are less than max_frames, choose the one with higher density of frames sampled
                selected_sampling_fps = sampling_fps
                num_frames_sampled = num_frames_sampled_at_fps
    return selected_sampling_fps


def get_frame_times_and_chosen_fps(selected_sampling_fps, total_frames, max_frames, video_fps):
    if selected_sampling_fps is None:
        frame_indices = np.linspace(0, total_frames, max_frames, endpoint=False, dtype=int)
    else:
        step_size = max(int(video_fps / selected_sampling_fps), 1)
        frame_indices = np.arange(0, total_frames, step_size)
    if len(frame_indices) > max_frames:
        frame_indices = frame_indices[:max_frames]
    return selected_sampling_fps, frame_indices

def _sampler_with_overrides(sampler, **sampler_overrides):
    if not sampler_overrides:
        return sampler
    valid_fields = {field.name for field in dataclasses.fields(sampler)}
    filtered_overrides = {} # only replace existing files in sampler
    for key, value in sampler_overrides.items():
        if key in valid_fields:
            filtered_overrides[key] = value
        # else:
        #     logging.getLogger(__name__).debug(
        #         "Ignoring sampler override '%s' for %s; field not present.",
        #         key, type(sampler).__name__,
        #     )

    if not filtered_overrides:
        return sampler
    # dataclass-friendly: shallow copy with selected fields replaced
    return dataclasses.replace(sampler, **filtered_overrides)

@dataclasses.dataclass
class TimeSampler:
    """Decides how to sample timestamps from a video based on duration"""
    max_frames: int = 8
    frame_sample_mode: str = "fps",
    candidate_sampling_fps: Tuple[float] = (0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 16.0)
    max_fps: Union[float, None, Tuple[Optional[float]]] = None
    is_training: bool = False

    def __call__(self, duration) -> Tuple[float, np.ndarray, Optional[str]]:
        """Decide what time stamps to sample"""
        max_frames = self.max_frames
        if self.frame_sample_mode == "uniform":
            if self.max_fps:
                raise NotImplementedError("Max FPS with uniform")
            times = np.linspace(0, duration, num=max_frames, endpoint=False, dtype=np.float64)
            return None, times, None
        if self.frame_sample_mode in ["uniform_last_frame", "uniform_last_frame_sample_fps"]:
            if self.frame_sample_mode == "uniform_last_frame_sample_fps":
                start, end = self.max_fps
                if self.is_training:
                    if np.random.random() < 0.1:
                        max_fps = start
                    else:
                        max_fps = np.random.uniform(start, end)
                else:
                    max_fps = start
            elif isinstance(self.max_fps, (tuple, list)):
                if self.is_training and len(self.max_fps) > 1:
                    max_fps = self.max_fps[np.random.randint(len(self.max_fps))]
                else:
                    max_fps = self.max_fps[0]
            else:
                max_fps = self.max_fps

            if max_fps is not None:
                max_duration = (self.max_frames-1) / max_fps  # -1 to include the last frame
                if max_duration < duration:
                    times = np.linspace(0, duration, num=max_frames, endpoint=True, dtype=np.float64)
                else:
                    times = np.arange(0.0, stop=duration, step=1/max_fps)
                    times = np.concatenate([times, [duration]], axis=0)
                    assert len(times) <= self.max_frames
            else:
                times = np.linspace(0, duration, num=max_frames, endpoint=True, dtype=np.float64)
            return None, times, None
        elif self.frame_sample_mode == "fps":
            # Try larger and larger FPSs until we hit one that can't span the video
            sampling_fps = self.candidate_sampling_fps[0]
            for candidate_fps in self.candidate_sampling_fps[1:]:
                if max_frames/candidate_fps < duration:
                    break
                sampling_fps = candidate_fps
            times = np.arange(0, max_frames) / sampling_fps
            times = times[times < duration]
            return sampling_fps, times, None
        else:
            raise NotImplementedError(self.frame_sample_mode)


@dataclasses.dataclass
class FrameSampler:
    """Decides how to sample frames from a video based on number of frames"""
    max_frames: int = 8
    frame_sample_mode: str = "fps"
    candidate_sampling_fps: Tuple[float] = (0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 16.0)
    rng: np.random.RandomState = None
    min_fps: int = None
    is_training: bool = False

    def __call__(self, video_fps, total_frames) -> Tuple[float, np.ndarray, Optional[str]]:
        """Decide what frame indices to sample"""
        assert total_frames > 0
            
        rng = self.rng if self.rng is not None else np.random
        if self.frame_sample_mode == "uniform":
            times = np.linspace(0, total_frames, num=min(self.max_frames, total_frames), endpoint=False, dtype=np.int32)
            return None, times, None
        elif self.frame_sample_mode == "debug":
            n = np.random.randint(1, min(self.max_frames, total_frames))
            times = np.linspace(0, total_frames, num=n, endpoint=False, dtype=np.int32)
            return None, times, None
        elif (self.frame_sample_mode.startswith("uniform_last_frame_min_") or
              self.frame_sample_mode.startswith("uniform_last_frame_max_fps_set_") or
              (self.frame_sample_mode == "uniform_last_frame" and self.min_fps is not None)
        ):
            if self.frame_sample_mode == "uniform_last_frame":
                min_fps = self.min_fps
            elif self.frame_sample_mode.startswith("uniform_last_frame_max_fps_set_"):
                options = self.frame_sample_mode.split("uniform_last_frame_max_fps_set_")[1].split("-")
                options = [float(x) for x in options]
                if not self.is_training:
                    min_fps = options[0]  # 0th option is eval default
                else:
                    min_fps = np.random.choice(options)
            # These other cases are for backwards-compatibility
            elif self.frame_sample_mode.startswith("uniform_last_frame_min_2-4"):
                if not self.is_training:
                    min_fps = 2
                else:
                    min_fps = 2 if (np.random.random() > 0.5) else 4
            elif self.frame_sample_mode.startswith("uniform_last_frame_min_2-6"):
                if not self.is_training:
                    min_fps = 2
                else:
                    r = np.random.random()
                    if r < 0.333:
                        min_fps = 2
                    elif r < 0.666:
                        min_fps = 4
                    else:
                        min_fps = 6
            else:
                min_fps = float(re.fullmatch("uniform_last_frame_min_([0-9\.]+)fps", self.frame_sample_mode).group(1))
            duration = total_frames / video_fps
            if total_frames <= 2:
                return None, np.arange(total_frames, dtype=np.int64), None
            if duration > (self.max_frames/min_fps - 1):  # -1 for first and last frame
                # uniform fallback
                times = np.linspace(0, total_frames-1, num=min(self.max_frames, total_frames), endpoint=True, dtype=np.int32)
                return None, times, None
            else:
                float_indices = np.arange(0.0, stop=total_frames-1, step=float(video_fps/min_fps))
                if np.round(float_indices[-1]) != total_frames-1:
                    float_indices = np.concatenate([float_indices, [total_frames-1]], axis=0)
                indices = np.round(float_indices)
                assert indices[-1] < total_frames
                assert len(float_indices) <= self.max_frames
                return min_fps, indices.astype(np.int32), None
        elif self.frame_sample_mode == "uniform_last_frame":
            times = np.linspace(0, total_frames-1, num=min(self.max_frames, total_frames), endpoint=True, dtype=np.int32)
            return None, times, None
        elif self.frame_sample_mode == "uniform_randomized":
            aug = None
            if total_frames <= self.max_frames:
                indices = np.arange(0, total_frames, dtype=np.int32)
                step = 1
            else:
                step = total_frames // self.max_frames
                indices = np.arange(0, self.max_frames, dtype=np.int32) * step
                remainder = (total_frames - step * (self.max_frames - 1))
                if rng.random() > 0.3 and self.is_training:
                    offset = rng.randint(-(step//2), remainder//2)
                    indices[1:] += offset
                    assert indices[1] > indices[0] and indices[-1] < total_frames
                    aug = "RE"
            return None, indices, aug
        elif self.frame_sample_mode in ["fps", "fps_uniform"]:
            selected_sampling_fps = get_sampling_fps(video_fps, self.max_frames, total_frames, self.frame_sample_mode, self.candidate_sampling_fps)
            sampling_fps, frame_indices = get_frame_times_and_chosen_fps(selected_sampling_fps, total_frames, self.max_frames, video_fps)
            return selected_sampling_fps, frame_indices, None
        else:
            raise NotImplementedError(self.frame_sample_mode)


@dataclasses.dataclass
class VideoLoaderConfig(BaseConfig):
    max_frames: int = 8
    frame_sample_mode: str = "fps"
    candidate_sampling_fps: Tuple[float] = (0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 16.0)
    cache_videos: bool = True
    loading_method: str = "decord_with_av_fallback"
    max_fps: Optional[Tuple[float]] = None
    time_sampling: bool = False

    def build_video_loader(self) -> 'VideoLoader':
        if self.time_sampling:
            sampler = TimeSampler(self.max_frames, self.frame_sample_mode, self.candidate_sampling_fps, self.max_fps)
        else:
            sampler = FrameSampler(self.max_frames, self.frame_sample_mode, self.candidate_sampling_fps, self.max_fps)
        return VideoLoader(sampler, self.cache_videos, self.loading_method)


@dataclasses.dataclass
class VideoLoader:
    sampler: Union[FrameSampler, TimeSampler]
    cache_videos: bool = True
    loading_method: str = "decord_with_av_fallback"

    def __call__(self, video_path: str, clip: Optional[Tuple[float, float]] = None,
                 subtitle: Optional[Union[str, Dict]] = None, decode_method: Optional[str] = None, is_training=False,
                 fake_timestamp_fps: Optional[int] = None, **sampler_overrides) -> VideoFrames:
        """Load frames from a video"""
        if self.cache_videos:
            video_path = resource_path(dirname(video_path), basename(video_path)).as_posix()
        elif is_url(video_path):
            video_path = io.BytesIO(read_file(video_path, "rb"))

        # Apply any overrides to the sampler for this video example
        sampler = _sampler_with_overrides(self.sampler, **sampler_overrides)
        # Somehow, this assignment has silent failure when self.sampler does not have a is_training
        sampler.is_training = is_training 

        if self.loading_method == "torchcodec_exact":
            frames, timestamps, target_fps, aug = load_video_torchcodec(video_path, sampler, clip, "exact")
        elif self.loading_method == "torchcodec_exact_dummy":
            frames, timestamps, target_fps, aug = load_video_torchcodec_dummy(video_path, sampler, clip, "exact")
        elif self.loading_method == "torchcodec_approx":
            frames, timestamps, target_fps, aug =  load_video_torchcodec(video_path, sampler, clip, "approximate")
        elif self.loading_method == "av":
            frames, timestamps, target_fps, aug =  load_video_av_noseek(video_path, sampler, clip)
        elif self.loading_method == "decord_with_av_fallback":
            if decode_method is None:
                try:
                    frames, timestamps, target_fps, aug =  load_video_decord(video_path, sampler, clip)
                except Exception as e:
                    frames, timestamps, target_fps, aug =  load_video_av_noseek(video_path, sampler, clip)
            elif decode_method == "decord":
                frames, timestamps, target_fps, aug =  load_video_decord(video_path, sampler, clip)
            elif decode_method == "av_noseek":
                frames, timestamps, target_fps, aug =  load_video_av_noseek(video_path, sampler, clip)
            else:
                raise ValueError(f"Unknown decode method {decode_method}, must be one of 'decord' or 'av_noseek'")
        else:
            raise NotImplementedError(self.loading_method)
        video_frames = VideoFrames(
            frames=frames,
            timestamps=timestamps,
            target_fps=target_fps,
            sampling_augmentation=aug,
            subtitle=subtitle,
        )
        if fake_timestamp_fps:
            # this is a hack to make the timestamps look like they were sampled at a particular fps
            # the last frame is processed separately since it may have a different time delta than the rest
            avg_fps_wo_last_frame = 1 / (timestamps[1:-1] - timestamps[:-2]).mean()
            eps = 0.5
            if abs(avg_fps_wo_last_frame - 2.0) >= eps:
                raise ValueError(f"This setup assumes the fps=2.0 but got fps={avg_fps_wo_last_frame} (ts={timestamps}). The conversion may be incorrect")
            multiplier = avg_fps_wo_last_frame / fake_timestamp_fps
            fake_timestamps = timestamps * multiplier
            fake_timestamps[-1] = fake_timestamps[-2] + 1 / fake_timestamp_fps
            return VideoFrames(
                frames=video_frames.frames,
                timestamps=fake_timestamps,
                target_fps=fake_timestamp_fps,
                sampling_augmentation=aug,
                subtitle=subtitle,
            )
        else:
            return video_frames


def _validate_clip(clip, duration):
    if clip[0] >= clip[1]:
        raise ValueError(f"Clip {clip} has start>=end")
    if clip[0] >= duration:
        raise ValueError(f"Invalid clip, start={clip[0]} but video duration={duration}")


def load_video(video_path: str, sample_fn, clip: Tuple[float, float]=None) -> VideoFrames:
    """Load frames from a video"""
    try:
        load_video_decord(video_path, sample_fn, clip)
    except Exception as e:
        return load_video_av_noseek(video_path, sample_fn, clip)


def get_video_size(video_path: str) -> Tuple[int, int]:
    from torchcodec.decoders import VideoDecoder
    video_dec = VideoDecoder(video_path, num_ffmpeg_threads=1)
    w, h = video_dec.metadata.width, video_dec.metadata.height
    assert w is not None and h is not None
    return w, h

def load_video_torchcodec(
    video_path: str,
    frame_sampler: TimeSampler,
    clip=None,
    seek_mode="exact"
):
    from torchcodec.decoders import VideoDecoder
    video_dec = VideoDecoder(video_path, seek_mode=seek_mode, num_ffmpeg_threads=1)
    fps = video_dec.metadata.average_fps
    if isinstance(frame_sampler, TimeSampler):
        # If the first frame starts at > 0, we effectively clip the video starting at that time
        # since (most) video players would also skip to that time
        time_offset = video_dec.metadata.begin_stream_seconds_from_content
        # Note this duration does assume we started playing at `time_offset`
        duration = video_dec.metadata.duration_seconds
        if clip is not None:
            _validate_clip(clip, duration)
            clip_duration = min(clip[1], duration)-clip[0]
            target_fps, target_timestamps, aug = frame_sampler(clip_duration)
            time_offset += clip[0]
        else:
            target_fps, target_timestamps, aug = frame_sampler(duration)

        # Floating point/rounding issues might cause `target_timestamps` to be very slightly
        # out-of-bounds, to handle this we sanity check then clip them
        assert all(x >= 0 for x in target_timestamps)
        assert all(x < duration+1e-6 for x in target_timestamps)
        # 1e-6 padding since torchcodec can throw out-of-bounds errors even if you ask for the
        # exact boundary value, we should still get the first/last frame anyway
        max_timestamp = video_dec.metadata.end_stream_seconds_from_content - 1e-6
        min_timestamp = video_dec.metadata.begin_stream_seconds_from_content + 1e-6
        # Note we avoid using numpy ops here to reduce floating precision issues
        timestamps = [x + time_offset for x in target_timestamps]
        timestamps = [max(min_timestamp, min(max_timestamp, x)) for x in timestamps]
        frames = video_dec.get_frames_played_at(timestamps)
        target_timestamps = np.array(target_timestamps)
    else:
        n_frames = video_dec.metadata.num_frames
        if clip is not None:
            duration = n_frames / fps
            _validate_clip(clip, duration)
            start_index = math.floor(clip[0] * fps)
            end_index = min(math.ceil(clip[1] * fps), n_frames)
            target_fps, indices, aug = frame_sampler(fps, end_index-start_index)
        else:
            start_index = 0
            target_fps, indices, aug = frame_sampler(fps, n_frames)
        frames = video_dec.get_frames_at(indices+start_index)
        target_timestamps = np.array(indices)/fps
    return (
        frames.data.numpy().transpose(0, 2, 3, 1),
        target_timestamps,
        target_fps,
        aug
    )


def load_video_torchcodec_dummy(
    video_path: str,
    frame_sampler: TimeSampler,
    clip=None,
    seek_mode="exact"
):
    from torchcodec.decoders import VideoDecoder
    video_dec = VideoDecoder(video_path, seek_mode=seek_mode, num_ffmpeg_threads=1)
    fps = video_dec.metadata.average_fps
    if isinstance(frame_sampler, TimeSampler):
        # If the first frame starts at > 0, we effectively clip the video starting at that time
        # since (most) video players would also skip to that time
        time_offset = video_dec.metadata.begin_stream_seconds_from_content
        # Note this duration does assume we started playing at `time_offset`
        duration = video_dec.metadata.duration_seconds
        if clip is not None:
            _validate_clip(clip, duration)
            clip_duration = min(clip[1], duration)-clip[0]
            target_fps, target_timestamps, aug = frame_sampler(clip_duration)
            time_offset += clip[0]
        else:
            target_fps, target_timestamps, aug = frame_sampler(duration)
        return (
            np.ones([len(target_timestamps), 378, 378, 3], dtype=np.uint8),
            np.array(target_timestamps),
            target_fps,
            aug
        )
    else:
        raise NotImplementedError()


def load_video_decord(video_path: str, sample_fn: Union[TimeSampler, FrameSampler], clip: Tuple[float, float]=None) -> VideoFrames:
    """Load video frames with decord"""
    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    video_fps = vr.get_avg_fps()
    if isinstance(sample_fn, TimeSampler):
        time_stamps = vr.get_frame_timestamp(list(range(len(vr))))
        duration = time_stamps[-1][1] - time_stamps[0][0]
        if clip is not None:
            _validate_clip(clip, duration)
            clip_duration = min(clip[1], duration)-clip[0]
            target_fps, target_timestamps, aug = sample_fn(clip_duration)
            target_timestamps = np.array(target_timestamps)
            offset = clip[0]
        else:
            target_fps, target_timestamps, aug = sample_fn(duration)
            target_timestamps = np.array(target_timestamps)
            offset = time_stamps[0, 0]
        ix = np.searchsorted(time_stamps[:, 1], target_timestamps + offset, side='right')
        ix = np.minimum(ix, len(time_stamps) - 1)
        frames = vr.get_batch(ix).asnumpy()
        return (
            frames,
            np.array(target_timestamps),
            None,
            aug
        )
    else:
        if clip:
            _validate_clip(clip, len(vr)/video_fps)
            start_frame = math.floor(clip[0] * video_fps)
            end_frame = min(math.ceil(clip[1] * video_fps), len(vr))
            sampling_fps, frame_indices, aug = sample_fn(video_fps, end_frame-start_frame)
        else:
            sampling_fps, frame_indices, aug = sample_fn(video_fps, len(vr))
            start_frame = 0
        frames = vr.get_batch(frame_indices + start_frame).asnumpy()
        return (
            frames,
            np.array(frame_indices)/video_fps,
            sampling_fps,
            aug
        )


def load_video_av_noseek(
    video_path: str,
    frame_sampler: Union[FrameSampler, TimeSampler],
    clip=None,
) -> VideoFrames:
    """Load a video frames by decoding all frames with pyav

    More robust than `load_video_decord` but can by much slower for long videos
    """
    # Behaves the same as the old version using `imageio.v3` but avoid extra the dependency
    with av.open(video_path) as container:
        video_stream = container.streams.video[0]
        fps = video_stream.average_rate or video_stream.guessed_rate
        it = container.decode(video=0)
        frames = list(it)
        if isinstance(frame_sampler, TimeSampler):
            stream = container.streams.video[0]
            start = frames[0].pts * stream.time_base
            container_end = stream.duration
            if container_end is not None:
                container_end *= stream.time_base
            if container_end is None or container_end < frames[-1].pts:
                # Some problem with stream duration, so use the frame PTS directly
                # and guess the duration of the last frame
                end = frames[-1].pts * stream.time_base + 1/fps
            else:
                end = container_end
            duration = float(end - start)
            if clip is not None:
                _validate_clip(clip, duration)
                clip_duration = min(clip[1], duration)-clip[0]
                target_fps, timestamps, aug = frame_sampler(clip_duration)
                offset = clip[0]
            else:
                target_fps, timestamps, aug = frame_sampler(duration)
                offset = float(start)
            timestamps = np.array(timestamps)
            end_time_stamps = np.array([float(frame.pts * stream.time_base) for frame in frames[1:]] + [duration])
            indices = np.searchsorted(end_time_stamps, timestamps + offset, side='right')
            indices = np.minimum(indices, len(frames) - 1)
        else:
            if clip is not None:
                duration = len(frames) / fps
                _validate_clip(clip, duration)
                start_index = math.floor(clip[0] * fps)
                end_index = min(math.ceil(clip[1] * fps), len(frames))
                target_fps, indices, aug = frame_sampler(fps, end_index-start_index)
            else:
                start_index = 0
                target_fps, indices, aug = frame_sampler(fps, len(frames))
            indices = np.array(indices)
            timestamps = np.array(indices)/float(fps)
            indices += start_index

        frames = [frames[i].to_ndarray(format="rgb24") for i in indices]
        return (
            np.stack(frames, axis=0),
            timestamps,
            target_fps,
            aug
        )
