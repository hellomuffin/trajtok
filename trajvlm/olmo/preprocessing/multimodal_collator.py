import logging
from typing import Dict, Any, List, Optional, Union

import numpy as np
import torch
from olmo.util import flatten_lists

from olmo import tokenizer
from olmo.preprocessing.preprocessor_utils import TensorSpec, VariablePaddingSpec
from olmo.tokenizer import get_special_token_ids

numpy_to_torch_dtype_dict = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


def _collate(tensors, max_shape=None, dtype=None, pad=None, pad_value=-1, allow_truncate=True):
    batch_shape = np.stack([x.shape for x in tensors if x is not None], 0).max(0)
    if pad == "to_max":
        row_shape = np.array(max_shape)
        assert np.all(batch_shape[1:] <= row_shape[1:])
        if not allow_truncate:
            if batch_shape[0] > row_shape[0]:
                raise ValueError()
            assert batch_shape[0] <= row_shape[0]
    elif pad is None:
        row_shape = batch_shape
    else:
        raise NotImplementedError(pad)

    # get the max per dim for all the dims in [1:] in tensor
    tensor = [x for x in tensors if x is not None][0]
    arr = np.full([len(tensors)] + row_shape.tolist(), pad_value,
                  dtype=dtype or tensor.dtype)
    for ix, tensor in enumerate(tensors):
        if tensor is not None:
            t = tensor[:row_shape[0]]
            slices = tuple(slice(None, dim) for dim in t.shape)
            arr[(ix,) + slices] = t
    return torch.from_numpy(arr)


class MMCollator:
    """Converts list of examples from our datasets into a tensor batch"""
    TEXT_KEYS = ["input_tokens", "target_tokens", "loss_masks", "subsegment_ids", "position_ids"]

    def __init__(self, special_tokens,
                 shapes_to_pad_to: Optional[Dict[str, Union[VariablePaddingSpec, TensorSpec]]]=None,
                 include_metadata=True, pad=None, skip_padding=None, cp_enabled=False):
        """
        :param max_text_len: truncate examples longer than this length
        :param include_metadata: whether to include the metadata in the out batch
        :param pad: how to pad the tensors
        :param max_crops: max number of crops to use if padding to the max sequence length
        """
        if pad:
            assert shapes_to_pad_to is not None
        self.shapes_to_pad_to = shapes_to_pad_to
        self.include_metadata = include_metadata
        self.pad = pad
        self.cp_enabled = cp_enabled
        self._special_tokens = np.array([
            special_tokens[tokenizer.IM_END_TOKEN],
            special_tokens[tokenizer.IM_START_TOKEN],
            special_tokens[tokenizer.IM_COL_TOKEN],
            special_tokens[tokenizer.IMAGE_LOW_RES_TOKEN],
            special_tokens[tokenizer.IMAGE_PATCH_TOKEN],
        ])[None, :]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        for ex in batch:
            if "point_target_ids" in ex:
                assert (ex["point_target_ids"][:, 0] >= 0).sum() == (ex["input_tokens"] == 151947).sum()

        assert len(batch) > 0, "Given an empty batch"
        keys = batch[0].keys()
        if self.pad is not None:
            max_sequence_len = self.shapes_to_pad_to["tokens"].shape[0]
            # Sanity checks
            for ex in batch:
                if np.any(self._special_tokens == ex["input_tokens"][max_sequence_len:][:, None]):
                    raise ValueError("An image would have gotten truncated!")
                if not self.cp_enabled:
                    ## In CP, as a device might only process image + prompt tokens and no response tokens where the loss
                    ## would be zero whch is ok.
                    if np.any(ex["loss_masks"] != 0) and np.all(ex["loss_masks"][:max_sequence_len] == 0):
                        raise ValueError("All loss tokens truncated!")
        else:
            max_sequence_len = None

        # Collate text fields
        out = {}
        for key in self.TEXT_KEYS:
            # If one example has subsegment_ids, all examples need it as well
            # Note it is okay if some batches have subsegment_ids and some (for different devices)
            # don't since it only used to modify the attention mask
            if key == "subsegment_ids":
                if any(key in ex for ex in batch):
                    for ex in batch:
                        if "subsegment_ids" not in ex:
                            ex["subsegment_ids"] = np.ones_like(ex["input_tokens"])
                else:
                    continue
            dtype = np.float32 if key == "loss_masks" else np.int64
            # for ex in batch:
            # if ex[key].shape[0] > max_sequence_len:
            #     raise ValueError(f"{key}: {ex[key].shape[0]} vs {max_sequence_len}")
            out[key] = _collate(
                [ex.get(key) for ex in batch], [max_sequence_len], dtype, pad=self.pad)

        # Collate any other fields
        for key, spec in self.shapes_to_pad_to.items():
            if key == "tokens":
                continue
            tensors = [ex.get(key) for ex in batch]
            if all(x is None for x in tensors):
                if self.pad is not None:
                    # Create an all-padding input, we might need this to make sure each device
                    # in a FSDP setup gets the same inputs
                    out[key] = torch.full(
                        [len(tensors)] + list(spec.shape), -1,
                        dtype=numpy_to_torch_dtype_dict[spec.dtype],
                        )
            else:
                if isinstance(spec, VariablePaddingSpec):
                    pad = None
                else:
                    pad = self.pad
                pad_value = 0 if spec.dtype == np.uint8 else -1
                out[key] = _collate([ex.get(key) for ex in batch], spec.shape,
                                    dtype=spec.dtype, pad=pad, pad_value=pad_value, allow_truncate=False)

        out["input_ids"] = out.pop("input_tokens")

        if "cum_token_pooling_bounds" in keys:
            out["cum_token_pooling_bounds"] = [torch.from_numpy(ex.get("cum_token_pooling_bounds", np.array([], dtype=np.int64))) for ex in batch]

        if "cum_image_bounds" in keys:
            out["cum_image_bounds"] = [torch.from_numpy(ex.get("cum_image_bounds", np.array([], dtype=np.int64))) for ex in batch]
        
        if "image_shard_boundaries" in keys:
            out["image_shard_boundaries"] = [ex.get("image_shard_boundaries", {}) for ex in batch]

        if "target_tokens" in out:
            out["labels"] = out.pop("target_tokens")
        
        out["metadata"] = [ex.get("metadata", {}) for ex in batch]

        # Maybe add metdata or worker state
        if "data_worker_state" in batch[0]:
            out["data_worker_state"] = [ex["data_worker_state"] for ex in batch]
        if self.include_metadata:
            out["metadata"] = [ex.get("metadata", {}) for ex in batch]
        return out
