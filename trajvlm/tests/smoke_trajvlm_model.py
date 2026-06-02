"""End-to-end smoke test for TrajVLM:
  - construct Molmo2 with TrajVitVisionBackboneConfig (duck-typed)
  - construct Molmo2PreprocessorConfig with TrajVitPreprocessorConfig (duck-typed)
  - run model.forward on a dummy image+text batch
  - check that the loss is finite and that grads flow into both LLM & connector

Run:
    /weka/prior-default/chenhaoz/home/.conda/envs/trajvlm/bin/python \
        reference_code/molmo2/tests/smoke_trajvlm_model.py
"""
import logging
import os
import sys

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("smoke")

# Make HF datasets quiet
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("MOLMO_DATA_DIR", "/weka/oe-training-default/mm-olmo")

from dataclasses import replace  # noqa: E402
from olmo.model_configs import QWEN3_4B_INSTRUCT as _Q  # noqa: E402
# Resolve ${oc.env:MOLMO_DATA_DIR} manually (we instantiate LlmConfig directly,
# bypassing OmegaConf which would otherwise expand it).
QWEN3_4B_INSTRUCT = replace(_Q, init_path=_Q.init_path.replace(
    "${oc.env:MOLMO_DATA_DIR}", os.environ["MOLMO_DATA_DIR"]))
from olmo.models.molmo2.molmo2 import Molmo2Config  # noqa: E402
from olmo.models.molmo2.molmo2_preprocessor import Molmo2PreprocessorConfig  # noqa: E402
from olmo.nn.trajvit_vision_backbone import TrajVitVisionBackboneConfig  # noqa: E402
from olmo.preprocessing.trajvit_preprocessor import TrajVitPreprocessorConfig  # noqa: E402


SEG_CKPT = ("/weka/prior-default/chenhaoz/home/open_videotok/results/ckpts/"
            "pt_filteredmixdata_all/filteredmixdata_all_seg_all_v2_simplesegmenter/latest.pth")


def main():
    device = "cuda"
    log.info("building TrajVLM config")
    backbone_cfg = TrajVitVisionBackboneConfig(
        pretrained_segmenter_path=SEG_CKPT,
    )
    preproc_cfg = TrajVitPreprocessorConfig()

    model_cfg = Molmo2Config(
        llm=QWEN3_4B_INSTRUCT,
        vision_backbone=backbone_cfg,
        mm_preprocessor=Molmo2PreprocessorConfig(
            image=preproc_cfg,
            video=preproc_cfg,
        ),
    )

    log.info("building model")
    from olmo.models.molmo2.molmo2 import Molmo2
    model = Molmo2(model_cfg, device=device).to(device)
    log.info("loading pretrained LLM weights (otherwise logits are all 0 and grads die)")
    model.reset_with_pretrained_weights()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"model built: {n_params:.1f} M params")

    # Build a minimal dummy batch by hand. Mirror the data pipeline output:
    #   input_ids: text + image-placeholder tokens (128 of <im_patch> for one image)
    #   images:    (B, num_image, n_patches=1, pixels = 3*image_res*image_res)
    #   image_masks: None
    #   token_pooling: zeros (unused by our backbone)
    tokenizer = model.config.build_tokenizer()
    image_start = tokenizer.image_start_token_id
    image_end = tokenizer.image_end_token_id
    im_patch = tokenizer.image_patch_token_id

    # text: "Describe: <im_start> 128×<im_patch> <im_end> A photo of a cat. <eos>"
    cap_ids = tokenizer.encode("A photo of a cat.")
    eos = tokenizer.eos_token_id or tokenizer.encode("\n")[0]
    seq = (
        [image_start] + [im_patch] * 128 + [image_end]
        + cap_ids + [eos]
    )
    input_ids = torch.tensor([seq], dtype=torch.long, device=device)
    # labels = -100 for image/non-response tokens; learn on the caption
    labels = torch.full_like(input_ids, -100)
    # response tokens = caption + eos
    response_start = 1 + 128 + 1
    labels[0, response_start: response_start + len(cap_ids) + 1] = input_ids[0, response_start: response_start + len(cap_ids) + 1]

    # images: 224×224×3 noise as a single "patch"
    img_pixels = (np.random.randn(1, 1, 1, 3 * 224 * 224).astype(np.float32) - 0.5) * 0.5
    images = torch.from_numpy(img_pixels).to(device)
    images = images.reshape(1, 1, 3, 224, 224)
    token_pooling = torch.zeros(1, 128, 1, dtype=torch.long, device=device)

    log.info(f"input_ids: {tuple(input_ids.shape)}, images: {tuple(images.shape)}")
    log.info("forward...")
    model.train()
    out = model(
        input_ids=input_ids,
        images=images,
        token_pooling=token_pooling,
        labels=labels,
    )
    logits = out.logits        # (B, seq_len, vocab)
    log.info(f"logits shape: {tuple(logits.shape)}")
    # Standard next-token CE on the response slice, with -100 ignored
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    log.info(f"loss = {loss.item():.4f}")
    assert torch.isfinite(loss), "loss is not finite"
    log.info("backward...")
    loss.backward()

    # Confirm grads flow into LLM, vit backbone, and connector
    def has_grad(name_filter):
        for n, p in model.named_parameters():
            if name_filter(n) and p.grad is not None and p.grad.abs().sum().item() > 0:
                return n
        return None

    where_vit = has_grad(lambda n: "patch_encoder" in n)
    where_perceiver = has_grad(lambda n: "trajectory_perceiver" in n)
    where_projector = has_grad(lambda n: "image_projector" in n)
    where_llm = has_grad(lambda n: ("transformer" in n) and ("vision" not in n))
    log.info(f"grad on patch_encoder: {where_vit}")
    log.info(f"grad on trajectory_perceiver: {where_perceiver}")
    log.info(f"grad on image_projector: {where_projector}")
    log.info(f"grad on LLM transformer: {where_llm}")
    assert all([where_vit, where_perceiver, where_projector, where_llm]), "some path has no grad"

    log.info("OK")


if __name__ == "__main__":
    main()
