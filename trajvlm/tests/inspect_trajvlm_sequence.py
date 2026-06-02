"""Inspect what the data pipeline + backbone produce for one PixMoCap example.

Prints the token layout, identifies image-patch positions, runs the backbone, and
checks that the (B, 128, d_model) trajectory tokens land at exactly the
`<im_patch>` positions in the input.

Run:
  /weka/prior-default/chenhaoz/home/.conda/envs/trajvlm/bin/python \
      reference_code/molmo2/tests/inspect_trajvlm_sequence.py
"""
import os
import sys

import numpy as np
import torch

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("MOLMO_DATA_DIR", "/weka/oe-training-default/mm-olmo")
os.environ.setdefault("HF_HOME", "/weka/oe-training-default/mm-olmo/huggingface")

from dataclasses import replace
from olmo.data.pixmo_datasets import PixMoCap
from olmo.model_configs import QWEN3_4B_INSTRUCT as _Q
from olmo.models.molmo2.molmo2 import Molmo2, Molmo2Config
from olmo.models.molmo2.molmo2_preprocessor import Molmo2PreprocessorConfig
from olmo.nn.trajvit_vision_backbone import TrajVitVisionBackboneConfig
from olmo.preprocessing.trajvit_preprocessor import (
    TrajVitImageConfig,
    TrajVitVideoConfig,
    TrajVitImagePreprocessor,
)

QWEN = replace(_Q, init_path=_Q.init_path.replace("${oc.env:MOLMO_DATA_DIR}", os.environ["MOLMO_DATA_DIR"]))


def main():
    cfg = Molmo2Config(
        llm=QWEN,
        vision_backbone=TrajVitVisionBackboneConfig(),
        mm_preprocessor=Molmo2PreprocessorConfig(
            image=TrajVitImageConfig(),
            video=TrajVitVideoConfig(),
        ),
    )
    tok = cfg.build_tokenizer()

    # 1) Run the IMAGE preprocessor on a real PixMoCap example
    print("=== loading 1 PixMoCap example ===")
    ds = PixMoCap("validation", "captions")
    ex = ds[0]
    print(f"  example keys: {list(ex.keys())}")
    print(f"  caption (first 120ch): {ex.get('caption','')[:120]}")
    # PixMoCap's `image` field is a filesystem path; load via PIL.
    image_field = ex["image"]
    if isinstance(image_field, np.ndarray) and image_field.dtype.kind == "U":
        image_field = str(image_field)
    if isinstance(image_field, (str, bytes, os.PathLike)):
        import PIL.Image
        image = np.asarray(PIL.Image.open(image_field).convert("RGB"))
    else:
        image = np.asarray(image_field)
    print(f"  image shape: {image.shape} dtype={image.dtype}")

    pre = TrajVitImagePreprocessor(tok, image_res=224, num_traj=128)
    td = pre(image)
    print()
    print("=== TokenizedVisionData from image preprocessor ===")
    print(f"  tokens shape: {td.tokens.shape}  dtype={td.tokens.dtype}")
    print(f"  images shape: {td.images.shape}  dtype={td.images.dtype}")
    print(f"  token_pooling shape: {td.token_pooling.shape}")

    # 2) Inspect the token layout
    print()
    print("=== token layout (first 10 + count of <im_patch> + last 5) ===")
    names = {
        tok.image_start_token_id: "<im_start>",
        tok.image_end_token_id:   "<im_end>",
        tok.image_patch_token_id: "<im_patch>",
    }
    print(f"  ids   : {td.tokens[:10].tolist()} ... {td.tokens[-5:].tolist()}")
    print(f"  tokens: {[names.get(int(t), str(int(t))) for t in td.tokens[:5]]} ...")
    n_patch = int((td.tokens == tok.image_patch_token_id).sum())
    print(f"  count <im_patch> = {n_patch}  (expected 128)")
    assert n_patch == 128

    # 3) Build the model + load Qwen3 weights + run backbone forward; verify the
    #    backbone's flat-output count matches placeholder count.
    print()
    print("=== building Molmo2 + backbone ===")
    model = Molmo2(cfg, device="cuda").to("cuda")
    print("  loading pretrained Qwen3-4B + segmenter weights (~3-4 min)…")
    model.reset_with_pretrained_weights()

    # 4) Prepare a batch of size 1 mirroring what the data loader gives the model.
    input_ids = torch.from_numpy(td.tokens.astype(np.int64))[None].cuda()
    images = torch.from_numpy(td.images.astype(np.float32))[None].cuda()  # (B=1, n_image=1, 1, H*W*3)
    images = images.reshape(1, 1, 3, 224, 224)
    token_pooling = torch.from_numpy(td.token_pooling.astype(np.int64))[None].cuda()

    print()
    print("=== forward sanity ===")
    print(f"  input_ids: {tuple(input_ids.shape)}  images: {tuple(images.shape)}")

    # Hook the backbone to capture its raw output
    vfeat_holder = {}
    orig = model.vision_backbone.forward
    def wrap(*a, **kw):
        out = orig(*a, **kw)
        vfeat_holder["v"] = out
        return out
    model.vision_backbone.forward = wrap

    with torch.no_grad():
        out = model(input_ids=input_ids, images=images, token_pooling=token_pooling)

    vfeat = vfeat_holder["v"]
    print(f"  backbone output: shape={tuple(vfeat.shape)}  mean={vfeat.mean().item():+.4f} std={vfeat.std().item():.4f}")
    # Distinctness check — std ACROSS the 128 trajectory rows for each channel
    print(f"  per-channel std across the 128 traj rows: mean={vfeat.std(dim=0).mean().item():.4f} "
          f"max={vfeat.std(dim=0).max().item():.4f}")
    # And pairwise cosine similarity between first 3 rows
    import torch.nn.functional as F
    cos = F.cosine_similarity(vfeat[0:1], vfeat[1:4], dim=-1)
    print(f"  cosine(row0, rows1..3) = {cos.tolist()}")
    n_placeholders = int((input_ids.view(-1) == model._image_high_res_id).sum().item())
    print(f"  <im_patch> placeholders in input_ids: {n_placeholders}")
    print(f"  backbone-output rows: {vfeat.shape[0]}")
    assert vfeat.shape[0] == n_placeholders, "row count must match placeholder count exactly"
    assert vfeat.shape[1] == cfg.llm.d_model, f"d_model mismatch: {vfeat.shape[1]} vs {cfg.llm.d_model}"

    # 5) Inspect resulting embeddings at image positions vs non-image positions.
    wte = model.transformer.wte(input_ids)             # (1, T, D)
    # The += inside Molmo2.forward already mutated x, but logits come back through
    # the LLM. To inspect just-the-add result we re-do it explicitly here.
    x = model.transformer.wte(input_ids).clone()
    mask = input_ids.view(-1) == model._image_high_res_id
    x.view(-1, x.shape[-1])[mask] += vfeat
    print()
    print("=== embedding-stream inspection ===")
    last = input_ids.shape[1] - 1
    for pos in [0, 1, 2, 64, last - 1, last]:
        kind = "image-patch" if mask.view(input_ids.shape)[0, pos].item() else "text"
        print(f"  pos {pos:>4} ({kind:>11}): mean={x[0,pos].mean().item():+.4f}  std={x[0,pos].std().item():.4f}  "
              f"vs wte_only mean={wte[0,pos].mean().item():+.4f} std={wte[0,pos].std().item():.4f}")

    print()
    print(f"=== logits ===")
    print(f"  logits shape: {tuple(out.logits.shape)}  finite? {bool(torch.isfinite(out.logits).all())}")
    log_p = torch.nn.functional.log_softmax(out.logits[0, -1], dim=-1)
    print(f"  last-pos entropy: {-(log_p.exp() * log_p).sum().item():.3f} (vs log V = {np.log(out.logits.size(-1)):.3f})")
    print()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
