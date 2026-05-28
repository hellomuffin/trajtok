# TrajTok-v2

Official open-source release for **"TrajTok-v2: Trajectory-aware visual tokenization for VLMs"** (Zheng et al., 2026).

📄 **Paper:** [arXiv:2602.22779](https://arxiv.org/abs/2602.22779)
🤗 **Released checkpoints:**
- Segmenter — [michaelzch001/trajtokv2-segmenter](https://huggingface.co/michaelzch001/trajtokv2-segmenter)
- TrajViT-v2 — [michaelzch001/trajtokv2-trajvitv2](https://huggingface.co/michaelzch001/trajtokv2-trajvitv2)

TrajTok-v2 produces a small number of *trajectory tokens* per image / video clip
that bundle pixels which belong to the same object instance over space and time.
These tokens are then consumed by a vision transformer (TrajViT-v2) or a VLM
(TrajVLM) at a fraction of the LLM-token cost of traditional patch tokenisation,
while preserving fine-grained object-grounded information.

This repository hosts three self-contained sub-packages:

| Package | What it gives you |
|---|---|
| [`segmenter/`](./segmenter) | The **trajectory segmenter** — DINOv3-small ConvNeXt + PerceiverResampler + soft-mask grouping. Trained on ~12 M images + videos (SA-1B, SA-V, internal mix). **Released checkpoint + easy evaluation + qualitative demo.** |
| [`trajvitv2/`](./trajvitv2) | **TrajViT-v2** — a SegmentTokenizer wrapping the segmenter, followed by a CLIP-style ViT-Large transformer over trajectory tokens. **Training + evaluation + checkpoint** (small-scale Panda-70M filtered subset; not a final-scale model). |
| [`trajvlm/`](./trajvlm) | **TrajVLM** — vision-language model: SigLIP2 ViT features pooled by our segmenter into trajectory tokens, fed into Qwen3-4B-Instruct. **Training + evaluation code.** |

Each sub-package has its own `README.md`, dependencies, and quickstart. Start
with the one matching what you want to do:

- **Want to use trajectory tokens for your own model?** → [`segmenter/`](./segmenter)
- **Want to reproduce TrajViT-v2 retrieval experiments?** → [`trajvitv2/`](./trajvitv2)
- **Want to train a VLM with trajectory tokens?** → [`trajvlm/`](./trajvlm)

## Citation

```bibtex
@article{zheng2026trajtokv2,
  title   = {TrajTok-v2: Trajectory-aware visual tokenization for vision-language models},
  author  = {Zheng, Chenhao and others},
  journal = {arXiv preprint arXiv:2602.22779},
  year    = {2026},
}
```

## License

Apache-2.0 — see [LICENSE](./LICENSE).

The DINOv3 ConvNeXt backbone used by the segmenter is bundled separately under
its own Apache-2.0 license (Meta AI). SigLIP2 weights used by TrajVLM are
licensed by Google under the Apache-2.0 license.
