# Grounding DINO + SAM2 Integration Feasibility

> Assessment Date: 2026-07-31 | Target: box segmentation (replacing class-agnostic AutoMaskGenerator pipeline)

---

## 1. Overview

**Current pipeline problem:** SAM2 AutoMaskGenerator is class-agnostic. Stage 2 geometric filter cannot distinguish brick walls from cardboard boxes. The texture filter (`batch_segment_texture.ipynb`) is a heuristic workaround.

**GDINO + SAM2 top-down:**
```
Image → Grounding DINO (text: "cardboard box") → bboxes → SAM2 Predictor → masks
```

---

## 2. Model Size Comparison

| Metric | SAM2 Hiera-L | GDINO Swin-T | GDINO Swin-B |
|--------|-------------|-------------|-------------|
| Params | ~224M | ~99M | ~398M |
| Backbone | Hiera | Swin-T (224²) | Swin-B (384²) |
| Weight file | ~900MB | ~660MB | ~1.6GB |
| Text encoder | — | BERT-base (~440MB) | Same |
| **Inference VRAM** | 10-18GB | 1.5-2.5GB | 4-6GB |
| Per-image speed | 30-90s | 0.3-0.6s | 0.6-1.2s |

---

## 3. VRAM Budget

24GB GPU environment:

| Setup | Loaded | 1024² Inference |
|-------|--------|-----------------|
| GDINO Swin-T + SAM2 Pred | ~5.0 GB | ~7 GB ✅ |
| GDINO Swin-B + SAM2 Pred | ~7.5 GB | ~10 GB ✅ |
| SAM2 AutoMask + GDINO | ~20 GB | ~22 GB ⚠️ |

- AutoMaskGenerator not needed — top-down only requires Predictor
- Both models can load/release sequentially, or co-reside (5+3.5=8.5GB, well under 24GB)

---

## 4. Inference Speed

| Step | Swin-T | Swin-B |
|------|--------|--------|
| GDINO detection | 0.3-0.6s | 0.6-1.2s |
| SAM2 set_image | ~0.5s | ~0.5s |
| SAM2 per-box | ~0.05s | ~0.05s |
| **20 boxes total** | **~2.5s** | **~3.7s** |

vs current ~30s, **~10× speedup**.

---

## 5. Model Downloads

| File | Size | Source |
|------|------|--------|
| `groundingdino_swint_ogc.pth` | 660 MB | [GitHub Releases](https://github.com/IDEA-Research/GroundingDINO/releases) |
| BERT-base-uncased | 440 MB | HuggingFace |
| Config + source | ~5 MB | `pip install groundingdino-py` |

**Total ~1.1 GB**. Use `HF_ENDPOINT=https://hf-mirror.com` for China-based servers.

---

## 6. Integration Architecture

```
Input (RGB) → resize(1024) → GDINO Swin-T → bboxes → SAM2 Pred → morphology → RGBA
                                       ↑
                          "cardboard box. package."
```

- text_prompt: `"cardboard box. package. parcel. carton."`
- box_threshold: 0.30, text_threshold: 0.25

---

## 7. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Small boxes missed | Medium | Lower `box_threshold` to 0.25 |
| Windows CUDA build failure | Medium | `groundingdino-py` pure-Python path |
| BERT download slow | High | `HF_ENDPOINT=https://hf-mirror.com` |
| Odd-shaped boxes missed | Medium | Fine-tune or few-shot prompting |

---

## 8. Recommended Path

1. **Run texture filter first** (`batch_segment_texture.ipynb`) → zero additional dependencies
2. If texture filter insufficient → deploy GDINO Swin-T
3. Skip Swin-B — 4× parameters, marginal accuracy gain
