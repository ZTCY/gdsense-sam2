# Notebooks

## `batch_segment_improved.ipynb`

5-stage pipeline — class-agnostic AutoMaskGenerator → geometry → texture → predictor → morphology.

| Flag | Behavior |
|------|----------|
| `ENABLE_TEXTURE_FILTER = True` | Stage 2.5 texture on (discard brick walls / flat surfaces) |
| `ENABLE_TEXTURE_FILTER = False` | 4-stage pipeline (geometry only) |
| `REFINE_WITH_PREDICTOR = False` | Skip SAM2 refinement (fast draft) |

## `batch_segment_gdino.ipynb`

Grounding DINO Swin-T + SAM2 top-down pipeline. Text-prompted detection replaces AutoMaskGenerator → faster and semantically meaningful.

| Flag | Behavior |
|------|----------|
| `TEXT_PROMPT` | What objects to detect (e.g. `"cardboard box. package."`) |
| `BOX_THRESHOLD` | Detection confidence floor |
| `TEXT_THRESHOLD` | Text-phrase match floor |

### One-time setup (Section 0)
```bash
pip install groundingdino-py
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# BERT tokenizer auto-downloaded on first run
```

## To sync to Vast.ai

```powershell
scp notebooks/*.ipynb root@<ip>:/workspace/gdsense-sam2/notebooks/
scp input/*.png input/*.jpg root@<ip>:/workspace/gdsense-sam2/input/
```
