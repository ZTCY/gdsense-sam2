# Notebooks

## `batch_segment_improved.ipynb`

5-stage SAM2 box segmentation pipeline. One notebook, all versions controlled by flags:

| Flag | Behavior |
|------|----------|
| `ENABLE_TEXTURE_FILTER = True` | Stage 2.5 texture filter on (discards brick walls) |
| `ENABLE_TEXTURE_FILTER = False` | 4-stage pipeline (geometry filter only) |
| `REFINE_WITH_PREDICTOR = False` | Skip SAM2 predictor refinement (fast draft mode) |

See git history (`git log -- notebooks/batch_segment_improved.ipynb`) for version evolution.

## To sync to Vast.ai

```powershell
scp notebooks/batch_segment_improved.ipynb root@<ip>:/workspace/gdsense-sam2/notebooks/
scp input/*.png input/*.jpg root@<ip>:/workspace/gdsense-sam2/input/
```
