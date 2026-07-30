# 🏭 GDSense-SAM2

**Physical AI Defect Detection — Two-Stage Cascade Pipeline**

```
Edge (PatchCore)  ──→  Cloud (SAM 2 / ViT)
  Fast screening         Precise localization + segmentation
```

> 🧪 SAM 2 validated on Vast.ai RTX 3090 with base+ checkpoint.

---

## 🚀 Quick Start

On any fresh Vast.ai (or other GPU) instance, just run:

```bash
bash scripts/setup_vastai.sh
```

Or manually:

```bash
git clone https://github.com/ZTCY/gdsense-sam2.git
cd gdsense-sam2

# 1. SAM 2 upstream
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e . && cd ..

# 2. System dependencies (OpenCV headless)
apt-get install -y libgl1-mesa-glx libxcb1

# 3. Download checkpoint (base+ recommended)
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/sam2-hiera-large', 'sam2_hiera_base_plus.pt', local_dir='/tmp')"
```

---

## 🧠 SAM 2 Model Variants

| Variant | VRAM | Params | Use Case |
|---------|------|--------|----------|
| **tiny** | ~6 GB | 48M | Rapid prototyping, low-resource dev |
| **small** | ~8 GB | 77M | Lightweight online inference |
| **base+** ⭐ | ~12 GB | 81M | **Recommended** — best accuracy/speed balance |
| **large** | ~18 GB | 224M | Offline high-precision batch processing |

> **Tip:** Use `base+` for development; scale up/down based on deployment GPU budget.

---

## 💻 Recommended Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | 8 GB VRAM (small) | 24 GB VRAM (base+/large) |
| **CUDA** | 11.8 | 12.4+ |
| **RAM** | 16 GB | 32 GB+ |
| **Disk** | 20 GB | 50 GB+ (with checkpoints) |
| **Python** | 3.10 | 3.12 |
| **PyTorch** | 2.0 | 2.6 |

---

## 📁 Project Structure

```
gdsense-sam2/
├── README.md
├── .gitignore
├── scripts/
│   └── setup_vastai.sh              # One-click environment bootstrap
├── notebooks/
│   ├── batch_segment_improved.ipynb # 5-stage pipeline (toggle-driven)
│   └── README.md
├── input/                           # Sample images (multi_box2/3/4)
├── output/                          # Segmentation results
│   ├── multi_box2/                  # 22 boxes
│   ├── multi_box3/                  # 12 boxes
│   └── multi_box4/                  # 34 boxes
└── sam2/ → git submodule (planned)
```

---

## 🔬 Box Segmentation Pipeline

Single notebook (`notebooks/batch_segment_improved.ipynb`), 5 configurable stages:

| Stage | Purpose | Control |
|-------|---------|---------|
| 1 | AutoMaskGenerator candidate masks | `POINTS_PER_SIDE`, thresholds |
| 2 | Geometric filter (rectangularity + fully-in-frame) | `MIN_RECTANGULARITY`, `BOUNDARY_MARGIN` |
| 2.5 | **Texture filter** (edge density + local variance) | `ENABLE_TEXTURE_FILTER` |
| 3 | Predictor bbox-prompt refinement | `REFINE_WITH_PREDICTOR` |
| 4 | Morphological clean-up → save RGBA | `MORPH_OPEN_RADIUS`, `MORPH_CLOSE_RADIUS` |

### Known Issues

| Issue | Root Cause | Status |
|-------|------------|--------|
| multi_box3 CUDA OOM | 5957×3971 raw image exceeds VRAM | Fixed with `MAX_IMAGE_SIZE=1024` downscale |
| multi_box4 brick wall false positives | Geometry filter cannot distinguish brick walls from boxes | Texture filter variant pending test |

### Next Steps

- [ ] Run texture filter on multi_box4 to validate brick wall suppression
- [ ] If texture filter insufficient → evaluate Grounding DINO as top-down alternative
- [ ] PatchCore edge-side integration
- [ ] End-to-end two-stage pipeline demo

---

## 📝 Status

- [x] SAM 2 environment set up (2026-07-29)
- [x] Automatic mask generator baseline
- [x] 5-stage improved pipeline — multi_box2/3/4 results (2026-07-31)
- [x] Texture filter notebook + Grounding DINO feasibility analysis
- [ ] Validate texture filter on multi_box4
- [ ] PatchCore edge-side integration
- [ ] End-to-end two-stage pipeline demo

---

## 📄 License

MIT
