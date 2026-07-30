# 🏭 GDSense-SAM2

**Physical AI Defect Detection — Two-Stage Cascade Pipeline**

```
Edge (PatchCore)  ──→  Cloud (Grounding DINO + SAM 2)
  Fast screening         Zero-shot box detection + precise segmentation
```

> 🧪 SAM 2 validated on Vast.ai RTX 3090 with base+ checkpoint.
> 🎯 Grounding DINO Swin-T top-down pipeline validated — ~10× faster, semantically-aware.

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

# 4. Grounding DINO (optional — for top-down pipeline)
pip install groundingdino-py
wget -P /workspace/checkpoints \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

---

## 🔬 Box Segmentation — Two Approaches

### Approach A: Grounding DINO + SAM2 ⚡ (Recommended)

Top-down zero-shot detection → precise mask refinement. **~10× faster** than AutoMaskGenerator, and semantically-aware (no brick-wall false positives).

**Notebook:** `notebooks/batch_segment_gdino.ipynb`

| Step | Engine | What it does | Time |
|------|--------|-------------|------|
| 1 | Grounding DINO (Swin-T) | Zero-shot box detection via text prompt | ~0.5s |
| 2 | Boundary filter | Skip boxes touching frame edges | — |
| 3 | SAM2 Predictor | Bbox-prompt mask refinement | ~0.05s/box |
| 4 | Morphological clean-up → RGBA | Open/close + alpha channel | — |

**Key flags:** `TEXT_PROMPT`, `BOX_THRESHOLD`, `TEXT_THRESHOLD`

**Why Grounding DINO?**
- AutoMaskGenerator blind-scans the whole image (144+ point grid per crop) → ~30s
- GDINO does one Swin-T forward to locate all boxes by text, then SAM2 only runs mask decode
- Net: ~2.5s/image vs ~30s; semantic prompt filters out brick walls, floor, etc.

### Approach B: AutoMaskGenerator + SAM2

Class-agnostic bottom-up baseline — useful when text prompts don't apply or as comparison point.

**Notebook:** `notebooks/batch_segment_improved.ipynb`

| Stage | Purpose | Control |
|-------|---------|---------|
| 1 | AutoMaskGenerator candidate masks | `POINTS_PER_SIDE`, thresholds |
| 2 | Geometric filter (rectangularity + fully-in-frame) | `MIN_RECTANGULARITY`, `BOUNDARY_MARGIN` |
| 2.5 | **Texture filter** (edge density + local variance) | `ENABLE_TEXTURE_FILTER` |
| 3 | Predictor bbox-prompt refinement | `REFINE_WITH_PREDICTOR` |
| 4 | Morphological clean-up → save RGBA | `MORPH_OPEN_RADIUS`, `MORPH_CLOSE_RADIUS` |

---

## 🧠 Model Variants

| Model | VRAM | Params | Speed (per image) | Use Case |
|-------|------|--------|-------------------|----------|
| **GDINO Swin-T** | 1.5-2.5 GB | ~99M | ~0.5s | Zero-shot text-to-box detection |
| **GDINO Swin-B** | 4-6 GB | ~398M | ~1s | Higher detection accuracy |
| **SAM2 Hiera-T** | ~6 GB | ~48M | ~0.03s/box | Rapid prototyping |
| **SAM2 Hiera-S** | ~8 GB | ~77M | ~0.04s/box | Lightweight online |
| **SAM2 Hiera-B+** ⭐ | ~12 GB | ~81M | ~0.05s/box | **Recommended** — best balance |
| **SAM2 Hiera-L** | ~18 GB | ~224M | ~0.08s/box | Offline high-precision |

> **Combined VRAM:** GDINO Swin-T + SAM2 Hiera-B+ ≈ 8.5 GB (co-resident on 24 GB GPU with room to spare)

---

## 💻 Recommended Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | 8 GB VRAM (SAM2 small) | 24 GB VRAM (GDINO + SAM2 B+ co-resident) |
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
│   ├── batch_segment_gdino.ipynb    # ⚡ GDINO + SAM2 top-down (recommended)
│   ├── batch_segment_improved.ipynb # AutoMaskGenerator baseline (5-stage, toggle-driven)
│   └── README.md
├── docs/
│   └── grounding_dino_feasibility.md # GDINO integration feasibility analysis
├── input/                           # Sample images (multi_box2/3/4)
├── output/                          # Segmentation results (gitignored — reproducible)
└── sam2/ → git submodule (planned)
```

---

## 🐛 Known Issues

| Issue | Root Cause | Status |
|-------|------------|--------|
| multi_box3 CUDA OOM | 5957×3971 raw image exceeds VRAM | Fixed with `MAX_IMAGE_SIZE=1024` downscale |
| multi_box4 brick wall false positives | AutoMaskGenerator is class-agnostic | ✅ Solved by GDINO text-prompt (`"cardboard box"`) |

---

## 🗺️ Roadmap

- [ ] Validate GDINO on more box types (tilted, stacked, irregular)
- [ ] Fine-tune GDINO if zero-shot misses edge cases
- [ ] PatchCore edge-side integration
- [ ] End-to-end two-stage pipeline demo

---

## 📝 Status

- [x] SAM 2 environment set up (2026-07-29)
- [x] AutoMaskGenerator baseline
- [x] 5-stage improved pipeline — multi_box2/3/4 results (2026-07-31)
- [x] Grounding DINO + SAM2 top-down pipeline — validated on RTX 3090 (2026-07-31)
- [x] GDINO integration feasibility analysis
- [ ] Validate GDINO on more box types + edge cases
- [ ] PatchCore edge-side integration
- [ ] End-to-end two-stage pipeline demo

---

## 📄 License

MIT
