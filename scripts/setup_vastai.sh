#!/usr/bin/env bash
# ============================================================
# GDSense-SAM2 — One-click environment bootstrap for Vast.ai
# Run this on any fresh GPU instance to get everything ready.
# ============================================================
set -euo pipefail

echo "========================================"
echo " GDSense-SAM2 — Environment Setup"
echo "========================================"

# ── 1. System libraries ──
echo ">>> [1/4] Installing system libs (libGL + libxcb) ..."
apt-get update -qq
apt-get install -y -qq libgl1-mesa-glx libxcb1

# ── 2. Clone SAM 2 upstream ──
echo ">>> [2/4] Cloning SAM 2 ..."
if [ ! -d "sam2" ]; then
    git clone https://github.com/facebookresearch/sam2.git
fi

# ── 3. pip install SAM 2 ──
echo ">>> [3/4] pip install SAM 2 dependencies ..."
cd sam2
pip install -e . -q
cd ..

# ── 4. Download base+ checkpoint to /tmp (keep workspace clean) ──
echo ">>> [4/4] Downloading SAM 2 base+ checkpoint ..."
pip install -q huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    'facebook/sam2-hiera-large',
    'sam2_hiera_base_plus.pt',
    local_dir='/tmp/sam2_weights'
)
print(f'✅ Checkpoint downloaded: {path}')
"

# ── Verify ──
echo ""
echo "========================================"
echo " Setup complete! Verifying..."
echo "========================================"
python3 -c "
import sam2
print('✅ import sam2 OK')
"

echo ""
echo "🎉 All done! Ready to run notebooks."
echo "   cd sam2/notebooks && jupyter lab"
