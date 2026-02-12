#!/bin/bash
# ==========================================================================
# RunPod GPU Instance Setup Script
# ==========================================================================
# One-click setup for fine-tuning Qwen2.5-Coder-14B with QLoRA + Unsloth
#
# Recommended hardware:
#   - A100 80GB  (best: full speed, fits 14B QLoRA comfortably)
#   - A6000 48GB (good: fits 14B QLoRA, slightly slower)
#   - A100 40GB  (ok: tight on memory, reduce batch size to 2)
#
# Usage:
#   bash setup_runpod.sh
#   bash setup_runpod.sh --skip-dataset
#
# Author: Ali Nasser
# ==========================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()   { echo -e "${GREEN}[SETUP]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }

WORKDIR="/workspace"
REPO_DIR="${WORKDIR}/math-training"
VENV_DIR="${WORKDIR}/venv"
DATASET_DIR="${REPO_DIR}/data"
OUTPUT_DIR="${WORKDIR}/output"

SKIP_DATASET=false
if [[ "${1:-}" == "--skip-dataset" ]]; then
    SKIP_DATASET=true
fi

echo ""
echo "=========================================================="
echo "  MORNINGSTAR-MATH Training Setup"
echo "  Fine-tuning Qwen2.5-Coder-14B with QLoRA + Unsloth"
echo "=========================================================="
echo ""

# ------------------------------------------------------------------
# 1. System Updates
# ------------------------------------------------------------------
log "Updating system packages..."
apt-get update -qq && apt-get install -y -qq git wget curl htop nvtop tmux jq > /dev/null 2>&1
log "System packages updated."

# ------------------------------------------------------------------
# 2. CUDA Verification
# ------------------------------------------------------------------
log "Verifying CUDA installation..."

if ! command -v nvidia-smi &> /dev/null; then
    error "nvidia-smi not found. Is this a GPU instance?"
fi

nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
echo ""

GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
GPU_MEM_GB=$((GPU_MEM_MB / 1024))
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)

log "GPU: ${GPU_NAME} (${GPU_MEM_GB} GB)"

if [[ $GPU_MEM_GB -lt 24 ]]; then
    error "GPU has only ${GPU_MEM_GB}GB VRAM. Need at least 24GB for 14B QLoRA."
elif [[ $GPU_MEM_GB -lt 40 ]]; then
    warn "GPU has ${GPU_MEM_GB}GB VRAM. Consider reducing batch size to 2."
elif [[ $GPU_MEM_GB -ge 80 ]]; then
    log "A100 80GB detected. Full speed ahead."
else
    log "${GPU_MEM_GB}GB VRAM -- sufficient for QLoRA training."
fi

if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | awk '{print $6}' | tr -d ',')
    log "CUDA toolkit: ${CUDA_VER}"
else
    warn "nvcc not found. PyTorch should still work with bundled CUDA."
fi

# ------------------------------------------------------------------
# 3. Python Virtual Environment
# ------------------------------------------------------------------
log "Setting up Python virtual environment..."

PYTHON_BIN=$(command -v python3.10 || command -v python3.11 || command -v python3)
PY_VER=$(${PYTHON_BIN} --version 2>&1)
log "Using ${PY_VER}"

${PYTHON_BIN} -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip setuptools wheel -q
log "Virtual environment ready: ${VENV_DIR}"

# ------------------------------------------------------------------
# 4. Project Setup
# ------------------------------------------------------------------
if [[ -d "${REPO_DIR}" ]]; then
    log "Project directory already exists at ${REPO_DIR}"
else
    log "Creating project structure..."
    mkdir -p "${REPO_DIR}/cloud" "${REPO_DIR}/data"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "${SCRIPT_DIR}/train_math.py" ]]; then
        cp "${SCRIPT_DIR}"/*.py "${REPO_DIR}/cloud/" 2>/dev/null || true
        cp "${SCRIPT_DIR}/requirements.txt" "${REPO_DIR}/cloud/" 2>/dev/null || true
        cp "${SCRIPT_DIR}/Modelfile.math" "${REPO_DIR}/cloud/" 2>/dev/null || true
        log "Copied training scripts to ${REPO_DIR}/cloud/"
    fi
fi

# ------------------------------------------------------------------
# 5. Install Dependencies
# ------------------------------------------------------------------
log "Installing Python dependencies..."

REQ_FILE="${REPO_DIR}/cloud/requirements.txt"
if [[ -f "${REQ_FILE}" ]]; then
    pip install -r "${REQ_FILE}" -q
else
    warn "requirements.txt not found. Installing core deps manually..."
    pip install "torch>=2.1.0" -q
    pip install "transformers>=4.36.0" "datasets>=2.16.0" "accelerate>=0.25.0" -q
    pip install "peft>=0.7.0" "trl>=0.7.4" "bitsandbytes>=0.41.0" -q
    pip install wandb tqdm sentencepiece protobuf scipy scikit-learn tensorboard -q
fi

log "Installing Unsloth (latest from GitHub)..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q
pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes -q

log "Installing flash-attention (may take 5-10 minutes)..."
pip install flash-attn --no-build-isolation -q 2>/dev/null || \
    warn "flash-attn install failed. Training will still work."

log "All dependencies installed."

# ------------------------------------------------------------------
# 6. Wandb Setup (Optional)
# ------------------------------------------------------------------
info "Weights & Biases logging is optional."
read -r -t 10 -p "Enable W&B logging? [y/N] " WANDB_CHOICE || WANDB_CHOICE="n"
if [[ "${WANDB_CHOICE,,}" == "y" ]]; then
    wandb login
    WANDB_FLAG="--wandb"
else
    WANDB_FLAG=""
    log "W&B disabled. Using TensorBoard."
fi

# ------------------------------------------------------------------
# 7. Dataset Check
# ------------------------------------------------------------------
if [[ "${SKIP_DATASET}" == false ]]; then
    log "Checking dataset..."
    JSONL_COUNT=$(find "${DATASET_DIR}" -name "*.jsonl" 2>/dev/null | wc -l | tr -d ' ')
    if [[ $JSONL_COUNT -gt 0 ]]; then
        for f in "${DATASET_DIR}"/*.jsonl; do
            COUNT=$(wc -l < "$f" | tr -d ' ')
            info "  $(basename "$f"): ${COUNT} examples"
        done
    else
        warn "No JSONL files in ${DATASET_DIR}"
        warn "Upload dataset or run prepare_math_dataset.py first."
    fi
fi

# ------------------------------------------------------------------
# 8. Output Directory
# ------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"

# ------------------------------------------------------------------
# 9. Summary
# ------------------------------------------------------------------
echo ""
echo "=========================================================="
echo -e "  ${GREEN}Setup Complete!${NC}"
echo "=========================================================="
echo ""
echo "  GPU:     ${GPU_NAME} (${GPU_MEM_GB} GB)"
echo "  Python:  ${PY_VER}"
echo "  Venv:    ${VENV_DIR}"
echo "  Dataset: ${DATASET_DIR}"
echo "  Output:  ${OUTPUT_DIR}"
echo ""
echo "  NEXT STEPS:"
echo "  1. source ${VENV_DIR}/bin/activate"
echo "  2. Upload dataset: scp data/*.jsonl runpod:${DATASET_DIR}/"
echo "  3. Train: cd ${REPO_DIR}/cloud && python train_math.py --dataset-dir ${DATASET_DIR} --output-dir ${OUTPUT_DIR}/math-qlora ${WANDB_FLAG}"
echo "  4. Export: python export_gguf.py --model-dir ${OUTPUT_DIR}/math-qlora/merged-model"
echo "  5. Monitor: watch -n1 nvidia-smi"
echo ""
echo "=========================================================="
