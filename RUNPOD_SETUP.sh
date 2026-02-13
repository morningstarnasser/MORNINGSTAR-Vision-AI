#!/bin/bash
# MORNINGSTAR Math Training - Automated RunPod Setup
# Author: Ali Nasser
# Führe dieses Script auf RunPod aus für komplettes Setup

set -e  # Stop on error

echo "=========================================="
echo "  MORNINGSTAR RunPod Setup"
echo "=========================================="

# 1. System Info
echo ""
echo "[1/8] System Check..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 --version

# 2. Update System
echo ""
echo "[2/8] System Update..."
apt-get update -qq
apt-get install -y git wget curl -qq

# 3. Python Environment
echo ""
echo "[3/8] Creating Python 3.11 Environment..."
# Check if Python 3.11 is available, otherwise use system Python
if command -v python3.11 &> /dev/null; then
    python3.11 -m venv /workspace/venv
else
    python3 -m venv /workspace/venv
fi
source /workspace/venv/bin/activate
pip install --upgrade pip -q

# 4. Install Dependencies
echo ""
echo "[4/8] Installing ML Libraries..."
pip install -q \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    transformers datasets accelerate peft trl bitsandbytes \
    scipy sentencepiece protobuf \
    tqdm wandb huggingface-hub

# 5. Install Unsloth (optional but recommended)
echo ""
echo "[5/8] Installing Unsloth (faster training)..."
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || echo "Unsloth install failed (optional)"

# 6. Clone Repository
echo ""
echo "[6/8] Cloning MORNINGSTAR Repository..."
cd /workspace
if [ -d "MORNINGSTAR-AI-MODEL" ]; then
    echo "Repository already exists, pulling latest..."
    cd MORNINGSTAR-AI-MODEL/math-training
    git pull
else
    git clone https://github.com/morningstarnasser/MORNINGSTAR-AI-MODEL.git
    cd MORNINGSTAR-AI-MODEL/math-training
fi

# 7. Prepare Dataset
echo ""
echo "[7/8] Downloading & Preparing Datasets..."
mkdir -p data

# Create dataset preparation script that doesn't need manual intervention
python3 << 'PYTHON_SCRIPT'
import json
import os
from datasets import load_dataset
from tqdm import tqdm

print("Loading datasets...")

# GSM8K
try:
    gsm8k = load_dataset("gsm8k", "main", split="train")
    print(f"✓ GSM8K: {len(gsm8k)} examples")
except Exception as e:
    print(f"✗ GSM8K failed: {e}")
    gsm8k = []

# MATH
try:
    math_ds = load_dataset("hendrycks/competition_math", split="train")
    print(f"✓ MATH: {len(math_ds)} examples")
except Exception as e:
    print(f"✗ MATH failed: {e}")
    math_ds = []

# Convert to training format
def format_example(ex):
    if "question" in ex and "answer" in ex:
        # GSM8K format
        return {
            "conversations": [
                {"role": "system", "content": "You are a math expert. Solve problems step by step."},
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["answer"]}
            ]
        }
    elif "problem" in ex and "solution" in ex:
        # MATH format
        return {
            "conversations": [
                {"role": "system", "content": "You are a math expert. Solve problems step by step."},
                {"role": "user", "content": ex["problem"]},
                {"role": "assistant", "content": ex["solution"]}
            ]
        }
    return None

print("\nConverting to training format...")
train_data = []

for ex in tqdm(gsm8k, desc="GSM8K"):
    formatted = format_example(ex)
    if formatted:
        train_data.append(formatted)

for ex in tqdm(math_ds[:10000], desc="MATH"):  # Limit to 10k
    formatted = format_example(ex)
    if formatted:
        train_data.append(formatted)

# Split train/val
split_idx = int(len(train_data) * 0.95)
train = train_data[:split_idx]
val = train_data[split_idx:]

print(f"\nTotal: {len(train_data)} examples")
print(f"Train: {len(train)} | Val: {len(val)}")

# Save
os.makedirs("data", exist_ok=True)
with open("data/train.jsonl", "w") as f:
    for item in train:
        f.write(json.dumps(item) + "\n")

with open("data/val.jsonl", "w") as f:
    for item in val:
        f.write(json.dumps(item) + "\n")

print(f"\n✓ Saved to data/train.jsonl and data/val.jsonl")
PYTHON_SCRIPT

# 8. Test Setup
echo ""
echo "[8/8] Testing Setup..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "=========================================="
echo "  ✓ SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Dataset ready:"
ls -lh data/*.jsonl
echo ""
echo "Next step:"
echo "  python cloud/train_math.py --dataset-dir data/"
echo ""
echo "Or with Weights & Biases logging:"
echo "  export WANDB_API_KEY=your_key"
echo "  python cloud/train_math.py --dataset-dir data/ --wandb"
echo ""
