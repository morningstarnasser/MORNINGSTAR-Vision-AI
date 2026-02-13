# MORNINGSTAR Math-Training — Deployment Guide

**Cloud GPU Training → Production Deployment**

---

## 🎯 Deployment Übersicht

```
Local Dev → Cloud Training → GGUF Export → Ollama/Production
    ↓           ↓              ↓              ↓
  Code        GPU A100      quantized      Local/API
  Tests       3 epochs        model         serving
```

---

## 1️⃣ VORBEREITUNG (Lokal)

### Dataset vorbereiten (einmalig)
```bash
# Option A: Lokale Preparation (braucht Python 3.11)
pip install datasets tqdm
python prepare_math_dataset.py

# Option B: In Cloud vorbereiten (empfohlen)
# Dataset wird automatisch von setup_runpod.sh geladen
```

### Code zu Git pushen
```bash
git add -A
git commit -m "Ready for training"
git push origin main
```

---

## 2️⃣ CLOUD GPU SETUP (RunPod)

### RunPod GPU Pod starten
1. https://runpod.io → "Deploy"
2. **Template:** PyTorch 2.1 + CUDA 12.1
3. **GPU:** A100 80GB (empfohlen) oder A6000 48GB
4. **Storage:** 50 GB Persistent Volume
5. **SSH aktivieren**

### Setup Script ausführen
```bash
# SSH in Pod
ssh root@XXX.XXX.XXX.XXX -p XXXXX

# Clone Repo
git clone https://github.com/morningstarnasser/MORNINGSTAR-AI-MODEL.git
cd MORNINGSTAR-AI-MODEL/math-training

# Auto-Setup
bash cloud/setup_runpod.sh
```

**Was setup_runpod.sh macht:**
- ✓ Python 3.11 Environment
- ✓ Installiert: unsloth, datasets, transformers, peft, trl
- ✓ Downloaded Datasets (GSM8K, MATH, Orca-Math, MathInstruct)
- ✓ Erstellt data/ mit train.jsonl und val.jsonl
- ✓ Test-Run von train_math.py

---

## 3️⃣ TRAINING STARTEN

### Basis Training (3 Epochen)
```bash
python cloud/train_math.py \
    --dataset-dir data/ \
    --output-dir /workspace/output/math-qlora \
    --epochs 3 \
    --lr 2e-4 \
    --batch-size 4 \
    --gradient-accumulation 4
```

**Training Zeit:**
- A100 80GB: ~6-8 Stunden
- A6000 48GB: ~10-12 Stunden

**Was passiert:**
- Lädt Qwen2.5-Coder-14B-Instruct (4-bit)
- LoRA Training (r=64, alpha=128)
- Speichert Checkpoints alle 500 steps
- Erstellt LoRA Adapter + Merged Model

### Mit Weights & Biases Logging
```bash
export WANDB_API_KEY=your_key_here
python cloud/train_math.py --dataset-dir data/ --wandb
```

### Resume von Checkpoint
```bash
python cloud/train_math.py \
    --dataset-dir data/ \
    --resume-from /workspace/output/math-qlora/checkpoint-1500
```

---

## 4️⃣ EXPORT ZU GGUF

### Nach Training: Export
```bash
python cloud/export_gguf.py \
    --model-dir /workspace/output/math-qlora/merged-model \
    --output-dir /workspace/export \
    --quant q4_k_m
```

**Erstellt:**
- `morningstar-math-q4_k_m.gguf` (~9 GB)
- `morningstar-math-q8_0.gguf` (~16 GB, höhere Qualität)

### GGUF zurück zu lokalem PC kopieren
```bash
# Auf lokalem PC:
scp -P XXXXX root@XXX.XXX.XXX.XXX:/workspace/export/*.gguf D:/math-training/
```

---

## 5️⃣ OLLAMA DEPLOYMENT (Lokal)

### Model in Ollama importieren
```bash
cd D:\math-training

# Erstelle Ollama Model
ollama create morningstar-math -f cloud/Modelfile.math
```

### Testen
```bash
ollama run morningstar-math
>>> What is 2^10 * 3^7 / (2^6 * 3^4)?

>>> A rectangle has length 12 and width 5. What is its area?
```

---

## 6️⃣ PRODUCTION API DEPLOYMENT

### FastAPI Server (Lokal)
```bash
cd inference/
python math_server.py

# Server läuft auf http://localhost:8000
```

### API Nutzung
```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{"problem": "What is the sum of divisors of 24?"}'
```

### Docker Deployment
```dockerfile
FROM ollama/ollama:latest
COPY morningstar-math.gguf /models/
COPY Modelfile.math /Modelfile
RUN ollama create morningstar-math -f /Modelfile
EXPOSE 11434
CMD ["ollama", "serve"]
```

```bash
docker build -t morningstar-math .
docker run -p 11434:11434 morningstar-math
```

---

## 7️⃣ HUGGINGFACE HUB UPLOAD

### Upload zu HuggingFace
```bash
pip install huggingface_hub

python -c "
from huggingface_hub import HfApi, upload_file
api = HfApi(token='hf_your_token_here')

upload_file(
    path_or_fileobj='morningstar-math-q4_k_m.gguf',
    path_in_repo='morningstar-math-q4_k_m.gguf',
    repo_id='kurdman991/morningstar-math',
    repo_type='model'
)
"
```

### README.md für HuggingFace
```markdown
---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-14B-Instruct
tags:
  - math
  - reasoning
  - qlora
  - gguf
---

# Morningstar-Math 14B

Fine-tuned Qwen2.5-Coder-14B on 50k math problems.

## Performance
- Level 1-2 (Basic): 95%+
- Level 3-4 (Intermediate): 85-90%
- Level 5 (Competition): 70-80%
- Level 6-7 (AIME/Olympiade): 50-60%

## Usage
\`\`\`bash
ollama pull kurdman991/morningstar-math
ollama run kurdman991/morningstar-math
\`\`\`
```

---

## 8️⃣ EVALUATION & MONITORING

### Post-Deployment Evaluation
```bash
cd eval/

# Komplette Evaluation
python evaluate_math.py --model morningstar-math --all-levels --verbose

# Vergleich mit Baseline
python compare_models.py \
    --models morningstar-math qwen2.5-coder:14b \
    --all-levels
```

### Expected Results (nach Training)
```
Level 1: 95%+ (war: 88.9%)
Level 2: 95%+
Level 3: 85-90%
Level 4: 85-90%
Level 5: 70-80%
Level 6: 50-60% (AIME)
Level 7: 40-50% (Olympiade)

Overall: 75-80% (vor Training: 60-70%)
```

---

## 9️⃣ PRODUCTION CHECKLIST

### Vor Go-Live:
- [ ] Training abgeschlossen (3 Epochen)
- [ ] GGUF Export erfolgreich
- [ ] Ollama Model erstellt
- [ ] Evaluation durchgeführt (>75% overall)
- [ ] API Server getestet
- [ ] README.md aktualisiert
- [ ] HuggingFace Upload (optional)
- [ ] GitHub Release erstellt

### Post-Launch:
- [ ] Monitor API Latenz
- [ ] Sammle User Feedback
- [ ] Track Accuracy auf echten Problemen
- [ ] Überlege Fine-Tuning v2 mit User-Daten

---

## 🚀 Alternative: Cloud API Deployment

### Hugging Face Inference API
```bash
# Upload model zu HF
# Dann nutze Inference API:
curl https://api-inference.huggingface.co/models/kurdman991/morningstar-math \
  -H "Authorization: Bearer $HF_TOKEN" \
  -d '{"inputs":"What is 2+2?"}'
```

### Replicate.com
```bash
# Erstelle Replicate Model
cog predict -i prompt="Solve: x^2 + 5x + 6 = 0"
```

---

## 📊 Cost Estimation

### RunPod Training:
- **A100 80GB:** $1.99/hour × 8 hours = ~$16
- **A6000 48GB:** $0.79/hour × 12 hours = ~$9.50
- **Storage:** $0.10/GB/month × 50GB = $5/month

### Production Hosting:
- **Lokal (Ollama):** $0 (eigene Hardware)
- **HuggingFace Inference:** $0.06 per 1k chars
- **Replicate:** $0.000725 per second

---

## 🔧 Troubleshooting

**Training OOM:**
```bash
python cloud/train_math.py --batch-size 2 --gradient-accumulation 8
```

**GGUF Export scheitert:**
```bash
pip install llama-cpp-python --force-reinstall
```

**Ollama import Fehler:**
```bash
# Checke GGUF Integrität
file morningstar-math.gguf
```

---

**Deployment Support:** https://github.com/morningstarnasser/MORNINGSTAR-AI-MODEL/issues
