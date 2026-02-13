# Known Issues & Workarounds

## Python 3.14.3 Kompatibilität

**Problem:** Python 3.14.3 ist zu neu für viele ML-Libraries

**Betroffene Packages:**
- `numpy` - Keine kompatible Version verfügbar
- `datasets` - Benötigt numpy
- `pyarrow` - Build Fehler

**Workarounds:**

### Option 1: Downgrade zu Python 3.11 (EMPFOHLEN)
```bash
# Download Python 3.11.x von python.org
# Installiere und nutze venv
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate  # Windows

pip install datasets tqdm transformers torch
```

### Option 2: Conda Environment
```bash
conda create -n morningstar python=3.11
conda activate morningstar
pip install datasets tqdm transformers torch peft trl
```

### Option 3: Docker (BESTE für Training)
```bash
docker pull pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
docker run -it -v $(pwd):/workspace pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime bash
cd /workspace
pip install datasets tqdm peft trl unsloth
```

### Option 4: Cloud GPU (RunPod/Vast.ai)
```bash
# Nutze vorinstallierte ML-Images
# Python 3.11 + CUDA + alle Libraries bereits installiert
bash cloud/setup_runpod.sh
```

---

## Slow Download Speeds

**Problem:** Lokale Internet-Verbindung sehr langsam (150-300 KB/s)

**Impact:**
- Model Downloads: 8-15 Stunden für 9GB Modelle
- pip installs: Timeouts und Fehler
- Dataset Downloads: Sehr langsam

**Workarounds:**

### 1. Cloud Download + Local Transfer
```bash
# Auf Cloud Server (RunPod/Colab):
ollama pull qwen2.5-coder:14b
# Dann GGUF zurück auf lokalen PC kopieren
```

### 2. Direkt in der Cloud trainieren
```bash
# Alles in Cloud machen, nur fertige Modelle zurück laden
bash cloud/setup_runpod.sh
python cloud/train_math.py --dataset-dir /workspace/data
python cloud/export_gguf.py --model-dir /workspace/output/math-qlora/merged-model
```

### 3. HuggingFace Hub statt Ollama
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Direkt von HuggingFace laden (oft schneller)
model = AutoModelForCausalLM.from_pretrained("kurdman991/morningstar-14b")
tokenizer = AutoTokenizer.from_pretrained("kurdman991/morningstar-14b")
```

---

## Dataset Preparation Failed

**Problem:** `prepare_math_dataset.py` scheitert wegen fehlender Dependencies

**Quick Fix:**
```bash
# Nutze vorbereitete Datasets von HuggingFace
from datasets import load_dataset

gsm8k = load_dataset("gsm8k", "main")
math_dataset = load_dataset("hendrycks/competition_math")
```

**Alternative:** Download pre-processed dataset
```bash
# Wenn jemand anderes das Dataset bereits erstellt hat:
wget https://huggingface.co/datasets/kurdman991/morningstar-math-data/train.jsonl
wget https://huggingface.co/datasets/kurdman991/morningstar-math-data/val.jsonl
```

---

## Ollama Model Creation Failed

**Problem:** `ollama create` scheitert mit "base model not found"

**Fix:**
```bash
# Stelle sicher dass Base Model existiert
ollama pull qwen2.5-coder:14b

# Dann erstelle Custom Model
ollama create morningstar -f Modelfile.morningstar
```

**Alternative:** Nutze HuggingFace GGUF direkt
```bash
# Download GGUF von HuggingFace
curl -L -o morningstar-14b.gguf "https://huggingface.co/kurdman991/morningstar-14b/resolve/main/morningstar-14b.Q4_K_M.gguf"

# Importiere in Ollama
ollama create morningstar -f Modelfile.morningstar
```

---

## Training: CUDA Out of Memory

**Problem:** GPU RAM reicht nicht

**Lösungen:**

### Reduziere Batch Size
```bash
python train.py --batch_size 2 --gradient_accumulation 8
```

### Nutze kleineres Model
```bash
# Statt 14B nutze 7B
python train.py --base_model Qwen/Qwen2.5-Coder-7B-Instruct --batch_size 4
```

### Nutze Unsloth (2-3x weniger VRAM)
```bash
python cloud/train_math.py --batch_size 4
# Unsloth optimiert automatisch
```

### Reduziere Max Sequence Length
```bash
python train.py --max_seq_length 2048  # statt 4096
```

---

## Evaluation zu langsam

**Problem:** 63 Probleme dauern zu lange

**Quick Fixes:**

### Nur einzelne Levels testen
```bash
python eval/evaluate_math.py --levels 1 2  # Nur Basic
python eval/evaluate_math.py --levels 6 7  # Nur Hard
```

### Weniger Probleme pro Level
```python
# In evaluate_math.py:
# Ändere get_test_problems() um nur 3 pro Level zu nutzen
```

### Nutze schnelleres Model für Quick Tests
```bash
# Erst mit kleinem Model testen
python eval/evaluate_math.py --model qwen2.5:7b --levels 1
```

---

## Git Push Failed (Authentication)

**Problem:** HuggingFace/GitHub auth scheitert

**Fix für HuggingFace:**
```bash
git remote remove huggingface
git remote add huggingface https://USERNAME:TOKEN@huggingface.co/USERNAME/REPO
git push huggingface main
```

**Fix für GitHub:**
```bash
# Nutze Personal Access Token statt Passwort
git remote remove origin
git remote add origin https://USERNAME:TOKEN@github.com/USERNAME/REPO.git
git push origin main
```

---

## Allgemeine Tipps

1. **Nutze Cloud für Training** - Schneller und bessere GPU-Verfügbarkeit
2. **Python 3.11 statt 3.14** - Beste Kompatibilität
3. **Teste lokal mit kleinen Modellen** - Schnelles Feedback
4. **Trainiere in der Cloud** - Bessere Hardware & Internet
5. **Nutze vorbereitete Datasets** - Spart Zeit bei Setup

---

**Bei weiteren Problemen:** https://github.com/morningstarnasser/MORNINGSTAR-AI-MODEL/issues
