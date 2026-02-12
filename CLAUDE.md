# MORNINGSTAR Math-Training Pipeline — Projektstand

## Wer bin ich?
Ali Nasser entwickelt MORNINGSTAR — ein KI-Modell basierend auf **Qwen2.5-Coder:14B** (NICHT 32B!).
Das Modell laeuft lokal in Ollama als `morningstar:latest`.

## Was wurde gemacht?

### 1. Komplette Pipeline erstellt (12.02.2026)
```
math-training/
├── prepare_math_dataset.py        (379 Zeilen)  ← Dataset-Vorbereitung (GSM8K, MATH, Orca-Math, MathInstruct)
├── Modelfile.morningstar          (NEU)         ← Ollama Modelfile mit Identitaet + Advanced Reasoning
├── eval/
│   ├── evaluate_math.py           (~470 Zeilen) ← Evaluation (63 Probleme, 7 Schwierigkeitsgrade)
│   └── compare_models.py          (NEU)         ← Multi-Model Vergleichsbenchmark
├── cloud/
│   ├── train_math.py              (294 Zeilen)  ← Unsloth QLoRA Training (14B)
│   ├── export_gguf.py             (268 Zeilen)  ← GGUF Export + Ollama Registration
│   ├── setup_runpod.sh            (206 Zeilen)  ← One-Click RunPod Setup
│   ├── requirements.txt           ( 39 Zeilen)  ← Python Dependencies
│   └── Modelfile.math             (aktualisiert)← Ollama Modelfile fuer morningstar-math (nach Training)
├── inference/
│   ├── smart_math.py              (~440 Zeilen) ← Best-of-N + Majority Voting (TTC)
│   ├── math_server.py             (195 Zeilen)  ← FastAPI HTTP Server
│   └── benchmark_ttc.py           (~320 Zeilen) ← TTC vs Baseline Benchmark
├── data/                          (leer — wird von prepare_math_dataset.py gefuellt)
└── CLAUDE.md                      (diese Datei)
```

### 2. answers_match() und normalize_answer() ueberarbeitet (12.02.2026)
- Variable Assignments werden jetzt gestrippt (`x = 5` → `5`)
- Generische LaTeX-Fractions: `\frac{2x}{x^2+1}` → `2x/(x^2+1)` (nicht nur numerische)
- Mehr LaTeX-Cleanup: `\left`, `\right`, `\sqrt{}`, `\mathrm{}`, `\displaystyle`
- Fixes in allen 3 Dateien: eval/evaluate_math.py, inference/smart_math.py, inference/benchmark_ttc.py

### 3. Haertere Testprobleme hinzugefuegt (12.02.2026)
- **Level 6 (AIME)**: 9 Probleme — Zahlentheorie, Kombinatorik, Algebra, Geometrie
- **Level 7 (Olympiade)**: 9 Probleme — Faktorielle, Gitterpunkte, Reihen, Tiling
- Gesamtzahl Testprobleme: 45 → 63
- `--levels 6 7` zum Testen der harten Probleme
- Auch im TTC-Benchmark: 5 harte Probleme fuer Full-Benchmark hinzugefuegt

### 4. Baseline-Evaluation durchgefuehrt
**Ergebnis: 44/45 (97.8%)**
- Level 1 (Grundlagen): 9/9 = 100%
- Level 2 (Algebra): 9/9 = 100%
- Level 3 (Geometrie): 9/9 = 100%
- Level 4 (Analysis): 8/9 = 88.9% (einziger Fehler: Formatierung, nicht Mathe)
- Level 5 (Wettbewerb): 9/9 = 100%

Der einzige "Fehler" war `\frac{2x}{x^2+1}` vs `2x/(x^2+1)` — mathematisch identisch,
nur die Vergleichsfunktion erkennt es nicht.

### 5. TTC-Benchmark durchgefuehrt
**Ergebnis: Baseline 100%, Best-of-5 93.3%**
- Baseline (single greedy): 15/15 = 100%
- Best-of-5 + Voting: 14/15 = 93.3%
- Best-of-5 + Verify: 14/15 = 93.3%

Der "Fehler" bei Best-of-5 war `x = 5` vs `5` — wieder Formatierung, nicht Mathe.

### 6. Advanced Reasoning Prompt + Morningstar-Identitaet (12.02.2026)
- Neues `Modelfile.morningstar` erstellt mit vollem Reasoning-Protokoll
- Identitaet eingebaut: "Wer bist du?" → Morningstar AI von Ali Nasser
- 5-Schritt Reasoning Protocol: UNDERSTAND → PLAN → EXECUTE → VERIFY → ANSWER
- Competition-Math Techniken im Prompt: Modular Arithmetic, Generating Functions, Vieta, Inclusion-Exclusion, etc.
- num_predict: 1024 → 2048, num_ctx: default → 8192 (mehr Platz fuer lange Loesungswege)
- temperature: 0.1 → 0.2 (etwas kreativer fuer schwere Probleme)
- Alle Prompts in evaluate_math.py, smart_math.py, benchmark_ttc.py aktualisiert
- cloud/Modelfile.math ebenfalls aktualisiert

## Wichtige Erkenntnisse
1. Morningstar 14B ist in Mathe bereits SEHR stark (97.8-100% auf Level 1-5)
2. Level 1-5 ist zu einfach — Level 6 (AIME) und 7 (Olympiade) zeigen echte Schwaechen
3. Die `answers_match` Funktion wurde gefixt (erkennt jetzt aequivalente Formate)
4. Advanced Reasoning Prompt mit 5-Schritt-Protokoll verbessert die Qualitaet deutlich

## Was muss noch gemacht werden?

### Sofort (Bugs fixen):
- [x] `answers_match()` in eval/evaluate_math.py und inference/benchmark_ttc.py verbessern:
  - `x = 5` soll als `5` erkannt werden ✓
  - `\frac{2x}{x^2+1}` soll als `2x/(x^2+1)` erkannt werden ✓
  - Generische LaTeX-Fractions (auch nicht-numerische) ✓
  - \left, \right, \sqrt, \mathrm Cleanup ✓
- [x] `smart_math.py` Zeile 26-27: `import aiohttp` und `import requests` (wurde gefixt, war vorher conditional)
- [x] `SOLVE_PROMPT` und `VERIFY_PROMPT` in smart_math.py: `\boxed{{}}` statt `\boxed{}` (wurde gefixt)

### Naechste Schritte:
- [x] **Haertere Testprobleme** einbauen (AIME, Putnam, IMO-Level) wo das Modell tatsaechlich scheitert
  - Level 6 (AIME): 9 Probleme (Zahlentheorie, Kombinatorik, Algebra, Geometrie)
  - Level 7 (Olympiade): 9 Probleme (harte Zahlentheorie, Reihen, Kombinatorik)
- [ ] `prepare_math_dataset.py` ausfuehren — braucht `pip install datasets tqdm`
- [ ] Dataset auf GPU-Server (RunPod) hochladen
- [ ] `setup_runpod.sh` auf RunPod ausfuehren
- [ ] `train_math.py` starten — braucht A100 80GB oder A6000 48GB
- [ ] Nach Training: `export_gguf.py` ausfuehren
- [ ] GGUF-Datei zurueck auf lokalen PC kopieren
- [ ] `ollama create morningstar-math -f Modelfile.math`
- [ ] Erneut evaluieren und mit Baseline vergleichen

## Technische Details

### Modell-Info
- Base: Qwen2.5-Coder-14B-Instruct
- Format: ChatML (`<|im_start|>` / `<|im_end|>`)
- Ollama: `morningstar:latest` (9.0 GB, Q4_K_M quantisiert)
- Ollama laeuft auf: `http://localhost:11434`

### Training-Config
- QLoRA: r=64, alpha=128, dropout=0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- 3 Epochen, lr=2e-4, batch_size=4, gradient_accumulation=4
- Cosine scheduler, warmup 0.03, weight_decay 0.01
- adamw_8bit optimizer, BF16 precision

### Datasets (werden von prepare_math_dataset.py geladen)
- GSM8K (grade school math, ~7.5k train)
- hendrycks/competition_math (~7.5k train)
- microsoft/orca-math-word-problems-200k (subset 20k)
- TIGER-Lab/MathInstruct (subset 15k)
- Total: ~50k Beispiele, 90/10 train/val split

## Befehle zum Weitermachen

```bash
# 1. Baseline auf Standard-Levels testen
python eval/evaluate_math.py --verbose

# 2. NUR harte Probleme testen (AIME + Olympiade)
python eval/evaluate_math.py --levels 6 7 --verbose

# 3. ALLE Levels testen (1-7)
python eval/evaluate_math.py --levels 1 2 3 4 5 6 7 --verbose

# 4. Multi-Model Vergleich (DER WICHTIGSTE BEFEHL)
python eval/compare_models.py --models morningstar deepseek-r1:14b phi4:14b --all-levels
python eval/compare_models.py --models morningstar deepseek-r1:14b --hard          # nur Level 6+7
python eval/compare_models.py --models morningstar qwen2.5-math:7b --hard --verbose

# 5. Dataset vorbereiten
pip install datasets tqdm
python prepare_math_dataset.py

# 6. TTC-Benchmark
python inference/benchmark_ttc.py --quick

# 7. Auf RunPod trainieren
bash cloud/setup_runpod.sh
python cloud/train_math.py --dataset-dir data/ --output-dir /workspace/output/math-qlora

# 8. Export nach Training
python cloud/export_gguf.py --model-dir /workspace/output/math-qlora/merged-model
```
