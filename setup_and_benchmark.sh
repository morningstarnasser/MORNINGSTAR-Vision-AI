#!/bin/bash
# ============================================================================
# MORNINGSTAR — Auto Setup & Benchmark
# Wartet auf Downloads, erstellt Modelle, lauft Benchmarks
# Developed by: Ali Nasser (github.com/morningstarnasser)
# ============================================================================
set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

cd "$(dirname "$0")"

echo ""
echo -e "${CYAN}${BOLD}  MORNINGSTAR — Auto Setup & Benchmark${NC}"
echo ""

# ─── Step 1: Wait for LLaVA 13B download ─────────────────
echo -e "${YELLOW}[1/6] Warte auf LLaVA 13B Download...${NC}"
while ! ollama list 2>/dev/null | grep -q "llava:13b"; do
  sleep 10
  echo -n "."
done
echo -e " ${GREEN}Done!${NC}"

# ─── Step 2: Create morningstar-vision ────────────────────
echo -e "${YELLOW}[2/6] Erstelle morningstar-vision...${NC}"
ollama create morningstar-vision -f Modelfile.vision
echo -e "  ${GREEN}morningstar-vision erstellt!${NC}"

# ─── Step 3: Run benchmark on morningstar (14B) ──────────
echo -e "${YELLOW}[3/6] Benchmark: morningstar (14B)...${NC}"
bash benchmark.sh morningstar 2>&1 | tee /tmp/benchmark-14b.txt
echo ""

# ─── Step 4: Wait for 32B download ───────────────────────
echo -e "${YELLOW}[4/6] Warte auf Qwen2.5-Coder 32B Download...${NC}"
while ! ollama list 2>/dev/null | grep -q "qwen2.5-coder:32b"; do
  sleep 30
  echo -n "."
done
echo -e " ${GREEN}Done!${NC}"

# ─── Step 5: Create morningstar-32b ──────────────────────
echo -e "${YELLOW}[5/6] Erstelle morningstar-32b...${NC}"
ollama create morningstar-32b -f Modelfile.32b
echo -e "  ${GREEN}morningstar-32b erstellt!${NC}"

# ─── Step 6: Run benchmark on morningstar-32b ────────────
echo -e "${YELLOW}[6/6] Benchmark: morningstar-32b...${NC}"
bash benchmark.sh morningstar-32b 2>&1 | tee /tmp/benchmark-32b.txt
echo ""

# ─── Summary ─────────────────────────────────────────────
echo ""
echo -e "${CYAN}${BOLD}  ════════════════════════════════════════${NC}"
echo -e "${BOLD}  Alle Modelle erstellt und getestet!${NC}"
echo ""
echo -e "  Installierte Modelle:"
ollama list 2>/dev/null | grep morningstar
echo ""
echo -e "${CYAN}${BOLD}  ════════════════════════════════════════${NC}"
