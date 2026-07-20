#!/usr/bin/env bash
set -uo pipefail
cd "/home/leafline/leafline"

VENV="/home/leafline/leafline/.venv/bin/python"
CONFIG="/home/leafline/leafline/3_Model/configs/finetune_v1.yaml"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Kiel Fine-Tuning Pipeline                      ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Schritt 1: Sommer-Stacks vorbereiten ─────────────────────────────────
echo "[$(date '+%H:%M:%S')] ── Schritt 1: prepare_data.py --summer ──"
"$VENV" 3_Model/src/prepare_data.py --config "$CONFIG" --summer
STEP1=$?
if [ $STEP1 -ne 0 ]; then
  echo "[$(date '+%H:%M:%S')] FEHLER in prepare_data.py (Exit $STEP1)"
  exit $STEP1
fi

# ── Schritt 2: Training ──────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] ── Schritt 2: train.py ──"
"$VENV" 3_Model/src/train.py --config "$CONFIG"
STEP2=$?
if [ $STEP2 -ne 0 ]; then
  echo "[$(date '+%H:%M:%S')] FEHLER in train.py (Exit $STEP2)"
  exit $STEP2
fi

echo ""
echo "[$(date '+%H:%M:%S')] ══ Pipeline abgeschlossen ══"
