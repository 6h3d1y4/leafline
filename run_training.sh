#!/usr/bin/env bash
# run_training.sh — Session-unabhängige Pipeline für Kiel Fine-Tuning
#
# Befehle:
#   ./run_training.sh          — Pipeline starten (oder Status zeigen wenn läuft)
#   ./run_training.sh status   — Status + letzte 30 Log-Zeilen
#   ./run_training.sh log      — Live-Log verfolgen (Ctrl+C zum Beenden)
#   ./run_training.sh attach   — In tmux-Session einsteigen (Ctrl+B D zum Rausgehen)
#   ./run_training.sh stop     — Pipeline abbrechen

set -euo pipefail

SESSION="kiel-training"
ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv/bin/python"
CONFIG="$ROOT/3_Model/configs/finetune_v1.yaml"
LOG="$ROOT/3_Model/runs/pipeline.log"

mkdir -p "$ROOT/3_Model/runs"

# ── Unterbefehle ────────────────────────────────────────────────────────────

case "${1:-start}" in

  status)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "Status: LÄUFT (tmux-Session '$SESSION')"
      echo ""
      echo "  Live ansehen : ./run_training.sh attach"
      echo "  Log tail     : ./run_training.sh log"
      echo "  Abbrechen    : ./run_training.sh stop"
    else
      echo "Status: NICHT AKTIV"
    fi
    if [ -f "$LOG" ]; then
      echo ""
      echo "── Letzte 30 Log-Zeilen ($LOG) ──"
      tail -30 "$LOG"
    fi
    exit 0
    ;;

  log)
    if [ ! -f "$LOG" ]; then
      echo "Noch kein Log vorhanden: $LOG"
      exit 1
    fi
    echo "Log verfolgen (Ctrl+C zum Beenden):"
    tail -f "$LOG"
    exit 0
    ;;

  attach)
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "Keine aktive Session gefunden. Starte zuerst: ./run_training.sh"
      exit 1
    fi
    echo "Einsteigen in tmux (Ctrl+B dann D = rausgehen ohne abzubrechen):"
    tmux attach -t "$SESSION"
    exit 0
    ;;

  stop)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      tmux kill-session -t "$SESSION"
      echo "Session '$SESSION' beendet."
    else
      echo "Keine aktive Session gefunden."
    fi
    exit 0
    ;;

  start|"")
    ;;  # weiter unten

  *)
    echo "Unbekannter Befehl: $1"
    echo "Gültig: start | status | log | attach | stop"
    exit 1
    ;;
esac

# ── Start ────────────────────────────────────────────────────────────────────

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Pipeline läuft bereits in Session '$SESSION'."
  echo ""
  echo "  Live ansehen : ./run_training.sh attach"
  echo "  Status       : ./run_training.sh status"
  exit 0
fi

# Pipeline-Script als temporäre Datei (vermeidet Quoting-Probleme in tmux)
PIPELINE="$ROOT/3_Model/runs/_pipeline_run.sh"
cat > "$PIPELINE" << SCRIPT
#!/usr/bin/env bash
set -uo pipefail
cd "$ROOT"

VENV="$VENV"
CONFIG="$CONFIG"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Kiel Fine-Tuning Pipeline                      ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Schritt 1: Sommer-Stacks vorbereiten ─────────────────────────────────
echo "[\$(date '+%H:%M:%S')] ── Schritt 1: prepare_data.py --summer ──"
"\$VENV" 3_Model/src/prepare_data.py --config "\$CONFIG" --summer
STEP1=\$?
if [ \$STEP1 -ne 0 ]; then
  echo "[\$(date '+%H:%M:%S')] FEHLER in prepare_data.py (Exit \$STEP1)"
  exit \$STEP1
fi

# ── Schritt 2: Training ──────────────────────────────────────────────────
echo ""
echo "[\$(date '+%H:%M:%S')] ── Schritt 2: train.py ──"
"\$VENV" 3_Model/src/train.py --config "\$CONFIG"
STEP2=\$?
if [ \$STEP2 -ne 0 ]; then
  echo "[\$(date '+%H:%M:%S')] FEHLER in train.py (Exit \$STEP2)"
  exit \$STEP2
fi

echo ""
echo "[\$(date '+%H:%M:%S')] ══ Pipeline abgeschlossen ══"
SCRIPT
chmod +x "$PIPELINE"

# Tmux-Session starten: Pipeline-Ausgabe geht live in tmux UND in Logfile
tmux new-session -d -s "$SESSION" \
  "bash '$PIPELINE' 2>&1 | tee '$LOG'; echo ''; echo '[Fertig — Ctrl+B D zum Schließen]'; exec bash"

echo ""
echo "Pipeline gestartet in tmux-Session '$SESSION'."
echo ""
echo "  Live ansehen : ./run_training.sh attach"
echo "  Nur Log tail : ./run_training.sh log"
echo "  Status       : ./run_training.sh status"
echo "  Abbrechen    : ./run_training.sh stop"
echo ""
echo "SSH-Disconnect ist kein Problem — tmux läuft weiter."
