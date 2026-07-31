# Model - Fine-Tuning & Evaluation

Betriebsanleitung: **wie Trainingsläufe und Auswertungen gemacht werden.** Für die
inhaltliche Modell-/Ergebnisdarstellung siehe
[`model_definition_evaluation.ipynb`](model_definition_evaluation.ipynb) und
[`results_notebook.ipynb`](results_notebook.ipynb).

---

## 0. Wichtigste Voraussetzung: als `nda` ausführen

Die Trainingsdaten unter `Data/Kiel/` sind per ACL **nur für den Nutzer `nda`**
lesbar - `leafline`/`joshuaj` haben keinen Zugriff. Alles, was Daten liest
(`prepare_data.py`, `train.py`, `evaluate.py`, die Notebooks), muss deshalb in
einer **`nda`-Sitzung** laufen:

```bash
ssh nda@joshua-desktop
cd ~/leafline
```

Details zu Konten, SSH/Tailscale und warum das so ist:
[`../PROJEKT_ANLEITUNG.md`](../PROJEKT_ANLEITUNG.md) Abschnitt 1.

---

## 1. Ein Trainingslauf in 2 Schritten

Jeder Lauf ist: **Daten aufbereiten → trainieren.** Beides wird von einem
Config-File (`configs/*.yaml`) gesteuert - welcher Lauf gemacht wird, hängt
allein am gewählten Config.

### Bequemster Weg (tmux-Wrapper, überlebt SSH-Disconnect)

```bash
./run_training.sh                                   # nutzt configs/finetune_step1_spring75.yaml
./run_training.sh start <config.yaml>               # anderer Config
./run_training.sh start <config.yaml> --summer      # zusätzliche prepare_data-Flags
```

Steuerung der laufenden Session (`kiel-training`):

```bash
./run_training.sh status    # Status + letzte 30 Log-Zeilen
./run_training.sh log       # Live-Log verfolgen (Ctrl+C = nur Ansicht beenden)
./run_training.sh attach    # in die tmux-Session (Ctrl+B dann D = raus, ohne Abbruch)
./run_training.sh stop      # Lauf abbrechen
```

### Manuell (ohne Wrapper)

```bash
CFG=3_Model/configs/finetune_step1_spring75.yaml
.venv/bin/python 3_Model/src/prepare_data.py --config $CFG
.venv/bin/python 3_Model/src/train.py        --config $CFG
```

In tmux als ein Block:

```bash
cd ~/leafline
mkdir -p 3_Model/runs/step1_spring75
tmux new-session -d -s kiel-step1 \
  '.venv/bin/python 3_Model/src/prepare_data.py --config 3_Model/configs/finetune_step1_spring75.yaml \
   && .venv/bin/python 3_Model/src/train.py --config 3_Model/configs/finetune_step1_spring75.yaml \
   2>&1 | tee 3_Model/runs/step1_spring75/train_stdout.log; exec bash'
# reinschauen: tmux attach -t kiel-step1   (raus: Ctrl+B dann D)
```

**Automatisches Resume:** `train.py` setzt automatisch aus
`runs/<name>/checkpoints/last.pt` fort, wenn vorhanden - ein Neustart beginnt
also nicht bei null. Für einen echten Neustart das `runs/<name>/`-Verzeichnis
(oder nur `checkpoints/last.pt`) vorher löschen.

---

## 2. Was ein Config steuert (`configs/*.yaml`)

Ein Lauf wird vollständig über den Config definiert - so lässt sich **je ein
Faktor isoliert** verändern (Ziel des `SCHEDULE.txt`-Vorgehens).

```yaml
paths:
  base:       .../Data/Kiel/TrainingAreas      # Datenwurzel
  pretrained: ~/pretrained_models/freudenberg2022.pt
  output:     .../3_Model/runs/<name>          # hierhin gehen Checkpoints + Logs

data:
  patch_size: 256
  train_areas: [...]        # Gebiete + gt_file. _summer-Varianten teilen die GT ihres Basisgebiets
  valid_areas: [...]
  test_areas:  [...]
  augment:                  # jede Komponente einzeln an/aus (fehlt der Block → alle an)
    hflip: true
    vflip: true
    brightness: true        # RGB-Helligkeit
    contrast: true          # RGB-Kontrast
    nir_scale: false        # NIR-Skalierung        ) spektrale Sommer→Frühjahr-
    ndvi_scale: false       # NDVI-Herunterskalierung) Simulation - bei 100% Frühjahr AUS
    spectral_noise: false   # Gauß-Rauschen ch0-4

model:
  in_channels: 5            # 5 = R G B I NDVI  |  6 = zusätzlich nDOM (Höhe)
  lr: 1.0e-4

training:
  batch_size: 8
  max_epochs: 50
  patience: 10              # Early Stopping nach 10 Epochen ohne val_F1-Verbesserung
  oversample_small: false   # kleine Kronen gezielt oversamplen (Sampling-Experiment)

postprocessing: {...}       # Watershed-/Schwellenparameter, nur für die Eval relevant
```

Wichtig: `in_channels: 5` lässt den nDOM-Kanal weg (RGBI+NDVI), `6` nimmt ihn
dazu - mehr braucht es für die Höhen-Ablation nicht. Die `augment`- und
`oversample_small`-Schalter existieren, damit man den Effekt jeder einzelnen
Maßnahme sauber messen kann.

### Vorhandene Configs

| Config | Kanäle | Daten | Besonderheit |
|---|---|---|---|
| `finetune_step1_spring75.yaml` | 5 (RGBI+NDVI) | 100% Frühjahr 7.5cm | **Schedule Schritt 1**, minimale Augmentierung |
| `finetune_step2_spring20.yaml` | 5 (RGBI+NDVI) | 100% Frühjahr 20cm | **Schedule Schritt 2**, braucht `prepare_data.py --train-spring20` (baut `_native20_spring`-Stacks aus `DOP20-spring/`) |
| `finetune_v1.yaml` | 6 (+nDOM) | Frühjahr + `_summer` | erster Baseline-Finetune (bestes val_F1 bisher) |
| `finetune_v1_no_ndom.yaml` | 5 | wie v1 | Höhen-Ablation von v1 |
| `finetune_v1_sampling_fix.yaml` | 6 | wie v1 | `oversample_small: true` |
| `finetune_v1_no_ndom_7.5.yaml` | 5 | wie v1 | nie gelaufen |

---

## 3. Optionale Cross-Validation / Hyperparameter-Suche

Statt eines festen Splits kann `train.py` k-Fold-CV über ein LR-Raster fahren
(Schedule: „cross-fold to get best hyperparam setup"). Es misst **nur den
Hyperparameter-Effekt** - es werden keine Modell-Checkpoints gespeichert:

```bash
.venv/bin/python 3_Model/src/train.py \
    --config 3_Model/configs/finetune_step1_spring75.yaml \
    --cv-folds 5 --lr-grid 5e-5,1e-4,2e-4
```

- poolt `train_areas` + `valid_areas` und teilt sie in `k` Folds,
- trainiert je Fold frisch vom Pretrained-Checkpoint,
- meldet mean±std val_F1 je LR und schreibt `runs/<name>/cv_results.csv`,
- `--lr-grid` weglassen → nur `model.lr` aus dem Config.

Ohne `--cv-folds` läuft der normale Single-Run.

---

## 4. Auswertung (Test-Set, baseline-vergleichbar)

Nach dem Training auf den Testgebieten evaluieren. `evaluate.py` misst
**kronenweise** (IoU-Matching ≥ 0.5), vergleichbar mit der Baseline:

```bash
# einmalig: native-20cm-Eval-Stacks für die Testgebiete bauen
.venv/bin/python 3_Model/src/prepare_data.py --config <config> --eval-resolutions

# Auswertung über alle drei Auflösungen (Default)
.venv/bin/python 3_Model/src/evaluate.py \
    --config <config> \
    --checkpoint 3_Model/runs/<name>/checkpoints/best.pt \
    --split test
# → runs/<name>/eval_test.csv  (Zeilen je Gebiet × Auflösung + Mikro-Schnitt)

# nur eine Auflösung:
#   --resolutions 20cm-spring
```

- Auflösungen: `7.5cm` (Frühjahr, nativ), `20cm` (Sommer DOP20), `20cm-spring`
  (Frühjahr DOP20). Labels stimmen mit `EVAL_PLAN` in
  [`../2_BaselineModel/baseline_model.ipynb`](../2_BaselineModel/baseline_model.ipynb)
  überein → direkt vergleichbar.
- `--eval-resolutions` braucht `Data/Kiel/TrainingAreas/DOP20-spring/`.
- Hinweis Postprocessing: `debug_predictions.ipynb` fand `min_dist=30, sigma=2`
  klar besser gegen Über-Segmentierung als der Config-Default (`10/1`). Das ist
  eine reine Eval-Frage (kein Training) - beim Auswerten gegentesten.

Der Gesamtvergleich (Baseline vs. Finetuning je Auflösung) läuft in
[`results_notebook.ipynb`](results_notebook.ipynb).

---

## 5. Ausgaben eines Laufs (`runs/<name>/`)

```
runs/<name>/
  checkpoints/best.pt   # bestes Modell nach val_F1
  checkpoints/last.pt   # letzter Stand (für Resume)
  train_log.csv         # je Epoche: train/val-Loss, val_F1/P/R, Threshold
  train_stdout.log      # vollständige Konsolenausgabe
  cv_results.csv        # nur bei --cv-folds
  eval_test.csv         # nach evaluate.py
```

Checkpoints sind ~183 MB je Datei und über `.gitignore` vom Repo ausgeschlossen -
nicht committen, separat übergeben.

---

## 6. Trainingsplan

Die geplante Reihenfolge der Läufe (getrennte Modelle je Auflösung/Jahreszeit,
dann Höhenkanal) steht in [`SCHEDULE.txt`](SCHEDULE.txt); der Stand der bereits
gelaufenen Experimente in [`TRAINING_STATUS.md`](TRAINING_STATUS.md).
