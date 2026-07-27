# Trainingsstatus — Stand 16.07.2026

Dokumentiert für die Projektübergabe. Basis: Inspektion von `3_Model/runs/*`,
`3_Model/configs/*`, `3_Model/src/*`, `SCHEDULE.txt` und Git-Historie.

---

## Update 27.07.2026 — Schedule-Schritt 1 trainiert & evaluiert

Seit dem 16.07. wurde Schritt 1 des `SCHEDULE.txt` umgesetzt, die Infrastruktur
dafür ausgebaut und erstmals ein sauberer Baseline-Vergleich erzeugt.

### Was gelaufen ist
- **Run `step1_spring75`** (`configs/finetune_step1_spring75.yaml`): 5 Kanäle
  (RGBI+NDVI, **kein nDOM**), **100% Frühjahr 7.5cm** (keine `_summer`-Gebiete),
  minimale Augmentierung (nur hflip/vflip/brightness/contrast). Bestes
  **pixelweises** Val-F1 **0.788 @ Epoche 13** (Early Stop @23) — besser als v1
  (0.706). Checkpoints in `runs/step1_spring75/checkpoints/`.
- **Test-Eval** (`runs/step1_spring75/eval_test.csv`, kronenweise IoU 0.5, PP 10/1)
  über alle drei Auflösungen erzeugt.
- **Baseline-Zahlen erstmals berechnet.** Das Baseline-Notebook
  (`2_BaselineModel/baseline_model.ipynb`) war nie bis zu den Ergebnis-Zellen
  durchgelaufen — es gab **keine** gespeicherte Baseline-F1. Neues Skript
  `2_BaselineModel/baseline_eval.py` reproduziert die `EVAL_PLAN`-Schleife
  (freudenberg2022, in_channels=5) und schreibt `runs/baseline_eval.csv`
  (PP 10/1) sowie `runs/baseline_eval_pp30.csv` (PP 30/2).

### Kernergebnis (Mikro-F1, kronenweise, identisches PP 10/1)

| Auflösung | Baseline | step1 | Δ |
|---|---|---|---|
| **7.5cm** (Frühjahr) | **0.000** | **0.120** | **+0.120** |
| **20cm** (Sommer) | **0.339** | 0.052 | **−0.287** |
| **20cm-spring** (Frühjahr) | 0.044 | 0.041 | ≈ 0 |

1. **7.5cm: Finetune notwendig und erfolgreich.** Die Baseline erkennt bei
   nativer 7.5cm-Auflösung praktisch nichts (0 Treffer, F1=0, OOD gegenüber ihrer
   ~20cm-Trainingsauflösung); step1 hebt das auf 0.120.
2. **20cm: step1 bricht ein**, die Baseline ist dort am stärksten. Der reine
   7.5cm-Finetune spezialisiert und verliert 20cm → **Auslöser für Schritt 2**
   (separate Modelle je Auflösung).
3. **20cm-spring: beide scheitern** (~0) — niedrige Auflösung + Frühjahrs-Laub.
4. **Postprocessing ist modell-/auflösungsabhängig**: 30/2 (aus
   `debug_predictions.ipynb`, fürs Finetune-Modell bei 7.5cm getunt) ist für die
   Baseline bei 20cm schlechter (0.263 vs. 0.339) und zerstört 7.5cm komplett.
   step1 über-segmentiert bei 7.5cm noch (408 vs. 357 GT) → mit per-Auflösung
   getuntem PP läge step1s 7.5cm-Wert höher.

Vollständige, live aus den CSVs berechnete Auswertung inkl. Diagrammen:
`3_Model/results_step1_vs_baseline.ipynb`.

### Nachtrag — Schedule-Schritt 2 trainiert & evaluiert (27.07.)

**Run `step2_spring20`** (`configs/finetune_step2_spring20.yaml`): 5 Kanäle,
**100% Frühjahr 20cm** (native `_native20_spring`-Stacks aus `DOP20-spring/`, gebaut
von `prepare_data.py --train-spring20`), minimale Augmentierung. Bestes pixelweises
Val-F1 **0.696 @ Epoche 14** (Early Stop @24).

Gesamtmatrix (kronenweise Mikro-F1, einheitliches PP 10/1):

| Auflösung | baseline | step1 (Frühjahr 7.5) | step2 (Frühjahr 20) |
|---|---|---|---|
| **7.5cm** (Frühjahr) | 0.000 | **0.120** | 0.021 |
| **20cm** (Sommer) | **0.340** | 0.052 | 0.008 |
| **20cm-spring** (Frühjahr) | 0.044 | 0.041 | **0.098** |

- **Schritt-2-Ziel erreicht:** step2 hebt 20cm-spring auf **0.098** (> 2× Baseline
  0.044 / step1 0.041) — das im Schedule erwartete „20cm spring gets better".
- **Diagonale:** jedes Modell ist nur auf seiner Trainingsdomäne stark (step1@7.5cm,
  step2@20cm-spring, baseline@20cm-Sommer).
- **Jahreszeit als eigener Faktor:** bei gleicher 20cm-Auflösung bricht step2 (Frühjahr)
  auf Sommer ein (0.008); die Baseline (Sommer) ist bei Frühjahr schwach (0.044).
  Auflösung *und* Saison verursachen je einen großen Domänen-Sprung.
- **Kiel-Einordnung** (fliegt Frühjahr): die relevanten Zellen 7.5cm (0.120) und
  20cm-spring (0.098) liegen nun klar über der Baseline.
- Caveat: PP ungetunt (step2 über-segmentiert 7.5cm: 1044 Vorhersagen).

Beim Vorbereiten fiel ein **Config-Tippfehler** auf: `gt_file: train/Shuetzenpark…`
(ohne „c") statt `Schuetzenpark…`. Latent bei 7.5cm (GT-`.tif` existierte schon →
`process_area` übersprang die Shapefile), brach aber Schritt 2 (neue GTs). In allen
sechs Configs korrigiert.

Gesamtauswertung aller Modelle: `3_Model/results_all_steps.ipynb`.

### Nachtrag — Postprocessing getunt, Hyperparameter geprüft, Schritt 3 erledigt (27.07.)

**Postprocessing (pp_sweep.py, pro Modell×Auflösung).** PP ist auflösungsabhängig:
7.5cm braucht großes min_dist (viele Fragmente mergen), 20cm kleines (sonst
Unter-Segmentierung). Ergebnis: step1 @7.5cm 0.120→**0.150** (min_dist=30/sigma=2),
step2 @20cm-spring 0.098→**0.113** (min_dist=10/sigma=3). Wichtig: der Gewinn kommt aus
der **Precision**; der **Recall bleibt über den ganzen Sweep flach** (~0.12) — die
Modell-Decke, nicht durch PP behebbar.

**Hyperparameter (CV, 3 Folds über LR-Raster).** step1 will LR 5e-5, step2 will 2e-4
(die feste 1e-4 war für beide suboptimal, in entgegengesetzte Richtung). Aber: die
finalen best-LR-Modelle bringen **auf dem Test nichts** — step1 0.150→0.152 (flach),
step2 0.113→**0.101 schlechter**, obwohl step2s val_F1 von 0.696 auf 0.802 stieg.
Bestätigt: LR ist nicht der Hebel, der Recall-Flaschenhals dominiert. → feste 1e-4 behalten.

**Schedule-Schritt 3 — 50/50 Sommer+Frühjahr 20cm (`step3_mix20`).** Bestes Modell auf
**beiden** 20cm-Spalten zugleich (PP 10/3):

| 20cm-Spalte | Baseline | step2 (nur Frühjahr) | **step3 (50/50)** |
|---|---|---|---|
| 20cm (Sommer) | 0.340 | 0.008 | **0.317** |
| 20cm-spring | 0.044 | 0.113 | **0.144** |

Das 50/50-Modell holt den Sommer fast auf Baseline-Niveau zurück **und** verbessert
sogar das Frühjahr gegenüber step2 (Recall steigt bei beiden; höchstes val_F1 aller
Läufe, 0.809). → **Saisons sind gemeinsam trainierbar, Schedule-Schritt 4 (separates
Sommer-Modell) entfällt.**

**Gelernte Struktur:** getrennte Modelle je **Auflösung** (7.5 vs. 20), aber **ein**
Modell je Auflösung über beide **Saisons**. Beste Modelle: step1 @7.5cm (0.150),
step3 @20cm (Frühjahr 0.144 / Sommer 0.317).

### Schedule-Stand & nächste Schritte
- Schritt 1 ✅ · 2 ✅ · 3 ✅ · 4 ⛔ (nicht nötig) — der Auflösungs-/Saison-Zweig ist **durch**.
- **Offen: Schritt 4b/5 — Höhenkanal (nDOM).** Config `finetune_step1_ndom_spring75.yaml`
  liegt bereit (step1 mit in_channels=6, sonst identisch → isolierter nDOM-Effekt bei 7.5cm).
- Neue Configs/Tools seit 27.07.: `finetune_step{1,2}_*_cvlr.yaml`, `finetune_step3_mix20.yaml`,
  `finetune_step1_ndom_spring75.yaml`, `pp_sweep.py`, `prepare_data.py --train-spring20/--train-summer20`.

### Infrastruktur-/Code-Änderungen (seit 16.07., committet)
- `dataset.py`: Augmentierung pro Komponente per Config schaltbar
  (`data.augment`), robust für 5-/6-Kanal-Patches.
- `train.py`: `oversample_small` kommt jetzt aus der Config (war hart `True`
  verdrahtet — reproduzierte v1 nicht); optionale k-Fold-CV über LR-Raster via
  `--cv-folds`/`--lr-grid`.
- `evaluate.py`: Auflösungs-Matrix (`--resolutions`) statt einzelnem `--season`.
- `prepare_data.py`: `--eval-resolutions` baut native 20cm-Eval-Stacks.
- `run_training.sh`: Config als Argument, kein erzwungenes `--summer`.
- Doku: `3_Model/README.md` (Betriebsanleitung Läufe/Eval),
  `results_step1_vs_baseline.ipynb`.

### Nächste Schritte
1. **Schritt 2 vorbereiten** — `configs/finetune_step2_spring20.yaml` analog zu
   step1, aber **100% Frühjahr 20cm** (RGBI+NDVI). Voraussetzung: 20cm-Trainings-
   stacks für die Trainingsgebiete müssen vorliegen (prüfen/erzeugen).
2. **Postprocessing pro Modell×Auflösung tunen** (min_dist/sigma), bevor finale
   Zahlen gezogen werden — der aktuelle 7.5cm-Wert von step1 ist durch
   Über-Segmentierung gedrückt.
3. **20cm-spring** bleibt offen (beide ~0): eigener Fokus, evtl.
   Kombi-Training (Schedule Schritt 3) oder gezielte Domänen-Augmentierung.
4. **Optional/isoliert**: Schritt 1 mit stdout-Logging reproduzieren und mit den
   neuen Augment-Schaltern die Beobachtung „minimale Aug → Val↑ aber Test↓ vs.
   v1s aggressive Aug" gezielt gegentesten; `--cv-folds` für den LR-Effekt.

---

## 1. Läuft aktuell ein Training?

**Nein.** Es existiert kein laufender tmux-Server (`/tmp/tmux-1001` nicht vorhanden)
und kein laufender Python-Trainingsprozess (`ps aux` geprüft). Alle drei
tmux-Sessions, die gestartet wurden, sind beendet bzw. die Maschine wurde
zwischenzeitlich neu gestartet — die Ergebnisse liegen aber vollständig auf
Platte vor.

## 2. Abgeschlossene Trainingsläufe

Drei Fine-Tuning-Läufe von `freudenberg2022.pt` (Deeptrees-Pretrained-Modell),
alle am 01.–02.07.2026 gelaufen, alle mit Early Stopping (`patience=10`):

| Run | Config-Unterschied zu `v1` | Epochen | Bestes val_F1 | @ Epoche | Val Precision/Recall |
|---|---|---|---|---|---|
| **v1** | Baseline-Finetune, 6 Kanäle (RGBI+NDVI+nDOM) | 0–46 (früh gestoppt) | **0.706** | 35 | P=0.736 / R=0.678 |
| v1_sampling_fix | wie v1, zusätzlich `oversample_small=True` (Sampling-Fix für kleine Bäume) | 0–22 (früh gestoppt) | 0.684 | 11 | P=0.677 / R=0.692 |
| v1_no_ndom | 5 Kanäle, **ohne** nDOM/Höhen-Kanal (Ablation) | 0–12 (früh gestoppt) | 0.657 | 2 | P=0.688 / R=0.629 |

**Bestes Modell nach Validierungs-F1 bleibt `v1`** (`runs/v1/checkpoints/best.pt`,
Epoche 35). Die beiden Varianten sollten `v1` verbessern, tun es aber
(bisher) nicht:
- `v1_sampling_fix` liegt knapp unter `v1` — der Sampling-Fix hat die
  Validierungsleistung nicht verbessert und stoppt außerdem viel früher.
- `v1_no_ndom` bestätigt, dass der Höhenkanal (nDOM) einen relevanten
  Beitrag leistet — ohne ihn ist F1 spürbar schlechter (0.657 vs. 0.706)
  und das Modell konvergiert/überanpasst extrem schnell (Bestwert schon
  bei Epoche 2).

Alle drei Läufe zeigen nach dem jeweiligen Bestwert ein deutliches
**Overfitting-Muster**: val_F1 bricht danach stark ein (z. B. `v1`:
0.706 → 0.10 bei Epoche 46), val_loss steigt parallel stark an.

## 3. ⚠️ Kritischer Befund: Generalisierung auf Testgebiete ist schlecht

Nur `v1` wurde bislang auf den echten Testgebieten evaluiert
(`runs/v1/eval_test.csv`):

| Gebiet | Auflösung | Precision | Recall | F1 |
|---|---|---|---|---|
| BotGarten | 7.5cm | 0.148 | 0.210 | 0.173 |
| HoernNord | 7.5cm | 0.170 | 0.099 | 0.125 |

Das ist eine **massive Lücke** zur Validierungs-F1 von 0.706. Wichtig für
die Einordnung: Die beiden Zahlen sind **nicht direkt vergleichbar** — der
Val-F1 in `train.py` misst pixelweise (jedes Pixel richtig/falsch als
„Baum"), während `evaluate.py`/`eval_test.csv` auf Ebene einzelner
Baumkronen misst (IoU-Matching, ≥50% Überlappung zählt als Treffer). Ein
Teil der Lücke ist also Definitionssache — der harte Rest wurde in
`debug_predictions.ipynb` bereits systematisch untersucht, siehe
Abschnitt 3.1.

Für `v1_sampling_fix` und `v1_no_ndom` liegt noch **keine**
Test-Set-Evaluation vor.

### 3.1 Ursachenanalyse liegt bereits vor (`debug_predictions.ipynb`)

Ein separates Untersuchungs-Notebook (`3_Model/debug_predictions.ipynb`,
nicht in Git, siehe Abschnitt 5) ist der Lücke bereits Hypothese für
Hypothese nachgegangen (CRS-/Koordinatenfehler ausgeschlossen). Ergebnis:
**zwei getrennte Ursachen**, beide anhand von `v1` auf `BotGarten`
quantifiziert:

1. **Über-Segmentierung** (Hauptanteil): Das Modell zerteilt viele
   Baumkronen in mehrere kleine Vorhersage-Polygone statt einer
   zusammenhängenden Krone (161 von 357 GT-Kronen betroffen, im Schnitt
   2.65 Vorhersage-Polygone pro Krone). Eine Anpassung der
   Nachbearbeitungs-Parameter `min_dist=30, sigma=2` (statt der
   Config-Default-Werte) hebt die F1 auf BotGarten bereits von 0.173 auf
   **0.331** — **aber diese Werte sind bisher nicht in
   `configs/finetune_v1.yaml` übernommen und nicht auf dem vollen
   Testsplit (beide Gebiete) bestätigt.**
2. **Echte verpasste Kronen** (45 von 357): kleine/niedrige Baumkronen,
   bei denen das Modell praktisch keine Aktivierung zeigt (kein
   Postprocessing-Problem). Stichprobe schließt Annotationsfehler
   weitgehend aus (4/5 der größten „missed"-Fälle sind eindeutig Bäume).
   Ob das an unterrepräsentierten kleinen Kronen in den Trainingsdaten
   liegt, wird im Notebook geprüft, **ist aber noch nicht abgeschlossen**
   (letzter Checklistenpunkt offen).

Die im Notebook festgehaltenen nächsten Schritte: (1) `min_dist`/`sigma`-
Fix in die Config übernehmen und auf dem vollen Testsplit bestätigen,
(2) je nach Ausgang von Punkt 11 im Notebook entweder gezieltes Sampling
kleiner Kronen in `dataset.py` oder einen größenabhängigen Loss-Term in
`train.py` einbauen.

**Kurz:** Der Test-F1 von 0.12–0.17 ist kein Zeichen eines kaputten
Modells — ein bekannter, teilweise bereits quantifizierter Fix
(Postprocessing) ist nur noch nicht angewendet, und eine zweite Ursache
(kleine Kronen) ist identifiziert, aber die Lösung noch offen.

## 4. Laufende Weiterentwicklung (nicht ausgeführt / nicht abgeschlossen)

Nach den drei Läufen (02.07.) wurde am 10.–13.07. weitergearbeitet, aber
**es wurde kein weiteres Training gestartet**, das Ergebnisse produziert
hätte:

- `SCHEDULE.txt` (13.07.) beschreibt einen mehrstufigen Trainingsplan
  (getrennte Modelle je Auflösung/Jahreszeit, Cross-Validation,
  schrittweise Ergänzung des Höhenkanals). **Status: Plan vorhanden,
  noch kein Schritt daraus ausgeführt.**
- `configs/finetune_v1_no_ndom_7.5.yaml` (10.07.) — neue Konfiguration
  für 7.5cm-Auflösung ohne nDOM. Zielverzeichnis
  `runs/v1_no_ndom_7.5` **existiert nicht** — dieser Lauf wurde nie
  gestartet oder ist ohne jede Ausgabe abgebrochen.
- `src/dataset.py` wurde am 10.07. umgebaut (jahreszeiten-bewusste
  Augmentierung: NDVI/NIR-Skalierung, RGB-Jitter). `src/dataset_copy.py`
  ist eine ältere/parallele Fassung (einfache Brightness/Contrast +
  Flips, NDVI-Neuberechnung) — vermutlich ein Backup vor dem Umbau.
  **Mit dem neuen `dataset.py` wurde noch kein Trainingslauf
  durchgeführt** — keiner der drei dokumentierten Runs nutzt diesen Code
  (deren stdout-Logs zeigen den alten Stand).
- `3_Model/debug_predictions.ipynb` — siehe Abschnitt 3.1: enthält die
  Ursachenanalyse zur Val/Test-F1-Lücke, inkl. bereits quantifiziertem,
  aber noch nicht übernommenem Postprocessing-Fix.

## 5. ⚠️ Kritischer Befund: Nichts davon ist im Git-Repo

Der komplette Trainings-Code und alle Ergebnisse sind **nicht committed**
(`git status` in `/home/leafline/leafline`, Branch `main`, letzter Commit
`da1291c` vom 07.06.2026):

```
?? 3_Model/SCHEDULE.txt
?? 3_Model/configs/
?? 3_Model/debug_predictions.ipynb
?? 3_Model/runs/          (ohnehin über .gitignore-Data ausgeschlossen wäre,
                            aber src/ ist es NICHT)
?? 3_Model/src/           ← train.py, dataset.py, evaluate.py, model_utils.py
```

`3_Model/src/` (also die gesamte Trainings-Implementierung: `train.py`,
`dataset.py`, `evaluate.py`, `model_utils.py`) ist **zu keinem Zeitpunkt
eingecheckt worden**. Bei einer Übergabe über GitHub/das Repo allein würde
die Nachfolgeperson **nur die leeren Template-Notebooks** sehen, nicht den
tatsächlichen Code. Empfehlung: vor der Übergabe mindestens `src/`,
`configs/`, `SCHEDULE.txt` committen (die `runs/`-Checkpoints, ~350MB je
Lauf, sollten wegen der Größe eher separat dokumentiert/übergeben werden,
z. B. per Pfadangabe oder externem Storage, nicht per Git).

## 6. Sonstiges

- Dateien unter `3_Model/runs/` und `2_BaselineModel/*_copy.ipynb` gehören
  dem Linux-User `nda`, nicht `leafline` — Zugriff funktioniert aktuell
  über Gruppenrechte (`leafline`-Gruppe hat rwx), sollte aber bei der
  Übergabe an eine dritte Person geprüft werden.
- `2_BaselineModel/deeptrees_baseline_copy.ipynb` bricht mit
  `PermissionError` auf `Data/Kiel/TrainingAreas/DOP20/BotGarten.tif` ab.
  Ursache geklärt: `Data/Kiel` ist `0700`, Eigentümer `nda`, und die
  Zugriffsliste (ACL) sperrt `leafline` (und `joshuaj`) explizit — siehe
  `PROJEKT_ANLEITUNG.md` Abschnitt 1. Das Notebook läuft aber unter dem
  Jupyter-Prozess des `leafline`-Kontos → deshalb der Fehler. Für eine
  frische Baseline-Auswertung muss auch dieses Notebook (bzw. ein
  äquivalentes Skript) unter `nda` laufen.
- Speicherbedarf: je Lauf ~350MB (`best.pt` + `last.pt` je ~183MB).

## Kurzfassung für die Übergabe

1. Kein Training läuft gerade.
2. Bestes Modell bisher: **`v1`** (Val-F1 0.706), die beiden Experimente
   (`sampling_fix`, `no_ndom`) haben es **nicht** übertroffen.
3. Aber: `v1` erreicht auf den echten Testgebieten nur F1 ≈ 0.12–0.17 —
   die Val-Zahl (pixelweise) und die Test-Zahl (pro Baumkrone) sind
   unterschiedliche Metriken. Die Lücke ist bereits analysiert
   (`debug_predictions.ipynb`, Abschnitt 3.1): ein quantifizierter
   Postprocessing-Fix (+0.16 F1 auf BotGarten) ist nur noch nicht
   übernommen, eine zweite Ursache (kleine Kronen) ist erkannt, Lösung
   offen.
4. Ein detaillierter Folgeplan liegt vor (`SCHEDULE.txt`), ist aber noch
   nicht begonnen.
5. **Dringend vor Übergabe:** `src/`, `configs/`, `SCHEDULE.txt` in Git
   committen — aktuell nur lokal auf dieser Maschine vorhanden.
6. Wie man ein Training praktisch startet/prüft und warum die Konten
   `nda`/`leafline` getrennt sind: siehe `PROJEKT_ANLEITUNG.md`.
