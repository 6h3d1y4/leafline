# Projekt-Anleitung: Struktur, Training, Zuständigkeiten

Diese Anleitung ist für Personen gedacht, die mit diesem Projekt arbeiten
sollen, aber nicht täglich mit Linux-Konsole, Python oder ML-Training zu tun
haben. Jeder Fachbegriff wird beim ersten Auftreten kurz erklärt. Stand:
16.07.2026.

Ergänzend siehe [`3_Model/TRAINING_STATUS.md`](3_Model/TRAINING_STATUS.md)
für den aktuellen Stand der Trainingsergebnisse.

---

## 1. Die drei Benutzer-Konten auf dem Server

Auf dem Server, auf dem dieses Projekt läuft, gibt es **drei verschiedene
Linux-Benutzerkonten**, die für unterschiedliche Dinge zuständig sind:

| Konto | Wofür |
|---|---|
| `joshuaj` | Persönliches Konto von Joshua (E-Mails, allgemeine Arbeit) |
| `leafline` | Das Projekt-Konto - hier liegt der Code (dieses Git-Repository), hier läuft z. B. Jupyter für Notebooks |
| `nda` | Das einzige Konto, das die **Trainingsdaten** lesen darf |

Das ist keine Schikane, sondern **Absicht**: Die Rohdaten unter `Data/Kiel`
(Luftbilder, Ground-Truth-Baumkronen) sind mit Linux-Dateiberechtigungen so
eingestellt, dass **nur `nda` sie lesen kann** - nicht einmal `leafline`
oder `joshuaj` haben Zugriff (das wurde technisch geprüft: die Zugriffsliste
für `Data/Kiel` erlaubt explizit nur `nda`, alle anderen sind ausdrücklich
gesperrt).

**Praktische Konsequenz:**

- Code ansehen/ändern, Ergebnisse (`3_Model/runs/...`) anschauen, in Git
  committen → geht als `leafline`.
- Trainingsdaten vorbereiten (`prepare_data.py`) oder ein Training starten
  (`train.py`, `evaluate.py`) → **geht nur als `nda`**, weil diese Skripte
  auf `Data/Kiel` zugreifen müssen. Man kann sich nicht einfach mit `su` von
  `leafline` aus zu `nda` wechseln (kein Passwort/keine sudo-Rechte dafür
  hinterlegt, getestet - siehe unten) - man muss sich direkt als `nda`
  anmelden (eigene SSH-Verbindung/Terminal-Sitzung als dieser Nutzer).

### 1.1 Wie meldet man sich an?

Beide Konten werden über **SSH** erreicht (verschlüsselte Fernzugriffs-
Verbindung von einem eigenen Rechner aus auf den Server). Die
Namensauflösung läuft über **Tailscale** (ein privates VPN-Netzwerk, das
Rechnernamen wie `joshua-desktop` automatisch auf die richtige Adresse
auflöst - dafür muss Tailscale auf dem eigenen Rechner laufen und im
selben "Tailnet" wie der Server sein).

- **Als `leafline` anmelden:**

  ```bash
  ssh leafline
  ```

  Das funktioniert nur, weil in der SSH-Konfiguration des eigenen Rechners
  (Datei `~/.ssh/config`, **client-seitig**, nicht auf dem Server) ein
  Kurzname `leafline` hinterlegt ist, der auf den passenden Host/Nutzer
  auflöst. Auf einem neuen Rechner muss dieser Eintrag zuerst angelegt
  werden - sonst funktioniert `ssh leafline` dort nicht.

- **Als `nda` anmelden:**

  ```bash
  ssh nda@joshua-desktop
  ```

  Hier wird Nutzername und Rechnername explizit angegeben, `joshua-desktop`
  wird über Tailscale aufgelöst. Der SSH-Schlüssel für dieses Konto ist
  bereits hinterlegt (kein Passwort nötig, sofern der eigene Rechner den
  passenden privaten Schlüssel hat und im Tailnet ist).

Wenn eine neue Person das Projekt übernimmt, muss geklärt werden, ob sie
(a) Zugriff auf das Tailscale-Netzwerk (Tailnet) bekommt und (b) ihr
eigener SSH-Schlüssel für das `nda`-Konto hinterlegt wird - sonst kann sie
kein Training starten oder Daten neu aufbereiten.

### 1.2 Claude Code steht nur auf `leafline` zur Verfügung

Auf `leafline` kann mit Claude Code (diesem KI-Assistenten) gearbeitet
werden - inklusive aller bisherigen Sessions/Verläufe, die hier fortgesetzt
werden können.

**Auf `nda` steht Claude Code nicht zur Verfügung - wegen des NDAs.** Die
Trainingsdaten unterliegen einer Vertraulichkeitsvereinbarung, die den
Einsatz von KI-Tools auf diesem Konto ausschließt. Das bedeutet konkret:

- Alles, was zwingend unter `nda` laufen muss - `prepare_data.py`,
  `train.py`, `evaluate.py`, `debug_predictions.ipynb` (siehe Abschnitt 4)
  - **muss von einem Menschen selbst ausgeführt werden**, ohne
  Unterstützung durch Claude Code. Diese Anleitung soll genau das so
  konkret wie möglich machen (siehe Abschnitt 5).
- Auf `leafline` kann Claude Code aber sehr wohl helfen, z. B. Configs zu
  schreiben, Logs/Ergebnisse aus `runs/` auszuwerten oder diese
  Dokumentation zu pflegen - nur eben nicht die Trainingsdaten selbst
  einsehen oder die eigentlichen Skripte ausführen.

### 1.3 Zwei Jupyter-Server - einer pro Konto

Sobald man als `leafline` angemeldet ist, hat man im Browser Zugriff auf
**zwei getrennte Jupyter-Server**, die schon laufen (jeweils über eine
SSH-Portweiterleitung vom Server auf den eigenen Rechner erreichbar):

| Adresse (im Browser) | Läuft unter Konto | Wofür |
|---|---|---|
| `http://localhost:8888` | `leafline` | Normale Notebooks (`1_DatasetCharacteristics`, `2_BaselineModel`-Vorlagen, `model_definition_evaluation.ipynb`, …) - kein Zugriff auf `Data/Kiel` |
| `http://localhost:8889` | `nda` | Notebooks, die Trainingsdaten brauchen - v. a. `3_Model/debug_predictions.ipynb` (siehe Abschnitt 4). Nur hier lässt sich dieses Notebook korrekt öffnen und ausführen, weil nur `nda` die Dateien unter `Data/Kiel` lesen darf |

**Wichtig:** Der `nda`-Jupyter-Server (Port 8889) ist gut geeignet, um
Notebook-Zellen anzusehen, Bilder/Overlays zu prüfen und **kurze
Testausführungen** zu machen - also genau das explorative Arbeiten aus
Abschnitt 3/4. Für den eigentlichen, stundenlangen Trainingslauf
(`train.py`) gilt trotzdem weiterhin: **Konsole + `tmux`, nicht der
Jupyter-Kernel** - die Gründe dafür stehen in Abschnitt 6 (kurz: ein
Notebook-Kernel überlebt keine Verbindungsabbrüche, ein `tmux`-Skript
schon). Der `nda`-Jupyter-Server ersetzt also nicht den in Abschnitt 5
beschriebenen Weg, sondern ergänzt ihn fürs Ansehen/Debuggen.

Da auf `nda` kein Claude Code läuft (Abschnitt 1.2), müssen Arbeiten in
diesem zweiten Jupyter-Server von Hand erledigt werden.

---

## 2. Projektstruktur - Was liegt wo?

```
leafline/
├── 0_LiteratureReview/       Recherche-Grundlagen
├── 1_DatasetCharacteristics/ Explorative Datenanalyse (Notebook)
├── 2_BaselineModel/          Vortrainiertes Modell ohne Fine-Tuning, als Vergleichsbasis
├── 3_Model/                  ← Das eigentliche Fine-Tuning. Siehe unten.
├── 4_Presentation/           Foliensatz / Ergebnispräsentation
├── Data/                     Rohdaten. "Kiel"-Unterordner ist nur für nda lesbar (siehe oben)
└── CoverImage/, notes/, ...  Nebensächlich
```

Der wichtige Ordner ist `3_Model/`:

```
3_Model/
├── src/                  Der eigentliche Trainings-Code (Python-Skripte)
│   ├── prepare_data.py   Schritt 1: Rohdaten in trainingsfertiges Format bringen
│   ├── train.py          Schritt 2: Das eigentliche Training
│   ├── evaluate.py       Schritt 3: Ein fertiges Modell auf einem Datensplit bewerten
│   ├── dataset.py        Interner Baustein: liefert Trainingsbeispiele an train.py
│   └── model_utils.py    Interner Baustein: lädt/passt das vortrainierte Modell an
├── configs/               Einstellungs-Dateien (YAML) - siehe Abschnitt 4
├── runs/                  Ergebnisse: ein Unterordner pro Trainingslauf
│   └── v1/, v1_no_ndom/, v1_sampling_fix/   (bisherige Läufe, siehe TRAINING_STATUS.md)
├── debug_predictions.ipynb   Untersuchungs-Notebook, siehe Abschnitt 3
├── SCHEDULE.txt            Geplante nächste Trainingsschritte
└── model_definition_evaluation.ipynb   Pflicht-Abgabe-Notebook (Kursvorlage, noch leer)
```

**Wichtig für die Übergabe:** `src/`, `configs/` und `SCHEDULE.txt` sind
aktuell **nicht** in Git eingecheckt (siehe `TRAINING_STATUS.md`,
Abschnitt 5). Sie liegen nur auf dieser Maschine unter `leafline`.

---

## 3. Warum ist das Training ein reines Python-Skript und kein Notebook?

Ein **Notebook** (`.ipynb`, z. B. in Jupyter) ist eine Datei, in der man
Code in einzelnen "Zellen" nacheinander per Klick ausführt und sofort
Ergebnisse/Bilder darunter sieht. Das ist ideal, wenn ein Mensch **live**
mitschaut und ausprobiert.

Ein **Skript** (`.py`) ist eine Datei, die man einmal von vorne bis hinten
komplett durchlaufen lässt - ohne dass jemand dabei sitzen muss.

Das Training (`train.py`) läuft **stundenlang** (der bisher längste Lauf
ging über ~47 Durchgänge/"Epochen" durch die Trainingsdaten, verteilt über
mehrere Sitzungen). Ein Skript ist dafür aus mehreren Gründen die bessere
Wahl:

1. **Es muss ohne Aufsicht laufen können.** Ein Jupyter-Notebook ist an eine
   Browser-Verbindung (bzw. den dahinterliegenden "Kernel"-Prozess)
   gebunden. Bricht die Verbindung ab (Laptop zuklappen, WLAN-Aussetzer,
   Browser-Tab versehentlich geschlossen), kann der Kernel abstürzen oder
   hängen bleiben - und das Training ist weg. Ein Skript, das in `tmux`
   läuft (siehe Abschnitt 5), läuft unabhängig von jeder Verbindung weiter.

2. **Es kann sich selbst fortsetzen.** `train.py` speichert nach jeder
   Epoche einen Zwischenstand ("Checkpoint", Datei `last.pt`) und prüft
   beim Start automatisch, ob so eine Datei schon existiert - falls ja,
   macht es genau dort weiter. Das ist bei den bisherigen Läufen mehrfach
   passiert (z. B. `v1` wurde über mehrere Tage/Sitzungen fortgesetzt).
   Ein Notebook könnte das auch, aber nur wenn eine Person das jedes Mal
   manuell wieder anstößt.

3. **Reproduzierbarkeit.** Ein Skript + eine Einstellungs-Datei (YAML,
   siehe Abschnitt 4) ist eine exakte, eindeutige Beschreibung "was genau
   wurde trainiert". In einem Notebook kann man Zellen in beliebiger
   Reihenfolge oder mehrfach ausführen - im Nachhinein ist oft nicht mehr
   sicher rekonstruierbar, was tatsächlich passiert ist.

4. **Durchgehendes Log statt Bildschirm-Ausgabe.** Das Skript schreibt
   laufend in zwei Dateien (`train_log.csv` mit den Zahlen,
   `train_stdout.log` mit dem vollen Text). Der Fortschritt ist also auch
   nachträglich oder von einem anderen Rechner aus einsehbar, ohne dass ein
   Notebook offen bleiben muss.

**Faustregel für dieses Projekt:** Alles, was lange dauert und am Ende ein
Ergebnis/eine Datei produzieren soll → Skript. Alles, was Ausprobieren,
Bilder ansehen und Denken in Zwischenschritten erfordert → Notebook.

---

## 4. Was macht `debug_predictions.ipynb`?

Das ist genau der Gegenfall zu Abschnitt 3: eine **Untersuchung**, kein
Produktionslauf - deshalb bewusst ein Notebook.

**Ausgangsproblem:** Das Training meldet für `v1` einen guten Wert
(F1 = 0.71 - ein gängiges Gütemaß, 1.0 = perfekt), aber die Auswertung auf
den zurückgehaltenen Testgebieten ergab nur F1 ≈ 0.12-0.17 - eine große,
zunächst unerklärte Lücke (dokumentiert in `TRAINING_STATUS.md`).

Das Notebook geht diese Lücke systematisch durch, Hypothese für Hypothese,
mit Bildern zum direkten Vergleich (Vorhersage vs. tatsächliche
Baumkronen-Umrisse übereinandergelegt). Das lässt sich nicht sinnvoll als
Skript schreiben, weil an mehreren Stellen ein Mensch das Bild ansehen und
beurteilen muss ("sieht das nach einem Baum aus oder nach einem
Registrierungsfehler?").

**Bisheriges Ergebnis der Untersuchung** (Stand des Notebooks):
Ein Koordinaten-/Ausrichtungsfehler wurde ausgeschlossen. Es gibt
stattdessen **zwei getrennte Ursachen**:

1. **Über-Segmentierung**: Das Modell zerlegt viele einzelne Baumkronen in
   mehrere kleine Vorhersage-Flecken statt einer zusammenhängenden Krone.
   Eine Anpassung von zwei technischen Nachbearbeitungs-Parametern
   (`min_dist`, `sigma` - steuern, wie na beieinander zwei separate
   Baumkronen erkannt werden) verbessert den F1-Wert auf einem Testgebiet
   bereits deutlich (0.173 → 0.331), **ist aber noch nicht in die
   Konfigurationsdatei übernommen worden.**
2. **Echte verpasste Baumkronen** (45 von 357 im Test): kleinere/niedrigere
   Kronen, die das Modell praktisch gar nicht erkennt. Das ist kein
   Postprocessing-Problem, sondern ein echtes Modell-/Trainingsdefizit,
   vermutlich weil solche kleinen Kronen in den Trainingsdaten
   unterrepräsentiert sind (wird im Notebook im letzten, noch nicht
   abgeschlossenen Abschnitt geprüft).

Diese Erkenntnisse sind wichtig für die Übergabe - sie erklären, warum die
Testergebnisse schlecht aussehen, ohne dass das Modell "kaputt" ist, und
geben konkrete nächste Schritte vor (im Notebook selbst als Checkliste
festgehalten).

**Zum Ausführen** braucht `debug_predictions.ipynb` denselben Datenzugriff
wie das Training - also ebenfalls das `nda`-Konto. Am einfachsten öffnet
man es über den `nda`-Jupyter-Server unter `http://localhost:8889`
(siehe Abschnitt 1.3) - dort lässt es sich ganz normal Zelle für Zelle
durchklicken.

---

## 5. Wie wird ein Training tatsächlich gestartet? (Schritt für Schritt)

Alle Befehle unten werden in einem Terminal ("Konsole", schwarzes
Text-Eingabefenster) eingegeben - **angemeldet als `nda`**, nicht als
`leafline` (siehe Abschnitt 1). Anmelden per `ssh nda@joshua-desktop`
(Details siehe Abschnitt 1.1). Da auf `nda` kein Claude Code zur Verfügung
steht (Abschnitt 1.2), müssen die folgenden Schritte von Hand ausgeführt
werden - genau dafür ist diese Anleitung so kleinschrittig gehalten.

### 5.1 `tmux` - warum und was ist das?

`tmux` ist ein Programm, das eine Terminal-Sitzung "am Leben hält", auch
wenn man die Verbindung trennt (SSH-Verbindung schließt, Laptop geht in
Standby, Internet fällt kurz aus). Ohne `tmux` würde ein laufendes Training
sofort abbrechen, sobald die Verbindung zum Server unterbrochen wird - mit
`tmux` läuft es einfach weiter, und man kann sich später wieder "dazuschalten".

**Neue Sitzung starten und Training anstoßen:**

```bash
tmux new -s training          # startet eine neue, benannte tmux-Sitzung "training"
cd ~/leafline
.venv/bin/python 3_Model/src/train.py --config 3_Model/configs/finetune_v1.yaml
```

(`.venv/bin/python` statt einfach `python` - das Projekt hat eine eigene,
abgeschottete Python-Installation mit genau den passenden Programmversionen,
das "virtuelle Environment"/"venv". Der normale `python`-Befehl würde die
falschen bzw. keine Programmversionen finden.)

**Die Sitzung verlassen, ohne das Training zu stoppen:**
Tastenkombination `Strg`+`b`, danach `d` drücken ("detach"). Man landet
wieder im normalen Terminal, das Training läuft im Hintergrund weiter.

**Später wieder reinschauen:**

```bash
tmux attach -t training
```

**Alle laufenden tmux-Sitzungen auflisten** (z. B. um zu prüfen, ob
irgendwo noch ein altes Training läuft):

```bash
tmux ls
```

Falls hier `error connecting to /tmp/tmux-... No such file or directory`
erscheint, heißt das: es läuft **aktuell gar kein** `tmux` und somit auch
kein Training (das war z. B. der Stand bei der Status-Prüfung am
16.07.2026 - siehe `TRAINING_STATUS.md`).

**Eine Sitzung endgültig beenden:** in der Sitzung `exit` eingeben, oder
von außen `tmux kill-session -t training`.

### 5.2 Fortschritt prüfen, ohne die Sitzung zu öffnen

Da `train.py` laufend in Dateien mitschreibt, muss man nicht einmal
`tmux attach` benutzen, um den Stand zu sehen:

```bash
tail -f ~/leafline/3_Model/runs/v1/train_stdout.log    # zeigt neue Zeilen live an, Strg+C zum Beenden
```

oder die Zahlen-Tabelle ansehen:

```bash
cat ~/leafline/3_Model/runs/v1/train_log.csv
```

### 5.3 Ist die Grafikkarte (GPU) gerade ausgelastet?

```bash
rocm-smi
```

zeigt an, ob die Grafikkarte (AMD, per ROCm angesprochen - das AMD-Gegenstück
zu NVIDIAs "CUDA") gerade rechnet. Praktisch, um zu prüfen, ob ein Training
wirklich aktiv ist oder z. B. hängengeblieben ist.

---

## 6. Warum Konsole/`tmux` und nicht einfach über Jupyter?

Jupyter läuft technisch auch als eigener Prozess, den man mit `tmux`
"überleben lassen" könnte - trotzdem wird für das eigentliche Training
bewusst **nicht** der Weg über einen Jupyter-Kernel/ein Notebook gewählt,
aus denselben Gründen wie in Abschnitt 3:

- Ein Jupyter-**Kernel** (der Prozess, der den Code eines Notebooks
  tatsächlich ausführt) ist empfindlicher: er kann bei langer Inaktivität,
  Speicherproblemen oder Verbindungsproblemen im Browser hängen bleiben
  oder neu starten - und ein neu gestarteter Kernel hat **keinen** der
  bisher berechneten Zwischenstände mehr im Speicher. Ein `train.py`-Lauf
  in `tmux` ist ein einzelner, einfacher Python-Prozess ohne diese
  Browser/Kernel-Zwischenschicht.
- Ein Notebook, das viele Stunden am Stück "eine Zelle ausführen" lässt,
  blockiert währenddessen effektiv das ganze Notebook für alles andere.
  Ein Skript in einer eigenen `tmux`-Sitzung stört nicht, wenn man daneben
  z. B. in einem anderen Notebook Ergebnisse auswerten möchte.
- Reproduzierbarkeit (s. o.): "Ich habe `train.py` mit `finetune_v1.yaml`
  laufen lassen" ist eine eindeutige, für andere nachvollziehbare Aussage.
  "Ich habe irgendwelche Zellen in einem Notebook ausgeführt" ist es nicht.

Jupyter/Notebooks bleiben trotzdem das richtige Werkzeug für alles
Explorative - Datenanalyse (`1_DatasetCharacteristics`), Baseline-Vergleich
(`2_BaselineModel`) und eben `debug_predictions.ipynb`.

Jupyter selbst wird übrigens absichtlich **nicht** mit `uv run jupyter lab`
gestartet, sondern mit `.venv/bin/jupyter lab --no-browser` - der `uv run`-
Weg versucht bei jedem Start alle Programmabhängigkeiten neu aufzulösen und
bricht dabei aktuell mit einem Fehler ab.

---

## 7. Kurz-Glossar

| Begriff | Bedeutung |
|---|---|
| **Checkpoint** (`.pt`-Datei) | Ein gespeicherter Zwischen- oder Endstand des trainierten Modells. `best.pt` = bester bisheriger Stand, `last.pt` = letzter Stand (zum Fortsetzen) |
| **Epoche** | Ein vollständiger Durchlauf durch alle Trainingsbeispiele |
| **F1-Wert** | Ein Gütemaß zwischen 0 (schlecht) und 1 (perfekt), das Genauigkeit und Vollständigkeit der Erkennung kombiniert |
| **Config/YAML** (`configs/*.yaml`) | Einstellungs-Datei: legt fest, welche Daten, wie viele Kanäle, welche Lernrate usw. für einen Lauf verwendet werden - ohne den Code selbst zu ändern |
| **venv** | "Virtuelles Environment" - eine in sich abgeschlossene Python-Installation nur für dieses Projekt, unter `leafline/.venv/` |
| **tmux** | Programm, das eine Terminal-Sitzung auch über Verbindungsabbrüche hinweg am Laufen hält (siehe Abschnitt 5.1) |
| **GPU / ROCm** | Die Grafikkarte, die das eigentliche Rechnen für das Training übernimmt (viel schneller als der normale Prozessor); ROCm ist die AMD-Software dafür |
| **ACL/Berechtigung** | Linux-Mechanismus, der festlegt, welches Benutzerkonto welche Dateien lesen/schreiben darf (Grund, warum `nda` nötig ist, siehe Abschnitt 1) |
