# Das Projekt komplett erklärt – von null bis „ich verstehe alles"

> Diese Datei ist dein Nachschlagewerk. Du musst sie **nicht** am Stück lesen.
> Lies Kapitel 1–4 einmal für den Überblick; den Rest schlägst du nach, wenn du
> eine konkrete Datei oder einen Begriff verstehen willst. Alles hier bezieht
> sich 1:1 auf den Code in diesem Ordner.

**Inhalt**
1. [Das Projekt in 60 Sekunden](#1-das-projekt-in-60-sekunden)
2. [Reinforcement Learning in einfachen Worten](#2-reinforcement-learning-in-einfachen-worten)
3. [Der GPU-Trick: tausende Welten gleichzeitig](#3-der-gpu-trick-tausende-welten-gleichzeitig)
4. [Die drei Bausteine](#4-die-drei-bausteine)
5. [Datei-für-Datei-Führer](#5-datei-für-datei-führer)
6. [Deep Dive: die Mission „Netzabdeckung“](#6-deep-dive-die-mission-netzabdeckung)
7. [Das neuronale Netz im Detail](#7-das-neuronale-netz-im-detail)
8. [Ein Trainingsschritt, Schritt für Schritt](#8-ein-trainingsschritt-schritt-für-schritt)
9. [Wie man alles startet](#9-wie-man-alles-startet)
10. [Das Trainingslog lesen](#10-das-trainingslog-lesen)
11. [Aktueller Stand & das offene Lade-Problem](#11-aktueller-stand--das-offene-lade-problem)
12. [Design-Entscheidungen & Warum (Interview-Gold)](#12-design-entscheidungen--warum-interview-gold)
13. [Glossar](#13-glossar)

---

## 1. Das Projekt in 60 Sekunden

Du baust eine **Simulation**, in der ein **Drohnenschwarm** lernt, bei einer
Katastrophe (Erdbeben, Flut – Mobilfunk ausgefallen) ein **fliegendes Funknetz**
über dem Gebiet aufzuspannen. Die Drohnen sollen:

- die Stellen mit Bedarf (Notunterkunft, Krankenhaus, Einsatzleitung) mit
  Internet-Kapazität versorgen,
- sich dafür zu einer Funk-Kette (**Mesh**) verbinden, die bis zum **Uplink**
  (Satellit) reicht,
- und **rechtzeitig zum Laden** fliegen, bevor der Akku leer ist – ohne dass
  dabei ein Funkloch entsteht.

Kein Mensch programmiert die Taktik. Die Drohnen **lernen sie selbst** per
Reinforcement Learning, indem sie millionenfach ausprobieren und für gutes
Verhalten belohnt werden.

**Warum das ein starkes Portfolio-Projekt ist:** Es verbindet drei anspruchsvolle
Themen – Multi-Agent-RL, GPU-beschleunigte Simulation und ein realistisches,
erklärbares Szenario mit echtem gesellschaftlichem Nutzen.

---

## 2. Reinforcement Learning in einfachen Worten

**Reinforcement Learning (RL)** = Lernen durch Belohnung. Kein Datensatz mit
„richtigen Antworten“, sondern ein **Agent**, der in einer **Umgebung** handelt
und für die Folgen belohnt oder bestraft wird. Wie Hundetraining: gutes Verhalten
→ Leckerli, schlechtes → nichts.

Der Kreislauf (merk dir diese fünf Wörter):

```
Beobachtung → Aktion → Belohnung → neue Beobachtung → …
  (State)      (Action)   (Reward)      (next State)
```

- **Beobachtung / State:** Was die Drohne gerade „sieht“ (Position, Akku,
  Richtung zum nächsten Funkloch …).
- **Aktion / Action:** Was sie tut (in eine Richtung fliegen, steigen, stehen).
- **Belohnung / Reward:** Eine Zahl, die sagt, wie gut das war. **Die Reward-
  Funktion ist die wichtigste Design-Entscheidung im ganzen RL** – sie definiert,
  was „gut“ überhaupt heißt.
- **Policy:** Die Strategie des Agenten – eine Funktion „Beobachtung → Aktion“.
  Genau diese Policy ist das neuronale Netz, das trainiert wird.

Ziel des Trainings: eine Policy finden, die die **summierte zukünftige Belohnung**
maximiert (nicht nur die von jetzt – auch die in 10 Sekunden). Wie stark die
Zukunft zählt, steuert **gamma** (siehe Glossar).

---

## 3. Der GPU-Trick: tausende Welten gleichzeitig

RL braucht enorm viel Erfahrung. Der Trick dieses Projekts: statt eine Welt nach
der anderen zu simulieren, liegen **alle Welten gleichzeitig** als ein großer
Zahlen-Block (**Tensor**) auf der Grafikkarte, und ein einziger Rechenschritt
bearbeitet sie alle parallel.

Die zentrale Konvention, die sich durch den **gesamten** Code zieht:

```
Form eines Zustands-Tensors:  (E, A, …)

E = Environments  → parallele Arenen/Welten   (z. B. 1024)
A = Agents        → Drohnen pro Welt           (z. B. bis 20)
…  = die eigentlichen Werte                    (z. B. x, y, Höhe)
```

Beispiel aus `env_netcov.py`: `self.pos` hat die Form `(E, A, 3)` – für jede der
E Welten, jede der A Drohnen, drei Zahlen (x, y, Höhe). Ein Aufruf von `step()`
bewegt damit *alle* Drohnen in *allen* Welten auf einmal. Deshalb schafft deine
RTX 4060 Ti rund **19.000 Simulationsschritte pro Sekunde**.

> **Merke:** Es gibt (fast) keine `for`-Schleife über Agenten oder Welten. Alles
> ist „vektorisiert“ – als Matrix-Rechnung formuliert, die die GPU liebt. Das ist
> der Grund, warum der Code manchmal wie Zauberei mit `unsqueeze`, `cdist` und
> `torch.where` aussieht: das sind nur Wege, Schleifen durch Matrix-Operationen
> zu ersetzen.

---

## 4. Die drei Bausteine

Alles im Projekt sind im Kern drei Teile, die in einer Schleife zusammenspielen:

```
   ┌──────────────────────┐   Beobachtung    ┌─────────────────────┐
   │   ENVIRONMENT         │ ────────────────▶│   POLICY (Netz)      │
   │   env_netcov.py       │                  │   attn.py            │
   │   Physik + Funk+Akku  │◀──────────────── │   Actor: wählt Aktion│
   └──────────────────────┘    Aktion         └─────────────────────┘
            │ Belohnung                                   ▲
            └───────────────────────────────────────────┘
                   TRAINER (MAPPO) passt die Netz-Gewichte an
                   train_netcov.py + mappo.py
```

### a) Das Environment – die Spielwelt (`env_*.py`)
Die Simulation selbst. Enthält **keine** KI, nur Physik und Regeln: bewegt
Drohnen, rechnet Funkausbreitung und Akku, verteilt Belohnung. Austauschbar – es
gibt mehrere Missionen (Zonenkontrolle, Relais, 3D-Suche, Netzabdeckung).

### b) Die Policy – das „Gehirn“ (`attn.py`, früher `mappo.py`)
Das neuronale Netz. Zwei Köpfe:
- **Actor** (dezentral): sieht die Sicht *einer* Drohne → wählt eine Aktion.
  Ein einziges Netz steuert alle Drohnen (**Parameter-Sharing**).
- **Critic** (zentral): schätzt, wie gut die Lage insgesamt ist. Wird **nur beim
  Training** gebraucht, nicht im Einsatz.

### c) Der Trainer – MAPPO (`mappo.py`, `train_netcov.py`)
Sammelt Erfahrung, rechnet aus, welche Aktionen besser als erwartet waren
(**Advantage** via **GAE**), und schiebt das Netz vorsichtig in die bessere
Richtung (**PPO**).

Dieses Muster heißt **CTDE** (*Centralized Training, Decentralized Execution*):
Beim Lernen darf der Critic die ganze Welt sehen; im Einsatz handelt jede Drohne
nur mit ihrer lokalen Sicht. Das ist der Standard-Ansatz bei MAPPO.

---

## 5. Datei-für-Datei-Führer

### Das Paket `swarm_sim/` (die wiederverwendbaren Bausteine)

| Datei | Was drin ist | Kernstück |
| --- | --- | --- |
| `config.py` | Alle Parameter an einem Ort: `SimConfig` (Welt/Physik) + `TrainConfig` (Hyperparameter) | Dataclasses |
| `physics.py` | Vektorisierte **2D**-Physik: Schub, quadratischer Luftwiderstand, Wandabprall, Agent-Agent-Kollisionen | `PhysicsWorld.step()` |
| `physics3d.py` | 3D-Variante der Physik (für die Such-Mission) | – |
| `swarm.py` | Regelbasiertes „Gehirn“ (**Boids**): Separation, Alignment, Kohäsion. Das war die Policy *vor* dem ML (Phase 1) | `BoidsController` |
| `mesh.py` | Funk-/Mesh-Netz: wer ist mit wem verbunden, welche **Gruppen** entstehen, wer ist **Anführer** | `MeshNetwork.compute()` |
| `mappo.py` | Der einfache MAPPO-Trainer (fester Agentenzahl) + PPO-Update + `compute_gae()` | `MAPPO.update()` |
| `attn.py` | Das **Attention**-Netz, das mit wechselnder Drohnenzahl klarkommt | `AttnActorCritic` |
| `env.py` | Basis-Mission: Zonenkontrolle, 2 Teams, Self-Play (Phase 3) | – |
| `env_dynamic.py` | Wie `env` + Nachschub, Ausfälle, Zerstörung (Phase 4) | – |
| `env_relay.py` | Funk-Relais-Mission: einige Drohnen halten die Verbindung | – |
| `env_search3d.py` | 3D-Such-Mission: Gebiet absuchen, Ziele finden, Vögel ausweichen | – |
| `env_netcov.py` | **Die Flaggschiff-Mission: Netzabdeckung** (Kapitel 6) | `NetCovEnv` |
| `mission_config.py` | Lädt `netzabdeckung_config.yaml`, sampelt Bereiche `[lo,hi]`, wendet das Curriculum an | `MissionConfig` |
| `recorder.py` | Speichert Replays als `.npz` | `ReplayRecorder` |
| `viewer.py` | matplotlib-Viewer (live + Replay abspielen) | – |
| `metrics.py` | Logging nach TensorBoard + GPU-/Durchsatz-Werte | `MetricsLogger` |

### Dateien im Hauptordner (die „Startknöpfe“)

| Datei | Zweck |
| --- | --- |
| `run_sandbox.py` | Phase-1-Demo ohne ML: `live` / `record` / `bench` |
| `train.py` | Basis-MAPPO-Training (Zonenkontrolle) |
| `train_dynamic.py` | Training mit Dynamik + Attention |
| `train_relay.py` | Training der Relais-Mission |
| `train_search3d.py` | Training der 3D-Suche |
| `train_netcov.py` | **Training der Netzabdeckung** (der aktuelle Fokus) |
| `show_netcov.py`, `show_relay.py` | Trainiertes Verhalten rendern/auswerten (Bilder nach `out/`) |
| `dashboard.py` | Lokales Browser-Dashboard (Beobachter + Starter), `http://localhost:7070` |
| `netzabdeckung_config.yaml` | **Alle** Zahlen der Netzabdeckungs-Mission (nichts hart im Code) |
| `resume_training.bat` | Setzt das Training vom gesicherten Checkpoint fort |

### Ordner

| Ordner | Inhalt | Im Git? |
| --- | --- | --- |
| `swarm_sim/` | der Paket-Code | ✅ |
| `out/` | Bilder/Logs der Läufe | nur `.png` ✅, Logs ❌ |
| `runs/` | TensorBoard-Daten | ❌ (gitignored) |
| `checkpoints/` | trainierte Modelle `.pt` | ❌ (zu groß → Hugging Face Hub) |
| `replays/` | aufgezeichnete Runden `.npz` | ❌ |
| `.venv/` | virtuelle Python-Umgebung | ❌ |

---

## 6. Deep Dive: die Mission „Netzabdeckung“

Alles hier steckt in `swarm_sim/env_netcov.py`; alle Zahlen in
`netzabdeckung_config.yaml`.

### Was die Drohne sieht (Beobachtung)

Zwei Teile, zusammen 7 + 13 Werte pro Drohne:

**`entities` (7 Werte, die „rohe“ Eigensicht):**
`[x, y, Höhe, vx, vy, vz, Akku]` – alles normiert (auf 0..1 skaliert).

**`self_global` (13 Werte, die „aufbereiteten Hinweise“):**
Richtung + Distanz zum **Uplink**, zur nächsten **Ladestation**, zur nächsten
**Versorgungslücke**, dazu Akku, Verbindungsstatus und Wind. Das ist die
„Navigationshilfe“, die dem Netz das Lernen erleichtert.

Dazu `team_global` (3 Werte, Team-Gesamtlage): Abdeckung, Anteil verbundener
Drohnen, mittlerer Akku. Die sieht nur der **Critic**.

### Was die Drohne tun kann (11 Aktionen)
Diskret: **stehen**, **8 Himmelsrichtungen** (horizontal), **hoch**, **runter**.
(Im Code: die Liste `_DIRS`.) „Diskret“ heißt: endlich viele feste Optionen – das
Netz gibt eine Wahrscheinlichkeit pro Option aus und würfelt daraus.

### Wie belohnt wird (die Reward-Funktion)
Pro Drohne und Schritt (Werte aus der YAML, Abschnitt `belohnung`):

| Term | Wert | Bedeutung |
| --- | --- | --- |
| `+ versorgte_kapazität` | 0.10 | **eigener** prioritätsgewichteter Deckungsbeitrag |
| `− funkloch` | 0.15 | **geteilte** Strafe pro unversorgtem Prio-Hotspot |
| `− absturz` | 0.50 | einmalig, wenn der Akku leer ist |
| `− kollision` | 0.30 | einmalig bei Hindernis-Kollision |
| `− akku_reserve` | bis 0.10 | je weiter unter der Sicherheitsreserve (35 %) |
| `− energie` | 0.00005 | winziger Dauer-Abzug (Energie kostet) |

Wichtige Idee: Der **Versorgungs-Reward ist pro Agent** (jede Drohne bekommt ihren
*eigenen* Beitrag), die **Funkloch-Strafe ist geteilt** (zwingt zur Kooperation).
Das verhindert „faule Drohnen“, die sich auf der Arbeit der anderen ausruhen.

### Das Funkmodell (der clevere Teil)
Eine Drohne versorgt einen Hotspot nur, wenn **beides** gilt:
1. Sie deckt ihn ab: Signalstärke `s(d) = max(0, 1 − (d/Reichweite)²) ≥ Schwelle`.
2. Sie ist **verbunden**: über eine Kette anderer Drohnen (**Mesh-Hops**) bis zum
   Uplink. Jeder Hop frisst Kapazität (`kapazitätsverlust_pro_hop`).

Die Verbindungstiefe wird in `_connectivity()` per **Breitensuche** über alle
Welten gleichzeitig berechnet. Hindernisse auf der Sichtlinie dämpfen das Signal
(`_blocking()`).

### Der Akku (warum das Lernen schwer ist)
- Verbrauch steigt mit Geschwindigkeit² und Wind (YAML-Formel).
- Laden nur nahe einer Ladestation und nur, wenn ein Platz frei ist
  (`_charging_mask()`).
- **Die Episode ist absichtlich viel länger als die Akku-Laufzeit.** Dadurch
  *muss* jede Drohne pro Runde mehrmals laden – nur so kann der Agent Laden
  überhaupt als nützlich entdecken.

### Das Curriculum (leicht → schwer)
Statt sofort die volle Schwierigkeit lernt der Schwarm in 4 Stufen:

| Stufe | Episode | Akku | Drohnen | Hotspots | Wind | Extras |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 12 min | 8 min | 6 | 4 | 0 | – |
| 2 | 20 min | 10 min | 10 | 3 | 0–5 | 2 Stationen |
| 3 | 30 min | 12 min | 12–20 | 5–8 | 0–10 | Ausfälle, wechselnder Bedarf |
| 4 | 40 min | 15 min | " | " | " | Hindernisse, Böen |

**Aufstieg**, sobald die gewichtete Abdeckung über 100 Updates im Schnitt > 90 %
liegt (`served_hist` in `train_netcov.py`).

---

## 7. Das neuronale Netz im Detail (`attn.py`)

Das Problem: Die Drohnenzahl schwankt (12–20, Ausfälle). Ein klassisches Netz hat
aber eine **feste** Eingabegröße. Lösung: **Attention** – ein Mechanismus, mit dem
jede Drohne „gewichtet auf die anderen schaut“, egal wie viele es sind. Tote
Drohnen werden per **Maske** ausgeblendet.

Aufbau von `AttnActorCritic`:

1. **`embed`**: wandelt die 7 Rohwerte jeder Drohne in einen `H=128`-dimensionalen
   Vektor um (ein kleines MLP).
2. **Attention (q, k, v)**: jede Drohne bildet aus den Embeddings zwei
   „Zusammenfassungen“ der anderen – einen Pool über Verbündete, einen über Gegner
   (`_masked_attention`). In der Netzabdeckung gibt es nur ein Team.
3. **Actor-Kopf**: `[eigenes Embedding + Ally-Pool + Enemy-Pool + self_global]`
   → 11 Logits → Wahrscheinlichkeiten über die Aktionen.
4. **Critic-Kopf**: aus gepoolten Embeddings + `team_global` → ein Wert.
   - `critic_value()` liefert **einen Wert pro Team**.
   - `agent_values()` liefert **einen Wert pro Drohne** – genau das nutzt
     `train_netcov.py` für die **lokale Kreditzuweisung** (jede Drohne bekommt
     ihren eigenen „War das gut?“-Maßstab).

„Permutationsinvariant“ heißt: Die Reihenfolge der Drohnen in der Liste ist egal –
das Netz liefert dasselbe Ergebnis. Genau richtig für einen Schwarm ohne feste
Rollen.

---

## 8. Ein Trainingsschritt, Schritt für Schritt (`train_netcov.py`)

Ein „Update“ (eine Runde Lernen) besteht aus zwei Phasen:

**Phase A – Erfahrung sammeln (Rollout, hier 32 Schritte):**
```
für r in 0..31:
    Actor wählt Aktionen für alle Drohnen in allen Welten  (ohne Lernen)
    Env führt sie aus → Belohnung, neue Beobachtung
    alles in Puffer-Tensoren (R, E, A, …) speichern
```

**Phase B – aus der Erfahrung lernen:**
```
1. compute_gae(): aus Belohnungen + Critic-Werten den ADVANTAGE ausrechnen
   („war diese Aktion besser oder schlechter als der Critic erwartet hat?“)
2. Advantage normieren (nur über lebende Drohnen)
3. 4 Epochen × 4 Minibatches:
      - Actor-Verlust mit PPO-Clip (verhindert zu große Schritte)
      - Critic-Verlust = MSE zwischen Wert-Schätzung und tatsächlichem Return
      - Entropie-Bonus (hält die Policy neugierig, gegen zu frühes Festfahren)
      - Gradienten, clip_grad_norm, opt.step()
4. Kennzahlen nach TensorBoard loggen
5. Curriculum prüfen: Abdeckung 100 Updates lang > 90 %? → nächste Stufe
```

Zwei bewusste Abweichungen vom Lehrbuch, die im Code kommentiert sind:
- **`gamma = 0.997`** statt 0.99 (länger in die Zukunft schauen, damit sich Laden
  „lohnt“, dessen Nutzen erst später kommt).
- **Kein Value-Clipping** im Critic (bei den unnormierten Wert-Zielen ließ das
  enge PPO-Clipping den Critic divergieren → einfacher maskierter MSE ist stabil).

---

## 9. Wie man alles startet

```powershell
# Umgebung aktivieren (einmal pro Terminal)
.venv\Scripts\activate

# Sandbox ohne ML anschauen
python run_sandbox.py live

# Netzabdeckung trainieren (neuer Lauf)
python train_netcov.py --updates 4000 --envs 1024 --name netcov_run16

# Training fortsetzen (Gewichte aus Checkpoint, Optimizer startet frisch)
python train_netcov.py --resume checkpoints\netcov_resume.pt --name netcov_run17

# Live zuschauen
python dashboard.py            # http://localhost:7070
tensorboard --logdir runs      # http://localhost:6006

# Ergebnis rendern
python show_netcov.py
```

Faustregel Dauer: erste Lernerfolge in 1–3 Stunden, über Nacht für gutes
Verhalten. `--envs` runter (z. B. 256) = weniger VRAM, langsameres Lernen.

---

## 10. Das Trainingslog lesen

Eine Zeile aus `out/netcov_run*.log`:
```
Update  475 | Stufe 1 | served 30.7% | conn 1.00 | holes 0.30 | crash 0.000 | batt 38% | 19,043 steps/s
```

| Feld | Bedeutung | Gut ist … |
| --- | --- | --- |
| `served` | gewichtete Abdeckung (Hauptziel) | **hoch** (→ 90 %+) |
| `conn` | Anteil mit Uplink verbundener Drohnen | hoch (→ 1.0) |
| `holes` | Anteil Hotspots ohne Versorgung | **niedrig** (→ 0) |
| `crash` | Abstürze pro Schritt (leerer Akku/Ausfall) | niedrig |
| `batt` | mittlerer Akkustand | stabil, nicht bei 0 |
| `steps/s` | GPU-Durchsatz | je höher, desto schneller lernt es |

---

## 11. Aktueller Stand & das offene Lade-Problem

Der Schwarm hat **Versorgen gelernt**, aber noch nicht das **gestaffelte Laden**.
Im Log sieht man einen **Sägezahn**:

```
served 30.9% | conn 1.00 | batt 38%   ← Akkus voll, gute Abdeckung
served 10.6% | conn 0.67 | batt 10%   ← Akkus sinken, Versorgung bricht ein
served  0.0% | conn 0.00 | batt  0%   ← ALLE gleichzeitig leer → Totalausfall
served 30.9% | …         | batt 38%   ← Episoden-Reset, von vorn
```

**Warum passiert das?** Die Drohnen optimieren den sofortigen Versorgungs-Reward
und fliegen, bis der Akku leer ist. Es fehlt der Anreiz, dass *immer nur einige*
laden gehen, während der Rest weiter versorgt. Das ist ein klassisches,
schwieriges RL-Problem: **verzögerte Belohnung** (Laden zahlt sich erst später
aus) + **Koordination** (wer geht wann?).

**Ideen für die nächste Session** (je eine als eigenes Experiment):
- Reward-Bonus dafür, dass *nie alle gleichzeitig* unter die Reserve fallen.
- Curriculum-Stufe 1 noch milder, damit Laden erst als nützlich entdeckt wird.
- Prüfen, ob die Reserve-Strafe stark genug ist oder vom Versorgungs-Reward
  übertönt wird.

---

## 12. Design-Entscheidungen & Warum (Interview-Gold)

Diese Punkte solltest du erklären können – sie zeigen echtes Verständnis:

- **MAPPO statt DQN:** Viele kooperierende Agenten mit geteilter Belohnung sind mit
  einem zentralen Critic + PPO stabiler als mit wertbasiertem DQN.
- **CTDE (zentraler Critic, dezentraler Actor):** Beim Training darf der Critic die
  Gesamtlage sehen; im Einsatz handelt jede Drohne nur lokal – realistisch und
  skalierbar.
- **Belohnung pro Agent, Funkloch geteilt:** löst das „Lazy-Agent“-Problem, ohne
  die Kooperation aufzugeben.
- **Attention statt fester Verkettung:** erlaubt wechselnde Schwarmgrößen und ist
  permutationsinvariant – passt zu Ausfällen/Nachschub.
- **Reward ~30× kleiner skaliert:** damit die Wert-Ziele in einer Größenordnung
  bleiben, die der Critic stabil lernen kann (sonst divergiert der Value-Loss).
- **Episode ≫ Akku-Laufzeit:** erzwingt mehrere Lade-Zyklen pro Runde – sonst ist
  Laden gar nicht lernbar.
- **`gamma = 0.997` & kein Value-Clipping:** gezielte, dokumentierte Abweichungen,
  um die verzögerte Lade-Belohnung lernbar zu machen bzw. den Critic stabil zu
  halten.
- **Alles aus einer Config-YAML:** reproduzierbare Experimente ohne Code-Änderung.
- **Vektorisiert auf der GPU:** tausende Welten parallel → genug Erfahrung in
  vernünftiger Zeit auf einer einzelnen Consumer-GPU.

---

## 13. Glossar

- **Agent:** Ein handelndes Wesen – hier eine Drohne.
- **Environment:** Die Simulation/Welt, in der Agenten handeln.
- **Policy:** Die Strategie „Beobachtung → Aktion“. Das ist das trainierte Netz.
- **Reward (Belohnung):** Zahl, die Verhalten bewertet. Steuert, was gelernt wird.
- **State / Beobachtung:** Was der Agent zu einem Zeitpunkt wahrnimmt.
- **Episode:** Eine Runde von Reset bis Reset.
- **Rollout:** Ein Stück gesammelte Erfahrung (hier 32 Schritte) vor einem Update.
- **Actor-Critic:** Netz mit zwei Köpfen – Actor wählt Aktionen, Critic bewertet.
- **PPO (Proximal Policy Optimization):** stabiler RL-Algorithmus; der „Clip“
  begrenzt, wie stark sich die Policy pro Update ändern darf.
- **MAPPO:** Multi-Agent-PPO – PPO für viele Agenten mit zentralem Critic.
- **CTDE:** Centralized Training, Decentralized Execution (s. o.).
- **GAE (Generalized Advantage Estimation):** Methode, den **Advantage** mit guter
  Balance aus Verzerrung und Rauschen zu schätzen.
- **Advantage:** „Wie viel besser/schlechter war diese Aktion als erwartet?“
- **gamma (Discount):** Wie stark zählt zukünftige Belohnung (0.99 ≈ ~100 Schritte
  Horizont, 0.997 ≈ länger).
- **Entropie / Entropie-Bonus:** Maß für „Zufälligkeit“ der Policy; der Bonus hält
  sie neugierig, damit sie nicht zu früh auf eine schlechte Taktik festfriert.
- **Attention:** Mechanismus, bei dem jedes Element gewichtet auf alle anderen
  schaut – funktioniert mit variabler Anzahl.
- **Boids:** Regelbasiertes Schwarmmodell (Separation, Alignment, Kohäsion) –
  die Nicht-ML-Policy aus Phase 1.
- **Mesh / Hop:** Funk-Kette zwischen Drohnen; ein „Hop“ = eine Weiterleitung.
- **Uplink:** Verbindung nach „draußen“ (Satellit/Netzknoten).
- **Curriculum:** Training von leicht nach schwer.
- **Self-Play:** Agenten trainieren gegen (frühere) Versionen ihrer selbst.
- **Tensor:** Mehrdimensionaler Zahlen-Block; die Datenstruktur von PyTorch.
- **Vektorisiert:** Als Matrix-Operation statt als Schleife geschrieben (GPU-schnell).
- **Checkpoint:** Gespeicherter Zwischenstand der Netz-Gewichte (`.pt`).

---

*Diese Datei wächst mit dem Projekt. Wenn sich Reward, Architektur oder Mission
ändern, hier und in `NOTES.md` nachziehen.*
