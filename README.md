# Schwarm-Simulation – Drohnenschwarm als Notfall-Funknetz

> **Work in Progress.** GPU-beschleunigte Multi-Agent-Arena, in der ein
> Drohnenschwarm per Reinforcement Learning lernt, ein fliegendes Funknetz
> über einer Krisenregion aufrechtzuerhalten – inklusive Akku-Management und
> rechtzeitigem Nachladen. Den Gesamtplan (7 Phasen) beschreibt
> `Schwarm-Simulation – Projektplan.md`.

Trainiert wird lokal auf einer RTX 4060 Ti mit **MAPPO** (Multi-Agent PPO) und
Self-Play. Die komplette Physik und Funkausbreitung laufen vektorisiert auf der
GPU, sodass **tausende Arenen parallel** simuliert werden (~19.000 Schritte/s).

> 📖 **Alles von Grund auf verstehen:** [`PROJEKT_ERKLAERT.md`](PROJEKT_ERKLAERT.md)
> erklärt das komplette Projekt – RL-Grundlagen, Architektur, jede Datei, die
> Mission, das neuronale Netz und die Design-Entscheidungen – in einfachen Worten.

## Aktueller Stand

Mehrere Umgebungen sind umgesetzt und trainierbar. Die Vision-Mission
**„Netzabdeckung"** (`env_netcov.py`) ist der aktuelle Schwerpunkt:

| Phase | Inhalt | Status |
| --- | --- | --- |
| 1 Sandbox | 2D-Physik (Schub, Trägheit, Luftwiderstand, Kollisionen), Boids-Schwarm, Viewer | ✅ |
| 2 Mesh-Netz | Funkradius, Gruppen, Mesh-Verbindungen | ✅ |
| 3 MAPPO + Dashboard | geteilter Actor / zentraler Critic, Self-Play, Live-Dashboard | ✅ |
| 4 Dynamik | Nachschub, Ausfälle, Attention für wechselnde Teamgrößen | ✅ |
| 5 3D | 3D-Such-Mission (`env_search3d.py`) | ✅ |
| 7 Missionen | Funk-Relais (`env_relay.py`), **Netzabdeckung (`env_netcov.py`)** | 🔄 aktiv |

**Offenes Problem (Trainingsstand):** Der Schwarm lernt die *Versorgung* gut,
aber noch nicht das rechtzeitige, gestaffelte *Nachladen*. Im Log zeigt sich
ein Sägezahn: Abdeckung steigt auf 30–80 %, fällt dann auf 0 %, wenn alle
Akkus gleichzeitig leer sind, und erholt sich erst beim Episoden-Reset.

6 Experimente mit Reward-Shaping und Curriculum (Staffel-Bonus, Reserve-Strafe,
mildere Stufe 1, Symmetrie-Features, Horizont/Gamma-Fix, Blackout-Strafe)
haben den Sägezahn bisher **nicht** gebrochen – bester Lauf (exp5, Horizont-Fix)
liegt bei ⌀ 11,2 % Abdeckung, Spitzen bis 79 %, Kollaps in ~44 % der Updates.
Reine Reward-Zusätze scheinen auf dem aktuellen Stufe-1-Setup ausgereizt;
nächster Kandidat ist eine gröbere Zeitskala der Simulation, damit ein
Lade-/Entlade-Zyklus überhaupt in den PPO-Horizont passt. Details in
`netzabdeckung_config.yaml` und `NOTES.md`.

## Setup

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
# PyTorch mit CUDA passend zur GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Nutzung

```powershell
# --- Sandbox (Phase 1, ohne ML) ---
python run_sandbox.py live                 # eine Arena live zuschauen
python run_sandbox.py record               # Runde aufzeichnen -> replays/*.npz
python run_sandbox.py bench --envs 4096    # GPU-Durchsatz messen

# --- Netzabdeckung trainieren (Phase 7, der aktuelle Fokus) ---
python train_netcov.py --updates 4000 --envs 1024 --name netcov_run16
python train_netcov.py --resume checkpoints\netcov_resume.pt --name netcov_run17

# --- Weitere Trainings ---
python train.py                 # Basis-MAPPO (Zonenkontrolle, 2 Teams)
python train_dynamic.py         # Dynamik: Nachschub/Ausfälle + Attention
python train_relay.py           # Funk-Relais-Mission
python train_search3d.py        # 3D-Such-Mission

# --- Anschauen / Auswerten ---
python show_netcov.py           # trainiertes Netzabdeckungs-Verhalten rendern
python -m swarm_sim.viewer replays\<datei>.npz

# --- Live-Dashboard (nur Python-Bordmittel) ---
python dashboard.py             # http://localhost:7070
tensorboard --logdir runs       # http://localhost:6006
```

## Wie es funktioniert

### Architektur

| Datei | Inhalt |
| --- | --- |
| `swarm_sim/config.py` | `SimConfig` + `TrainConfig` – alle Parameter / Hyperparameter an einem Ort |
| `swarm_sim/physics.py`, `physics3d.py` | vektorisierte 2D-/3D-Physik, Form `(E, A, 2/3)` |
| `swarm_sim/swarm.py` | `BoidsController` – regelbasiertes „Gehirn" (Phase 1) |
| `swarm_sim/mesh.py` | Funk-/Mesh-Netz: Verbindungen, Gruppen |
| `swarm_sim/mappo.py` | `MAPPO`: geteilter Actor + zentraler Critic, PPO-Update, GAE |
| `swarm_sim/attn.py` | `AttnActorCritic` – maskierte Attention für wechselnde Schwarmgrößen |
| `swarm_sim/env_*.py` | Umgebungen: Basis, dynamisch, relay, search3d, **netcov** |
| `swarm_sim/mission_config.py` | lädt `netzabdeckung_config.yaml` (nichts hart im Code) |
| `swarm_sim/recorder.py`, `viewer.py`, `metrics.py` | Replays, Viewer, Logging |
| `dashboard.py` | lokales Live-Dashboard (Beobachter + Starter) |

**Tensor-Konventionen:** `E` = parallele Arenen, `A` = Agenten pro Arena.
Zustand als `(E, A, …)`. Jede Policy hat dieselbe Schnittstelle: Beobachtung
rein, Aktion/Schub raus – so ließ sich die Boids-Regel später 1:1 durch eine
gelernte Policy ersetzen.

### Algorithmus: MAPPO mit Self-Play

- **Actor (dezentral):** sieht nur die lokale Beobachtung einer Drohne und wählt
  eine Aktion. *Ein* geteiltes Netz steuert alle Drohnen (Parameter-Sharing).
- **Critic (zentral):** sieht die Team-Gesamtsicht und schätzt einen Wert pro
  Team (Centralized Training, Decentralized Execution – Standard bei MAPPO).
- Vorteile über **GAE**, Update per **PPO-Clip**, geclippter Value-Loss.
- Für wechselnde Teamgrößen nutzt das Netz **maskierte Attention** über die
  *lebenden* Agenten (permutationsinvariant).

*Warum MAPPO und nicht DQN?* Kontinuierliche Kooperation vieler Agenten mit
geteilter Belohnung ist mit einem zentralen Critic + PPO deutlich stabiler als
mit wertbasierten Verfahren wie DQN.

### Mission „Netzabdeckung" (env_netcov.py)

- **Beobachtung je Drohne (7+13 Werte):** eigene Position/Höhe/Geschwindigkeit/
  Akku; dazu Richtung & Distanz zu Uplink, Ladestation und nächster
  Versorgungslücke, Verbindungsstatus und Wind.
- **Aktionen (11 diskret):** stehen, 8 Horizontalrichtungen, hoch, runter.
- **Reward (pro Agent, gegen „faule" Agenten):** + eigener prioritäts­gewichteter
  Deckungsbeitrag, − geteilte Funkloch-Strafe, − Absturz (leerer Akku),
  − Kollision, − Unterschreitung der Akku-Reserve, − Energie.
- **Funkmodell:** Signalstärke `s(d)=max(0, 1-(d/reichweite)²)`, Mesh-Hops zum
  Uplink, Kapazitätsverlust pro Hop; Hotspots gelten erst als versorgt, wenn die
  Drohne sie abdeckt **und** über die Funkkette mit dem Uplink verbunden ist.
- **Curriculum:** 4 Stufen von leicht (kurze Episoden, großer Funkradius) nach
  schwer (Wind, Böen, Hindernisse, bewegliche Nachfrage). Aufstieg bei
  Abdeckung > 90 % über 100 Episoden.

## Hardware

Windows + RTX 4060 Ti, Ryzen 9, 32 GB RAM. Training läuft lokal; erste
Lernerfolge in 1–3 Stunden, gutes Verhalten über Nacht.

## Lizenz

[Apache License 2.0](LICENSE).
