# Lern-Log – Schwarm-Simulation

Format pro Session: Datum · Was gemacht · Entscheidungen & Warum · Probleme &
Lösungen · Offene Fragen.

---

## 2026-10-21 – Repo aufgesetzt, Stand dokumentiert

**Was gemacht**
- Git-Repo initialisiert, `.gitignore` (schließt `.venv/`, `checkpoints/*.pt`,
  `runs/`, Logs, Replays aus), Apache-2.0-Lizenz, README auf den echten Stand
  gebracht (behauptete vorher „Phase 1", tatsächlich bis Phase 7).
- Diese `NOTES.md` rückwirkend aus Config und Trainingslogs rekonstruiert.
- Pfad-Bug in `resume_training.bat` gefixt (zeigte auf einen alten Ordnernamen).

**Entscheidungen & Warum**
- Lizenz Apache 2.0 (vom Nutzer gewünscht; gibt zusätzlich expliziten
  Patentschutz gegenüber MIT).
- Modelle/Checkpoints kommen NICHT ins Repo (zu groß) → später Hugging Face Hub.

**Offene Fragen / nächster Schritt**
- Das Lade-Problem (siehe unten) angehen: Reward/Curriculum so, dass der Schwarm
  rotierendes Nachladen lernt, statt kollektiv leerzulaufen.

---

## 2026-10-11 bis 10-12 – Netzabdeckungs-Mission, Runs 1–15 (rekonstruiert)

**Was gemacht**
- Vollmodell `env_netcov.py` gebaut: 3D-Kinematik, Funkausbreitung mit
  Mesh-Hops zum Uplink, Hotspot-Bedarf nach Typ/Priorität, Ladestationen,
  Wind/Böen, Hindernis-Dämpfung – alles aus `netzabdeckung_config.yaml`.
- MAPPO mit Attention-Netz (`attn.py`) für variable Drohnenzahl (12–20).
- 15 Trainingsläufe über ~2 Tage (`out/netcov_run*.log`), Checkpoints bis
  Update 4000, Resume-Mechanismus (`netcov_resume.pt`).

**Entscheidungen & Warum (aus den Config-Kommentaren)**
- Reward ~30× kleiner skaliert, damit der Critic bei gamma 0.99 stabil fittet
  (vorher divergierender Value-Loss).
- Reward PRO AGENT statt einer geteilten Team-Belohnung → gegen „faule" Agenten
  (jede Drohne bekommt eigenen Deckungsbeitrag; Funkloch bleibt geteilt).
- run9: Akku-Reserve-Strafe stärker + Reserve-Schwelle höher (35 %), damit das
  Lade-Signal kommt, solange die Station noch erreichbar ist.
- run13: Ladestation direkt beim Hotspot → Drohne lädt beim Versorgen und
  „entdeckt" das Laden; später schrittweise weiter weg.
- run15: Funkradius kleiner + mehrere verteilte Hotspots → eine Drohne reicht
  nicht mehr, echter Schwarm-Druck.
- Episodenlänge bewusst ≫ Akku-Laufzeit, damit pro Episode mehrmals geladen
  werden MUSS (sonst ist Laden gar nicht lernbar).

**Problem (noch offen)**
- Sägezahn im Log: `served` steigt auf ~30 % (Akku voll), fällt auf 0 %, sobald
  alle Akkus gleichzeitig leer sind (`batt 0 %`, `conn 0`), erholt sich erst
  beim Reset. Der Schwarm lernt Versorgen, aber nicht das gestaffelte Nachladen.

**Offene Fragen**
- Braucht es einen expliziten Anreiz für *gestaffeltes* Laden (z. B. Bonus, wenn
  nie alle gleichzeitig unter Reserve sind)?
- Ist die Reserve-/Heimkehr-Strafe stark genug, oder überdeckt der
  Versorgungs-Reward sie?
