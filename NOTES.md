# Lern-Log – Schwarm-Simulation

Format pro Session: Datum · Was gemacht · Entscheidungen & Warum · Probleme &
Lösungen · Offene Fragen.

---

## 2026-10-21 (abends) – Auto-Loop: Experimente gegen den Sägezahn

Ziel: der Schwarm soll gestaffeltes Nachladen lernen (nicht alle Akkus
gleichzeitig leer). Vorab freigegebene Experiment-Leiter, der Loop arbeitet sie
ab und misst den Sägezahn objektiv (Skript: Anteil Updates mit kollektiv leerem
Akku + Versorgungs-Kollaps, mittlere Abdeckung in der 2. Lauf-Hälfte).

Baseline-Referenz: run15 – `served` schwingt 0–30 %, Akku kippt regelmäßig auf
0 %, Erholung erst beim Reset.

**exp1 – Staffel-Bonus** (`staffel_bonus: 0.04`, geteilt, × Anteil Drohnen über
Reserve). 1000 Updates, Stufe 1, frischer Start. → **gescheitert, Sägezahn bleibt.**
- served (2. Hälfte) im Schnitt nur 5,6 %, Kollaps in 62 % der Updates, Akku weiter 0 %.
- `ready` ≈ 0,06: der Schwarm hält fast nie Reserven → Bonus wird kaum verdient,
  zu wenig Gradient. Zeitweise Park-Falle (Akku ~90 %, aber served ≈ 0, Überlebende
  parken geladen). Warum: der Bonus belohnt den *Zustand* „geladen", erzeugt aber
  keinen Anreiz für den *Weg* dahin (rechtzeitig heimfliegen), solange Versorgen
  kurzfristig mehr zahlt.

**exp2 – stärkere Reserve-Strafe + höhere Schwelle** (Staffel-Bonus aus;
`akku_reserve` -0.1 → -0.2, `sicherheitsreserve_prozent` 35 → 45). →
**gescheitert, Sägezahn bleibt.** served (2. Hälfte) nur 4,2 %, Kollaps in 67 %.
Neuer Charakter: Leer-Abstürze seltener (`frac_batt_empty` 0,05), aber Akku steigt
in der Kollaps-Phase auf 89–100 % bei served ≈ 0 → **alle parken gleichzeitig
voll an der Station**, niemand versorgt. Die stärkere Strafe hat die Park-Falle
verschärft, nicht gestaffelt.

**Diagnose nach exp1+exp2:** Das Problem ist Symmetrie, nicht Strafhöhe. Eine
geteilte Policy (Parameter-Sharing) + synchroner Episoden-Reset → alle Drohnen
tun gleichzeitig dasselbe (erst alle versorgen bis leer, dann alle laden). Es
fehlt ein Mechanismus zur Aufteilung „du lädst / du versorgst".

**exp3 – milderes Stufe-1** (Reward zurück auf Baseline: `akku_reserve` -0.1,
`sicherheitsreserve_prozent` 35, Staffel-Bonus 0; Curriculum `stufe_1.hotspots`
4 → 1). Station steht `beim_hotspot` → dieselbe Drohne kann versorgen UND laden,
der Ladezyklus wird entdeckbar; spätere Stufen ziehen die Hotspots wieder hoch.
Testet NUR die Curriculum-Änderung. → **gescheitert, Sägezahn bleibt** — aber am
wenigsten schlimm: served-Spitzen bis 70 %, Kollaps „nur" in 38 % (exp1 62 %,
exp2 67 %). Die Aufgabe ist also klar lernbar; der Schwarm läuft trotzdem
kollektiv leer.

**Fazit nach der Leiter (exp1–exp3):** Reward-Shaping (Bonus/Strafe) und
Vereinfachung (Curriculum) lösen den Sägezahn NICHT. Bestätigte Ursache:
**Symmetrie ohne Rollenaufteilung.** Eine geteilte Policy + fast gleiche
Beobachtungen ⇒ alle Drohnen entscheiden gleich ⇒ alle versorgen bis leer, dann
alle laden. Es fehlt lokale Information, an der sich die *identische* Policy
*unterschiedlich* verhalten kann.

**Nächste Richtung (braucht Okay, da Beobachtungs-/Reward-Entscheidung):**
- A) Symmetrie brechen über die Beobachtung: pro Drohne z. B. „mein Akku-Rang im
  Team" / „Anteil Team lädt gerade" / „Anteil Team unter Reserve". Dann lernt die
  geteilte Policy „bin ich unter den Schwächsten → laden, sonst versorgen"
  (gestaffelte Rotation mit *einem* Netz, klassischer MARL-Fix).
- B) Reward auf *Kontinuität*: den Einbruch der Abdeckung direkt bestrafen
  (z. B. Minimum der Abdeckung über ein Zeitfenster), statt nur Pro-Schritt-Werte.
- Loop hier gestoppt: Leiter abgearbeitet, weitere Hebel sind neue Entscheidungen.

**exp4 – Symmetrie über Beobachtung brechen** (freigegeben nach der ersten Leiter):
jede Drohne sieht jetzt Akku-Rang im Team, Anteil Team lädt, Anteil Team unter
Reserve (`self_global_dim` 13 → 16). → **gescheitert, Sägezahn bleibt** (served
5,9 %, Kollaps 52 %, Akku → 0, `ready` ~0,06). Kein Curriculum-Aufstieg.

**DURCHBRUCH bei der Diagnose (eigentliche Ursache):** Beim Vergleich aller Läufe
fiel die Periode auf: der „Sägezahn" wiederholt sich alle **~225 Updates** (run15:
Erholung bei 700, 925, 1150, 1375). Das ist KEIN Trainingsrauschen, sondern **genau
eine Episode**: `ep_steps = 12 min · 60 / dt(0.1s) = 7200 Schritte`, `rollout_len =
32` → `7200 / 32 = 225 Updates pro Episode`. Der per-Update geloggte „Sägezahn" ist
also EINE Episode, die abläuft: voller Akku beim Reset → versorgen → nach ~8 min
(Akku-Laufzeit Stufe 1) kollektiv leer → kein Mid-Episode-Recovery → Reset.
- Warum lädt niemand rechtzeitig nach? **Zeitskalen-Mismatch mit gamma.** gamma
  0.997 → Kredit-Horizont ~1/(1−γ) = 333 Schritte = **~33 s**. Ein Lade-Zyklus
  (hinfliegen, laden, später profitieren) dauert **Minuten** (Akku 8 min = 480 s).
  Der diskontierte Nutzen des Ladens liegt ~15× jenseits des Horizonts → für den
  Agenten quasi unsichtbar. Darum konnte KEIN Reward-Bonus/-Strafe und KEINE
  Beobachtung es lösen: das Problem ist die Kredit-Zuweisung über die Zeit, nicht
  der Anreiz oder die Information.
- Kandidaten für die nächste Leiter (brauchen Okay): (1) gamma deutlich höher
  (0.999 → ~100 s, 0.9995 → ~200 s), ggf. längere Rollouts; (2) Zeitskala der Sim
  gröber (größeres dt / weniger Schritte pro Minute), damit der Zyklus in den
  Horizont passt; (3) Kontinuitäts-Reward (Einbruch bestrafen) als Ergänzung.

---

## 2026-10-21 (nachts) – Voll-autonomer Loop (Nutzer: "nichts mehr machen, Recap am Ende")

Loop auf voll autonom umgestellt: längere Läufe, Entscheidungen ohne Rückfrage,
committen + pushen nach jedem Experiment, Recap am Ende.

**exp5 – Horizont-Fix** (`gamma` 0.997 → 0.9995 ≈ Horizont 33 s → 200 s;
`rollout_len` 32 → 64; exp4-Obs bleiben). 2000 Updates. → **bester Lauf, aber
Sägezahn bleibt.** served ⌀ 11,3 % (Bestwert), Spitzen 79 %, `frac_batt_empty`
0,05, Kollaps noch 44 %. Deutung: Horizont ~200 s half, ist aber < Akku 480 s →
Laden immer noch zu weit weg. gamma allein reicht nicht (höher = zu viel Varianz).

**exp6 – Kontinuitäts-/Blackout-Reward** (zusätzlich zum exp5-Setup): harte
geteilte Strafe, wenn Team-Abdeckung unter Schwelle fällt (`blackout_strafe`,
Hinge unter `blackout_schwelle`). Macht den Totalausfall teuer statt nur einzelne
Funklöcher → direkter Druck, die Abdeckung NIE ganz einbrechen zu lassen = gestaffelt
laden. 3000 Updates. → Lauf gestartet.

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
