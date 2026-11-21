# Mitmachen

Dieses Projekt ist öffentlich, weil das offene Problem (gestaffeltes Nachladen
statt Kollektiv-Blackout, siehe README "Aktueller Stand" und `NOTES.md`) genau
die Art von Experiment ist, bei der mehr ausprobierte Reward-/Curriculum-Ideen
schneller zu einer Lösung führen. Wenn du trainierst und etwas findest, das
den Sägezahn bricht oder zumindest etwas über das Problem lernt – her damit.

## So trainierst du selbst weiter

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python train_netcov.py --updates 3000 --envs 1024 --name mein_experiment
python show_netcov.py   # Verhalten ansehen
```

Config/Reward-Gewichte stehen in `netzabdeckung_config.yaml`, nicht im Code.
Die aktuellen Metriken (`served`, `conn`, `holes`, `crash`, `batt`, `ready`)
werden pro Update ins Log geschrieben – daran erkennst du den Sägezahn.

## main ist geschützt – bitte per Branch + Pull Request

Direkte Pushes auf `main` sind nicht möglich. Ablauf:

1. Fork oder (bei Schreibrechten) eigenen Branch anlegen: `git checkout -b exp7-<kurzname>`
2. Trainieren, Config-Änderung committen.
3. Ergebnis in `NOTES.md` eintragen – gleiches Format wie die bisherigen
   `exp*`-Einträge: Was geändert wurde, Zahlen (served ⌀, Spitze,
   Kollaps-Anteil), Fazit. **Keine Checkpoints, Logs oder `runs/` committen**
   (siehe `.gitignore`) – nur Config, Code, Notizen.
4. Pull Request öffnen, Template ausfüllen.

Ziel ist, dass jedes eingereichte Experiment nachvollziehbar und vergleichbar
bleibt – nicht nur "funktioniert bei mir", sondern mit Zahlen gegen die
bisherigen Läufe in `NOTES.md`.

## Was nicht gebraucht wird

- Keine Refactorings/Architektur-Änderungen ohne vorherige Absprache (Issue
  aufmachen).
- Keine neuen Abhängigkeiten ohne kurze Begründung im PR.
- Keine Modelle/Checkpoints im Repo (zu groß – später ggf. Hugging Face Hub).

## Lizenz

Beiträge stehen unter der [Apache License 2.0](LICENSE) wie der Rest des Repos.
