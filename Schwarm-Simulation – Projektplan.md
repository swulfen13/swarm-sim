# Schwarm-Simulation – Projektplan

Oct 11, 2026 · @Swulfen

## Ziel und Rahmen

Wir bauen ein Spiel bzw. eine Simulation, in der mehrere Schwärme gegeneinander antreten und per Reinforcement Learning selbst Taktiken lernen.

- Mehrere Teams, Agenten kommen dazu, fallen aus oder werden "zerstört"
- Schwärme teilen sich auf und finden sich wieder, das Mesh-Netz baut sich selbst neu
- Spielphysik in einer virtuellen Arena, kein reales Flugmodell
- Keine Anbindung an echte Drohnen, Flugcontroller oder reale Zielerfassung

## Hardware und Technik

Die RTX 4060 Ti mit Ryzen 9 und 32 GB DDR5 reicht für das ganze Projekt.

| Baustein | Werkzeug |
| --- | --- |
| Sprache | Python |
| ML und GPU | PyTorch mit CUDA |
| Physik | Eigene Spielphysik in PyTorch, tausende Arenen parallel auf der GPU |
| Umgebung | PettingZoo-Schnittstelle (Standard für Multi-Agent) |
| Training | MAPPO mit Self-Play |
| Viewer | Ursina oder Browser mit three.js für Replays |

## Phasen

Jede Phase liefert etwas Sichtbares, bevor die nächste startet.

- [ ] **1. Sandbox ohne ML:** Arena, Physik (Schub, Trägheit, Luftwiderstand, Kollisionen), Schwarmverhalten nach einfachen Regeln, Viewer
- [ ] **2. Mesh-Netz:** Funkradius pro Agent, Gruppen berechnen sich bei jeder Änderung neu, Anführer-Wahl pro Gruppe
- [ ] **3. Erstes ML + Dashboard:** 2 Teams mit je 5 Agenten in 2D, gemeinsames Gehirn pro Team (MAPPO), Self-Play; Monitoring-Dashboard läuft ab dem ersten Training mit
- [ ] **4. Dynamik:** Nachschub, Ausfälle und Zerstörung während der Runde, Attention für wechselnde Teamgrößen
- [ ] **5. Ausbau:** 3D, 3–4 Teams, Hindernisse, Curriculum von leicht nach schwer, Liga aus alten Versionen
- [ ] **6. Showcase:** Replays der besten Matches, Analyse der entstandenen Taktiken
- [ ] **7. Karten und Missionen:** Gelände mit Hindernissen, Karten-Editor und Zufallskarten, Missionen wie Personensuche, Kartieren, Funk-Relais und Lieferung

## Größenordnung und Zeitaufwand

Im Training sind etwa 3–4 Teams mit je 10–20 Agenten realistisch, zum Zuschauen auch mehr.

Reine Rechenzeit auf der RTX 4060 Ti, grob geschätzt. Die Entwicklungszeit kommt dazu: realistisch einige Wochen bis wenige Monate an Abenden und Wochenenden.

| Stufe | Trainingszeit |
| --- | --- |
| Erste Lernerfolge in 2D | 1–3 Stunden |
| Gutes Teamverhalten, 2 Teams in 2D | eine Nacht |
| Ausfälle, Nachschub, wechselnde Teamgrößen | 1–2 Nächte |
| 3D mit 3–4 Teams und Hindernissen | 2–4 Tage |
| Eine Mission (z. B. Personensuche) auf einer Karte | 1–2 Nächte |
| Alle Missionen auf Zufallskarten | 1–2 Wochen |

## Fortschritt verfolgen

Trainingsfortschritt zeigt sich in Kurven, Wertungen und Replays.

- **Live-Kurven:** TensorBoard oder Weights & Biases zeigen Belohnung, Siegquote und Überlebensrate pro Team, auch am Handy
- **Elo-Wertung:** Jede gespeicherte Version tritt gegen ältere an und bekommt eine Wertung wie im Schach
- **Replays:** Alle paar Stunden automatisch ein Replay vom aktuellen Stand, ideal für Vorher-nachher-Vergleiche
- **Checkpoints:** Regelmäßig gespeicherte Zwischenstände, jederzeit im Viewer ladbar

### Monitoring-Dashboard

Ein Dashboard im Browser zeigt alle Trainingswerte live und ist auch vom Handy aus erreichbar. Grundlage ist Weights & Biases oder TensorBoard, ergänzt um eigene Ansichten für Replays und Missionen.

| Bereich | Was angezeigt wird |
| --- | --- |
| Loss | Policy-Loss, Value-Loss, Entropie (wie zufällig die Agenten noch handeln) |
| Fortschritt | Belohnung pro Team, Siegquote, Überlebensrate, Elo-Verlauf |
| Missionen | Erfolgsquote, gefundene Personen, abgesuchte Fläche, Zeit bis zum Fund |
| Schwarm | Gruppenanzahl, Mesh-Verbindungen, Ausfälle und Neubildungen |
| System | Trainingsschritte pro Sekunde, GPU-Auslastung, VRAM, Temperatur, Restzeit |
| Replays | Neueste Replay-Videos und Vergleich alter mit neuen Versionen |

Warnungen per Benachrichtigung, wenn das Training hängt oder die Kurven abstürzen.

## Karten und Missionen

In Phase 7 bekommt die Arena echtes Gelände und die Schwärme bekommen Aufgaben jenseits des Kampfs.

- **Gelände:** Stadt, Wald, Gebirge, Trümmerfeld mit Hindernissen, die Sicht und Funk blockieren
- **Karten:** Editor für eigene Maps und zufällig generierte Karten, damit nichts auswendig gelernt wird
- **Wetter:** Wind und Nebel als Zusatzschwierigkeit
- **Personensuche:** Vermisste finden, Gebiet aufteilen, keine Bereiche doppelt absuchen
- **Kartieren:** Unbekanntes Gebiet möglichst schnell erfassen
- **Funk-Relais:** Einige Agenten halten die Verbindung für die Suchenden
- **Lieferung:** Hilfsgüter zu Zielen bringen, mit begrenztem Akku
- **Gemischt:** Teams suchen um die Wette oder arbeiten zusammen

## Was man damit erreichen kann

Das Projekt bringt vor allem echtes Know-how in Multi-Agent RL und ein starkes Vorzeigestück.

- **Lernen:** Reinforcement Learning, Self-Play, GPU-Simulation und verteilte Systeme praktisch verstehen
- **Portfolio:** Ein visuell beeindruckendes Projekt für GitHub und die eigene Website
- **Content:** Videos, in denen Schwärme über Trainingsstunden Taktiken entwickeln, kommen auf YouTube und TikTok gut an
- **Spiel:** Daraus kann ein eigenes Strategie- oder Zuschauerspiel werden
- **Zivile Anwendungen:** Dieselben Prinzipien (Selbstorganisation, Mesh-Netz, Ausfallsicherheit) stecken in Suche und Rettung, Lichtshows, Landwirtschaft und Lagerrobotern
