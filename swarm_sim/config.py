"""Zentrale Konfiguration der Simulation.

Ein einziges Dataclass haelt alle Parameter von Arena, Physik und
Schwarmverhalten. So laesst sich jede Runde reproduzierbar beschreiben und
spaeter (Phase 3) direkt als Teil des RL-Environments weiterreichen.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class SimConfig:
    # --- Arena ------------------------------------------------------------
    arena_w: float = 100.0          # Breite der Arena (Weltkoordinaten)
    arena_h: float = 100.0          # Hoehe der Arena

    # --- Batch / Agenten --------------------------------------------------
    num_envs: int = 1               # parallele Arenen auf der GPU
    num_agents: int = 30            # Agenten pro Arena
    num_teams: int = 2              # Teams (Phase 1 nur zur Einfaerbung)

    # --- Physik -----------------------------------------------------------
    dt: float = 0.05                # Zeitschritt pro Simulationsschritt (s)
    mass: float = 1.0               # Masse je Agent (Traegheit)
    radius: float = 1.0             # Kollisionsradius je Agent
    max_thrust: float = 20.0        # maximale Schubkraft (|F|)
    max_speed: float = 25.0         # Geschwindigkeits-Obergrenze
    drag_coef: float = 0.02         # quadratischer Luftwiderstand c_d
    wall_restitution: float = 0.6   # Energieerhalt beim Wandabprall (0..1)
    collision_restitution: float = 0.4  # Energieerhalt bei Agent-Kollision

    # --- Schwarmverhalten (Boids) ----------------------------------------
    perception_radius: float = 15.0   # Sichtweite fuer Nachbarn
    separation_radius: float = 4.0    # ab hier aktiv abstossen
    w_separation: float = 2.0         # Gewicht Abstand halten
    w_alignment: float = 1.0          # Gewicht Richtung angleichen
    w_cohesion: float = 0.8           # Gewicht zur Gruppenmitte
    w_goal: float = 0.6               # Gewicht Richtung Teamziel
    w_wander: float = 0.3             # Gewicht Zufallsdrift

    # --- Mesh-Netz (Phase 2) ---------------------------------------------
    radio_radius: float = 18.0        # Funkreichweite je Agent
    mesh_same_team: bool = True       # Verbindungen nur zwischen Teamkollegen

    # --- Zonenkontrolle (Phase 3) ----------------------------------------
    zone_radius: float = 22.0         # Radius des Zielbereichs (Arenamitte)
    episode_len: int = 128            # Schritte pro Runde, dann Reset

    # --- Dynamik (Phase 4) -----------------------------------------------
    start_active_per_team: int = 3    # anfangs aktive Agenten je Team (Rest Reserve)
    combat_radius: float = 9.0        # Nahkampf-Reichweite fuer Zerstoerung
    destroy_margin: int = 2           # Gegner - Verbuendete >= Marge -> Ausfall
    spawn_interval: int = 20          # Schritte zwischen Nachschub-Aktivierungen

    # --- 3D-Such-Mission (Phase 5) ---------------------------------------
    arena_d: float = 40.0             # Hoehe der Box-Arena (Flughoehe)
    min_altitude: float = 4.0         # Drohnen bleiben ueber dem Boden
    grid_size: int = 20               # Suchraster: grid_size x grid_size Bodenzellen
    num_targets: int = 6              # versteckte Ziele je Arena
    footprint_base: float = 3.0       # Kamera-Fussabdruck am Boden (Grundradius)
    footprint_alt_gain: float = 0.35  # zusaetzlicher Radius je Hoeheneinheit
    fail_prob: float = 0.002          # Wahrscheinlichkeit zufaelliger Ausfall je Drohne/Schritt
    bird_count: int = 3               # gleichzeitige Voegel je Arena
    bird_speed: float = 18.0          # Fluggeschwindigkeit der Voegel
    bird_danger_radius: float = 5.0   # Drohne in diesem Umkreis wird getroffen

    # --- Krisen-Relais / Netzabdeckung (Phase 7) -------------------------
    # Drohnen halten ein Funknetz ueber die Flaeche. Eine Bodenzelle gilt als
    # VERSORGT, wenn eine Drohne sie abdeckt UND diese Drohne ueber die
    # Funk-Kette mit der Basisstation verbunden ist. Akku zwingt zum
    # rechtzeitigen Abloesen (nachladen an der Basis), ohne Versorgungsluecke.
    charge_radius: float = 10.0       # Umkreis der Basis, in dem geladen wird
    battery_drain: float = 0.0035     # Grundverbrauch je Schritt (Schweben)
    battery_move_drain: float = 0.012 # zusaetzlicher Verbrauch bei Vollgas
    battery_charge: float = 0.06      # Ladung je Schritt nahe der Basis
    battery_start_min: float = 0.5    # Start-Akku min (zufaellig gestaffelt)
    num_demand: int = 5               # Bevoelkerungs-Hotspots (hoeheres Gewicht)
    demand_sigma: float = 14.0        # Ausdehnung eines Hotspots (Weltkoord.)
    empty_penalty: float = 2.0        # Strafe je Drohne, die im Feld leerlaeuft
    # Dynamische Nachfrage: Hotspots bewegen sich / aendern Staerke waehrend
    # der Runde (Menschen sammeln sich woanders, Lage aendert sich live).
    demand_dynamic: bool = True
    demand_move_speed: float = 7.0    # Driftgeschwindigkeit der Hotspots (Welt/s)
    demand_intensity_jitter: float = 0.04  # zufaellige Staerkeschwankung je Schritt
    demand_relocate_interval: int = 45     # Schritte, dann springt 1 Hotspot neu
    # Reward-Shaping + Curriculum
    served_reward_scale: float = 25.0 # Gewicht der versorgten (gewichteten) Flaeche
    low_batt_threshold: float = 0.35  # ab hier "Heimkehr"-Anreiz
    w_homecome: float = 1.2           # Staerke des Heimkehr-Anreizes bei Low-Akku
    curriculum: bool = True           # leicht -> schwer ueber das Training
    curriculum_frac: float = 0.6      # Anteil des Trainings zum Hochrampen
    radio_easy_mult: float = 1.6      # Funkradius am Anfang groesser (leichter)
    batt_easy_mult: float = 0.45      # Akkuverbrauch am Anfang milder

    # --- Technik ----------------------------------------------------------
    device: str = "cuda"            # "cuda" oder "cpu"
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    """Hyperparameter fuer MAPPO (Phase 3)."""

    total_updates: int = 2000         # Anzahl PPO-Updates
    rollout_len: int = 32             # Schritte pro Rollout vor einem Update
    lr: float = 3e-4
    gamma: float = 0.99               # Discount
    gae_lambda: float = 0.95          # GAE
    clip: float = 0.2                 # PPO-Clip
    epochs: int = 4                   # Optimierungs-Epochen pro Rollout
    minibatches: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128                 # MLP-Breite

    log_every: int = 1                # Updates zwischen TensorBoard-Logs
    ckpt_every: int = 100             # Updates zwischen Checkpoints
    ckpt_dir: str = "checkpoints"

    def as_dict(self) -> dict:
        return asdict(self)
