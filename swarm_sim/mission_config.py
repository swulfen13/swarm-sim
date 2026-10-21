"""Lader fuer die Missions-Config (netzabdeckung_config.yaml).

Leitregel aus der Datei: "Alles als Parameter halten, nichts hart im Code."
Dieses Modul liest die YAML und stellt die Werte so bereit, dass das
Environment pro Episode aus den angegebenen BEREICHEN zufaellig sampeln kann
([lo, hi] -> gleichverteilt) und das 4-stufige Curriculum angewendet wird.

Nichts hier trifft inhaltliche Annahmen ueber Zahlen - die kommen alle aus
der YAML. Fehlt ein Schluessel, gibt es einen klaren Fehler bzw. Default.
"""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_PATH = "netzabdeckung_config.yaml"


def load_raw(path: str | Path = DEFAULT_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def as_range(v) -> tuple[float, float]:
    """Macht aus einem Skalar oder [lo, hi] immer ein (lo, hi)-Paar."""
    if isinstance(v, (list, tuple)):
        return float(v[0]), float(v[1])
    return float(v), float(v)


class MissionConfig:
    """Haelt die rohe YAML und liefert Stufen-/Bereichswerte.

    stage (1..4) waehlt die Curriculum-Stufe; stage-Overrides ueberschreiben
    die Basiswerte (z. B. weniger Drohnen, kein Wind in Stufe 1).
    """

    def __init__(self, raw: dict, stage: int = 1):
        self.raw = raw
        stage_keys = [k for k in raw.get("curriculum", {}) if str(k).startswith("stufe_")]
        self.num_stages = max(1, len(stage_keys))
        self.set_stage(stage)

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH, stage: int = 1) -> "MissionConfig":
        return cls(load_raw(path), stage=stage)

    # ------------------------------------------------------------------ #
    def set_stage(self, stage: int) -> None:
        self.stage = max(1, min(stage, self.num_stages))
        self.ov = self.raw.get("curriculum", {}).get(f"stufe_{self.stage}", {}) or {}

    # ------------------------------------------------------------------ #
    def _ov(self, key, fallback):
        """Stufen-Override falls vorhanden, sonst Basiswert."""
        return self.ov[key] if key in self.ov else fallback

    # --- bequeme Zugriffe mit Curriculum-Overrides -------------------- #
    def map_size_m(self) -> tuple[float, float]:
        km = self.raw["simulation"]["karte"]["groesse_km"]
        return float(km[0]) * 1000.0, float(km[1]) * 1000.0

    def altitude_m(self) -> tuple[float, float]:
        return as_range(self.raw["simulation"]["karte"]["flughoehe_m"])

    def control_dt(self) -> float:
        return 1.0 / float(self.raw["simulation"]["steuerung_hz"])

    def episode_minutes(self) -> float:
        # Episodenlaenge ist bewusst von der Akku-Laufzeit ENTKOPPELT, damit eine
        # Episode mehrere Lade-Zyklen umfasst - sonst kann Laden nie gelernt werden.
        return float(self._ov("episoden_min",
                     self._ov("flugdauer_min", self.raw["akku"]["flugdauer_schwebeflug_min"])))

    def battery_minutes(self) -> float:
        """Reale Schwebeflug-Dauer eines vollen Akkus (unabhaengig von der Episode)."""
        return float(self._ov("akku_flugdauer_min", self.raw["akku"]["flugdauer_schwebeflug_min"]))

    def charge_minutes(self) -> float:
        """Dauer fuer 0->100 %, je nach Lademodus (voll / schnell / tausch)."""
        st = self.raw["ladestation"]
        modus = str(st.get("lademodus", "voll")).lower()
        if modus == "tausch":
            return float(st["akkutausch_min"])
        if modus == "schnell":
            return float(st["schnellladung_20_auf_80_min"]) * (100.0 / 60.0)  # 20->80% hochgerechnet
        return float(st["ladezeit_0_auf_100_min"])

    def drones_range(self) -> tuple[int, int]:
        lo, hi = as_range(self._ov("drohnen", self.raw["drohne"]["anzahl_training"]))
        return int(round(lo)), int(round(hi))

    def hotspots_range(self) -> tuple[int, int]:
        lo, hi = as_range(self._ov("hotspots", self.raw["bedarf"]["hotspots_anzahl"]))
        return int(round(lo)), int(round(hi))

    def stations_range(self) -> tuple[int, int]:
        lo, hi = as_range(self._ov("stationen", self.raw["ladestation"]["anzahl"]))
        return int(round(lo)), int(round(hi))

    def wind_ms(self) -> tuple[float, float]:
        return as_range(self._ov("wind_ms", self.raw["umwelt"]["wind_ms"]))

    def gust_enabled(self) -> bool:
        return bool(self._ov("boeen", "boeen_ms" in self.raw["umwelt"]))

    def obstacles_enabled(self) -> bool:
        return bool(self._ov("zufallskarten_mit_hindernissen", False))

    def failures_enabled(self) -> bool:
        return bool(self._ov("ausfaelle", False))

    def demand_shift_enabled(self) -> bool:
        return bool(self._ov("wechselnder_bedarf", False))

    def flight_minutes(self) -> float:
        return float(self._ov("flugdauer_min", self.raw["akku"]["flugdauer_schwebeflug_min"]))

    # --- unveraenderliche Modellwerte (keine Stufen-Overrides) -------- #
    @property
    def drone(self):        return self.raw["drohne"]
    @property
    def akku(self):         return self.raw["akku"]
    @property
    def station(self):      return self.raw["ladestation"]
    @property
    def funk(self):         return self.raw["funk"]
    @property
    def bedarf(self):       return self.raw["bedarf"]
    @property
    def umwelt(self):       return self.raw["umwelt"]
    @property
    def reward(self):       return self.raw["belohnung"]
    @property
    def advance_rule(self): return self.raw.get("curriculum", {}).get("aufstieg_wenn", "")
