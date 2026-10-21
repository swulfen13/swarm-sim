"""Replay-Aufzeichnung: speichert eine Arena ueber die Zeit als .npz.

Aufgezeichnet wird standardmaessig nur Environment 0 (zum Zuschauen reicht
eine Arena). Das Format ist bewusst simpel, damit der Viewer - und spaeter
ein Browser/three.js-Viewer in Phase 5 - es leicht laden kann.

Gespeicherte Felder:
    pos     (T, A, 2) float32   Positionen je Frame
    team    (A,)      int64     Teamzugehoerigkeit (konstant)
    arena   (2,)      float32   [arena_w, arena_h]
    dt      ()        float32   Zeitschritt
    created ()        str       ISO-8601-Zeitstempel der Aufnahme
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np

from .config import SimConfig


class ReplayRecorder:
    def __init__(self, cfg: SimConfig, env_index: int = 0):
        self.cfg = cfg
        self.env_index = env_index
        self._frames: list[np.ndarray] = []
        self._team: np.ndarray | None = None

    def capture(self, state: dict) -> None:
        """Haengt den aktuellen Frame an (erwartet CPU-Tensoren aus state_cpu)."""
        pos = state["pos"][self.env_index]            # (A, 2)
        self._frames.append(pos.numpy().astype(np.float32))
        if self._team is None:
            self._team = state["team"][self.env_index].numpy().astype(np.int64)

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    def save(self, path: str | Path) -> Path:
        if not self._frames:
            raise RuntimeError("Keine Frames aufgezeichnet - erst capture() aufrufen.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pos = np.stack(self._frames, axis=0)          # (T, A, 2)
        np.savez_compressed(
            path,
            pos=pos,
            team=self._team,
            arena=np.array([self.cfg.arena_w, self.cfg.arena_h], dtype=np.float32),
            dt=np.float32(self.cfg.dt),
            radio_radius=np.float32(getattr(self.cfg, "radio_radius", 18.0)),
            created=_dt.datetime.now().replace(microsecond=0).isoformat(),
        )
        return path

    @staticmethod
    def load(path: str | Path) -> dict:
        """Laedt ein Replay als Dict mit numpy-Arrays (fuer den Viewer)."""
        data = np.load(Path(path), allow_pickle=False)
        return {
            "pos": data["pos"],
            "team": data["team"],
            "arena": data["arena"],
            "dt": float(data["dt"]),
            "radio_radius": float(data["radio_radius"]) if "radio_radius" in data else 18.0,
            "created": str(data["created"]),
        }
