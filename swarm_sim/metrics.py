"""Lokales Monitoring ueber TensorBoard.

Schreibt alle Kennzahlen in lokale Event-Dateien unter runs/<name>/. Nichts
wird hochgeladen - das Dashboard laeuft komplett auf dem eigenen Rechner:

    tensorboard --logdir runs
    # dann im Browser http://localhost:6006 oeffnen
    # vom Handy aus (gleiches WLAN):
    tensorboard --logdir runs --host 0.0.0.0
    # dann http://<PC-IP>:6006

Die Kennzahlen folgen den Bereichen aus dem Projektplan:
    loss/*     Policy-, Value-Loss, Entropie   (ab Phase 3)
    progress/* Belohnung, Siegquote, Elo ...   (ab Phase 3)
    mesh/*     Gruppen, Verbindungen ...        (ab Phase 2)
    system/*   Steps/s, GPU, VRAM, Temperatur
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path


class MetricsLogger:
    def __init__(self, run_name: str | None = None, logdir: str | Path = "runs"):
        from torch.utils.tensorboard import SummaryWriter

        run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
        self.path = Path(logdir) / run_name
        self.writer = SummaryWriter(log_dir=str(self.path))
        self._last_t = time.perf_counter()
        self._last_step = 0

    # ------------------------------------------------------------------ #
    def log(self, metrics: dict, step: int) -> None:
        """Schreibt ein Dict {tag: wert} als Skalare."""
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, step)

    # ------------------------------------------------------------------ #
    def log_throughput(self, step: int, envs: int) -> None:
        """Berechnet Env-Steps/s seit dem letzten Aufruf und loggt sie."""
        now = time.perf_counter()
        dt = now - self._last_t
        dstep = step - self._last_step
        if dt > 0 and dstep > 0:
            self.writer.add_scalar("system/env_steps_per_s", envs * dstep / dt, step)
        self._last_t = now
        self._last_step = step

    # ------------------------------------------------------------------ #
    def log_gpu(self, step: int) -> None:
        """Loggt GPU-Auslastung, VRAM und Temperatur (via nvidia-smi)."""
        stats = self.gpu_stats()
        if stats:
            self.log({f"system/{k}": v for k, v in stats.items()}, step)

    @staticmethod
    def gpu_stats() -> dict:
        """Fragt nvidia-smi ab. Leeres Dict, wenn keine GPU/kein nvidia-smi."""
        query = "utilization.gpu,memory.used,memory.total,temperature.gpu"
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip().splitlines()[0]
            util, mem_used, mem_total, temp = (x.strip() for x in out.split(","))
            return {
                "gpu_util_pct": float(util),
                "vram_used_mb": float(mem_used),
                "vram_total_mb": float(mem_total),
                "gpu_temp_c": float(temp),
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self.writer.flush()
        self.writer.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
