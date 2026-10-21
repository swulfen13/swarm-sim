"""Vektorisierte 2D-Physik fuer tausende Arenen parallel.

Der gesamte Zustand liegt als PyTorch-Tensor auf der GPU. Ein Schritt
verarbeitet alle Environments gleichzeitig; es gibt keine Python-Schleife
ueber Agenten oder Arenen. Genau diese Struktur braucht Phase 3 (MAPPO),
damit das Sammeln von Erfahrung schnell bleibt.

Konventionen der Tensor-Formen:
    pos, vel : (E, A, 2)   float32   Position / Geschwindigkeit
    team     : (E, A)      int64     Teamzugehoerigkeit
    actions  : (E, A, 2)   float32   gewuenschter Schubvektor je Agent
"""

from __future__ import annotations

import torch

from .config import SimConfig


class PhysicsWorld:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)

        E, A = cfg.num_envs, cfg.num_agents
        self.pos = torch.zeros(E, A, 2, device=self.device)
        self.vel = torch.zeros(E, A, 2, device=self.device)
        # Teams moeglichst gleichmaessig verteilen: [0,1,0,1,...]
        self.team = (torch.arange(A, device=self.device) % cfg.num_teams).expand(E, A).contiguous()
        # alive-Maske: tote/inaktive Agenten bewegen sich nicht und kollidieren
        # nicht. Default alle lebend -> Phasen 1-3 unveraendert. Phase 4 setzt sie.
        self.alive = torch.ones(E, A, device=self.device, dtype=torch.bool)
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        """Setzt Positionen zufaellig und Geschwindigkeiten auf 0.

        env_mask (E,) bool erlaubt es, nur einzelne Arenen neu zu starten
        (wird in spaeteren Phasen bei Rundenende gebraucht).
        """
        cfg = self.cfg
        E, A = cfg.num_envs, cfg.num_agents
        new_pos = torch.rand(E, A, 2, device=self.device, generator=self.gen)
        new_pos[..., 0] *= cfg.arena_w
        new_pos[..., 1] *= cfg.arena_h
        if env_mask is None:
            self.pos = new_pos
            self.vel = torch.zeros(E, A, 2, device=self.device)
        else:
            m = env_mask.view(E, 1, 1)
            self.pos = torch.where(m, new_pos, self.pos)
            self.vel = torch.where(m, torch.zeros_like(self.vel), self.vel)

    # ------------------------------------------------------------------ #
    def step(self, actions: torch.Tensor) -> None:
        """Ein Physikschritt fuer alle Environments.

        actions: (E, A, 2) gewuenschte Schubkraefte. Werden auf max_thrust
        begrenzt. Reihenfolge: Schub + Luftwiderstand -> Beschleunigung ->
        Geschwindigkeit (gedeckelt) -> Position -> Kollisionen.
        """
        cfg = self.cfg
        alive_f = self.alive.unsqueeze(-1).float()   # (E, A, 1)

        # Schub auf maximale Kraft begrenzen (Richtung bleibt erhalten);
        # tote Agenten bekommen keinen Schub.
        thrust = self._clip_norm(actions, cfg.max_thrust) * alive_f

        # Quadratischer Luftwiderstand: F = -c_d * |v| * v
        speed = self.vel.norm(dim=-1, keepdim=True)
        drag = -cfg.drag_coef * speed * self.vel

        accel = (thrust + drag) / cfg.mass
        self.vel = self.vel + accel * cfg.dt
        self.vel = self._clip_norm(self.vel, cfg.max_speed)
        self.vel = self.vel * alive_f                # tote Agenten stehen still
        self.pos = self.pos + self.vel * cfg.dt

        self._resolve_walls()
        self._resolve_collisions()

    # ------------------------------------------------------------------ #
    def _resolve_walls(self) -> None:
        """Reflexion an den Arenawaenden mit Energieverlust."""
        cfg = self.cfg
        r, e = cfg.radius, cfg.wall_restitution
        for ax, upper in ((0, cfg.arena_w), (1, cfg.arena_h)):
            p = self.pos[..., ax]
            v = self.vel[..., ax]
            # untere Wand
            lo = p < r
            p = torch.where(lo, 2 * r - p, p)
            v = torch.where(lo, -v * e, v)
            # obere Wand
            hi = p > upper - r
            p = torch.where(hi, 2 * (upper - r) - p, p)
            v = torch.where(hi, -v * e, v)
            self.pos[..., ax] = p
            self.vel[..., ax] = v

    # ------------------------------------------------------------------ #
    def _resolve_collisions(self) -> None:
        """Paarweise Agent-Agent-Kollisionen, vollvektorisiert.

        A ist klein (<= ~30), daher ist die A x A Matrix guenstig. Bei
        Ueberlappung werden die Agenten entlang der Verbindungsachse
        auseinandergeschoben und die Normalkomponente der Relativ-
        geschwindigkeit teil-elastisch umgekehrt.
        """
        cfg = self.cfg
        E, A, _ = self.pos.shape
        min_dist = 2 * cfg.radius

        diff = self.pos.unsqueeze(2) - self.pos.unsqueeze(1)      # (E,A,A,2)
        dist = diff.norm(dim=-1)                                   # (E,A,A)
        eye = torch.eye(A, device=self.device, dtype=torch.bool)
        overlap = (dist < min_dist) & ~eye                         # (E,A,A)
        # tote Agenten kollidieren nicht (weder als i noch als j)
        alive_pair = self.alive.unsqueeze(2) & self.alive.unsqueeze(1)
        overlap = overlap & alive_pair

        if not overlap.any():
            return

        safe_dist = dist.clamp(min=1e-6).unsqueeze(-1)             # (E,A,A,1)
        normal = diff / safe_dist                                  # Einheitsrichtung i<-j
        pen = (min_dist - dist).clamp(min=0.0).unsqueeze(-1)       # Eindringtiefe
        mask = overlap.unsqueeze(-1)

        # Positionskorrektur: jede Seite haelft der Eindringtiefe
        push = torch.where(mask, normal * pen * 0.5, torch.zeros_like(normal))
        self.pos = self.pos + push.sum(dim=2)

        # Geschwindigkeitsaustausch entlang der Normalen (teil-elastisch)
        rel_v = self.vel.unsqueeze(2) - self.vel.unsqueeze(1)      # (E,A,A,2)
        vn = (rel_v * normal).sum(dim=-1, keepdim=True)            # Annaeherung < 0
        approaching = (vn < 0) & mask
        impulse = torch.where(
            approaching,
            -(1 + cfg.collision_restitution) * vn * normal * 0.5,
            torch.zeros_like(normal),
        )
        self.vel = self.vel + impulse.sum(dim=2)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _clip_norm(vec: torch.Tensor, max_norm: float) -> torch.Tensor:
        """Skaliert Vektoren auf hoechstens max_norm Laenge (Richtung bleibt)."""
        norm = vec.norm(dim=-1, keepdim=True)
        scale = (max_norm / norm.clamp(min=1e-6)).clamp(max=1.0)
        return vec * scale

    # ------------------------------------------------------------------ #
    def state_cpu(self) -> dict:
        """Snapshot des Zustands als CPU-Tensoren (fuer Viewer/Recorder)."""
        return {
            "pos": self.pos.detach().to("cpu"),
            "vel": self.vel.detach().to("cpu"),
            "team": self.team.detach().to("cpu"),
        }
