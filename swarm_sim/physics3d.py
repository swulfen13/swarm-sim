"""Vektorisierte 3D-Physik fuer die Such-Mission (Phase 5).

Wie physics.py, aber mit Positionen/Geschwindigkeiten in 3D. Die Arena ist
eine Box [0,W] x [0,H] x [min_altitude, D]; Drohnen bleiben also ueber dem
Boden. Alles vollvektorisiert ueber alle Environments.

    pos, vel : (E, A, 3)   float32
    alive    : (E, A)      bool
"""

from __future__ import annotations

import torch

from .config import SimConfig


class PhysicsWorld3D:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        E, A = cfg.num_envs, cfg.num_agents
        self.lo = torch.tensor([0.0, 0.0, cfg.min_altitude], device=self.device)
        self.hi = torch.tensor([cfg.arena_w, cfg.arena_h, cfg.arena_d], device=self.device)
        self.pos = torch.zeros(E, A, 3, device=self.device)
        self.vel = torch.zeros(E, A, 3, device=self.device)
        self.team = (torch.arange(A, device=self.device) % cfg.num_teams).expand(E, A).contiguous()
        self.alive = torch.ones(E, A, device=self.device, dtype=torch.bool)
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        cfg = self.cfg
        E, A = cfg.num_envs, cfg.num_agents
        r = torch.rand(E, A, 3, device=self.device, generator=self.gen)
        self.pos = self.lo + r * (self.hi - self.lo)
        self.vel = torch.zeros(E, A, 3, device=self.device)

    # ------------------------------------------------------------------ #
    def step(self, actions: torch.Tensor) -> None:
        """actions: (E, A, 3) Schubkraefte. Tote Drohnen bewegen sich nicht."""
        cfg = self.cfg
        alive_f = self.alive.unsqueeze(-1).float()
        thrust = self._clip_norm(actions, cfg.max_thrust) * alive_f

        speed = self.vel.norm(dim=-1, keepdim=True)
        drag = -cfg.drag_coef * speed * self.vel
        accel = (thrust + drag) / cfg.mass
        self.vel = self.vel + accel * cfg.dt
        self.vel = self._clip_norm(self.vel, cfg.max_speed) * alive_f
        self.pos = self.pos + self.vel * cfg.dt
        self._resolve_walls()
        self._resolve_collisions()

    # ------------------------------------------------------------------ #
    def _resolve_walls(self) -> None:
        r = self.cfg.radius
        e = self.cfg.wall_restitution
        for ax in range(3):
            lo_b = self.lo[ax] + r
            hi_b = self.hi[ax] - r
            p = self.pos[..., ax]
            v = self.vel[..., ax]
            lo = p < lo_b
            p = torch.where(lo, 2 * lo_b - p, p)
            v = torch.where(lo, -v * e, v)
            hi = p > hi_b
            p = torch.where(hi, 2 * hi_b - p, p)
            v = torch.where(hi, -v * e, v)
            self.pos[..., ax] = p
            self.vel[..., ax] = v

    # ------------------------------------------------------------------ #
    def _resolve_collisions(self) -> None:
        cfg = self.cfg
        E, A, _ = self.pos.shape
        min_dist = 2 * cfg.radius
        diff = self.pos.unsqueeze(2) - self.pos.unsqueeze(1)      # (E,A,A,3)
        dist = diff.norm(dim=-1)
        eye = torch.eye(A, device=self.device, dtype=torch.bool)
        overlap = (dist < min_dist) & ~eye
        overlap = overlap & (self.alive.unsqueeze(2) & self.alive.unsqueeze(1))
        if not overlap.any():
            return
        safe = dist.clamp(min=1e-6).unsqueeze(-1)
        normal = diff / safe
        pen = (min_dist - dist).clamp(min=0.0).unsqueeze(-1)
        mask = overlap.unsqueeze(-1)
        push = torch.where(mask, normal * pen * 0.5, torch.zeros_like(normal))
        self.pos = self.pos + push.sum(dim=2)
        rel_v = self.vel.unsqueeze(2) - self.vel.unsqueeze(1)
        vn = (rel_v * normal).sum(dim=-1, keepdim=True)
        approaching = (vn < 0) & mask
        impulse = torch.where(approaching,
                              -(1 + cfg.collision_restitution) * vn * normal * 0.5,
                              torch.zeros_like(normal))
        self.vel = self.vel + impulse.sum(dim=2)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _clip_norm(vec: torch.Tensor, max_norm: float) -> torch.Tensor:
        norm = vec.norm(dim=-1, keepdim=True)
        scale = (max_norm / norm.clamp(min=1e-6)).clamp(max=1.0)
        return vec * scale
