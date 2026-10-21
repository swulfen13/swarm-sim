"""Regelbasiertes Schwarmverhalten (Boids) als Platzhalter-"Gehirn".

In Phase 1 steuert dieser Controller die Agenten nach festen Regeln
(Separation, Alignment, Cohesion, Zielsuche, Wander). In Phase 3 wird er
durch eine gelernte MAPPO-Policy ersetzt - die Schnittstelle bleibt gleich:
Zustand rein, Schubvektor (E, A, 2) raus.
"""

from __future__ import annotations

import torch

from .config import SimConfig


class BoidsController:
    def __init__(self, cfg: SimConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.gen = torch.Generator(device=device).manual_seed(cfg.seed + 1)
        # Pro Team ein wanderndes Ziel; Agenten desselben Teams streben dorthin.
        self.goals = self._random_goals()

    def _random_goals(self) -> torch.Tensor:
        cfg = self.cfg
        g = torch.rand(cfg.num_envs, cfg.num_teams, 2, device=self.device, generator=self.gen)
        g[..., 0] *= cfg.arena_w
        g[..., 1] *= cfg.arena_h
        return g

    # ------------------------------------------------------------------ #
    def act(self, pos: torch.Tensor, vel: torch.Tensor, team: torch.Tensor) -> torch.Tensor:
        """Berechnet den gewuenschten Schubvektor je Agent.

        pos, vel : (E, A, 2)   team : (E, A)
        return   : (E, A, 2) Schubkraefte (noch ungedeckelt; die Physik
                   begrenzt auf max_thrust).
        """
        cfg = self.cfg
        E, A, _ = pos.shape

        diff = pos.unsqueeze(1) - pos.unsqueeze(2)        # (E,A,A,2) j - i
        dist = diff.norm(dim=-1)                           # (E,A,A)
        eye = torch.eye(A, device=self.device, dtype=torch.bool)

        same_team = team.unsqueeze(1) == team.unsqueeze(2)  # (E,A,A)
        in_range = (dist < cfg.perception_radius) & ~eye
        neigh = in_range & same_team                        # nur Teamkollegen
        neigh_f = neigh.unsqueeze(-1).float()
        count = neigh_f.sum(dim=2).clamp(min=1.0)           # (E,A,1)

        # --- Cohesion: Richtung zur mittleren Nachbarposition ------------
        mean_pos = (pos.unsqueeze(1) * neigh_f).sum(dim=2) / count
        cohesion = mean_pos - pos
        cohesion = self._safe_dir(cohesion)

        # --- Alignment: mittlere Nachbargeschwindigkeit angleichen -------
        mean_vel = (vel.unsqueeze(1) * neigh_f).sum(dim=2) / count
        alignment = self._safe_dir(mean_vel)

        # --- Separation: von sehr nahen Nachbarn wegdruecken -------------
        close = (dist < cfg.separation_radius) & ~eye       # alle Teams meiden
        close_f = close.unsqueeze(-1).float()
        inv = diff / dist.clamp(min=1e-6).unsqueeze(-1).pow(2)  # 1/d, Richtung i<-j
        separation = -(inv * close_f).sum(dim=2)
        separation = self._safe_dir(separation)

        # --- Zielsuche: zum wandernden Teamziel --------------------------
        # team (E,A) -> passendes Ziel je Agent aus self.goals (E,T,2)
        goal_per_agent = torch.gather(
            self.goals, 1, team.unsqueeze(-1).expand(E, A, 2)
        )
        goal = self._safe_dir(goal_per_agent - pos)

        # --- Wander: kleiner Zufallsimpuls -------------------------------
        wander = torch.randn(E, A, 2, device=self.device, generator=self.gen)

        force = (
            cfg.w_cohesion * cohesion
            + cfg.w_alignment * alignment
            + cfg.w_separation * separation
            + cfg.w_goal * goal
            + cfg.w_wander * wander
        )
        return force * cfg.max_thrust

    # ------------------------------------------------------------------ #
    def wander_goals(self, step: int, period: int = 400) -> None:
        """Setzt die Teamziele periodisch neu, damit Bewegung ins Spiel kommt."""
        if step > 0 and step % period == 0:
            self.goals = self._random_goals()

    @staticmethod
    def _safe_dir(vec: torch.Tensor) -> torch.Tensor:
        """Normiert auf Einheitslaenge, 0-Vektoren bleiben 0."""
        norm = vec.norm(dim=-1, keepdim=True)
        return vec / norm.clamp(min=1e-6)
