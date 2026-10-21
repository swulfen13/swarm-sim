"""Zonenkontrolle - vektorisierte Multi-Agent-Umgebung fuer MAPPO (Phase 3).

Ein Zielbereich liegt in der Arenamitte. Jedes Team wird pro Schritt belohnt,
wenn es mehr eigene Agenten in der Zone hat als der Gegner (Nullsummen).

CTDE-Schnittstelle:
  actor_obs   (E, A, obs_dim)    dezentrale, team-relative Beobachtung je Agent
  critic_state(E, T, state_dim)  zentrale Team-Sicht (alle Agenten) fuer den Critic
  reward      (E, T)             Team-Belohnung pro Schritt
  done        (E,)               True am Rundenende (alle Envs synchron)

Diskrete Aktionen: 0 = stehen, 1..8 = Schub in 8 Richtungen.
Teams sind fest alternierend verteilt (0,1,0,1,...) und gleich gross.
"""

from __future__ import annotations

import math

import torch

from .config import SimConfig
from .physics import PhysicsWorld

# 0 = stehen, danach 8 Himmelsrichtungen (Einheitsvektoren)
_DIR_LIST = [(0.0, 0.0)]
for k in range(8):
    a = math.pi * k / 4.0
    _DIR_LIST.append((math.cos(a), math.sin(a)))
NUM_ACTIONS = len(_DIR_LIST)  # 9


class ZoneControlEnv:
    def __init__(self, cfg: SimConfig):
        assert cfg.num_agents % cfg.num_teams == 0, "num_agents muss durch num_teams teilbar sein"
        self.cfg = cfg
        self.world = PhysicsWorld(cfg)
        self.device = self.world.device
        self.E, self.A, self.T = cfg.num_envs, cfg.num_agents, cfg.num_teams
        self.k = self.A // self.T  # Agenten pro Team

        self.dirs = torch.tensor(_DIR_LIST, device=self.device, dtype=torch.float32)  # (9,2)
        self.center = torch.tensor([cfg.arena_w / 2, cfg.arena_h / 2], device=self.device)
        self.half = torch.tensor([cfg.arena_w / 2, cfg.arena_h / 2], device=self.device)

        # feste Team-Zugehoerigkeit je Spalte und Index-Listen je Team
        self.team_row = self.world.team[0]  # (A,)
        self.idx_by_team = [torch.nonzero(self.team_row == t).flatten() for t in range(self.T)]

        self.t_step = 0
        self.cum_in_zone = torch.zeros(self.E, self.T, device=self.device)

        self.obs_dim = 17
        self.state_dim = self.A * 4 + 4  # pos+vel aller Agenten (team-relativ) + Zone/Counts

    # ------------------------------------------------------------------ #
    def reset(self):
        self.world.reset()
        self.t_step = 0
        self.cum_in_zone.zero_()
        return self._observe()

    # ------------------------------------------------------------------ #
    def step(self, actions: torch.Tensor):
        """actions: (E, A) long in [0, NUM_ACTIONS). Gibt obs, state, reward, done, info."""
        thrust = self.dirs[actions] * self.cfg.max_thrust      # (E, A, 2)
        self.world.step(thrust)

        in_zone = self._in_zone()                               # (E, A) float
        team_cnt = self._team_counts(in_zone)                   # (E, T)
        self.cum_in_zone += team_cnt

        # Nullsummen-Belohnung je Team (fuer 2 Teams: Differenz / Teamgroesse)
        reward = torch.zeros(self.E, self.T, device=self.device)
        total = team_cnt.sum(dim=1, keepdim=True)
        for t in range(self.T):
            others = (total.squeeze(1) - team_cnt[:, t]) / max(self.T - 1, 1)
            reward[:, t] = (team_cnt[:, t] - others) / self.k

        self.t_step += 1
        done = self.t_step % self.cfg.episode_len == 0
        done_vec = torch.full((self.E,), bool(done), device=self.device, dtype=torch.bool)

        info = {"in_zone_frac": (team_cnt.sum(dim=1) / self.A).mean().item()}
        if done:
            winner = self.cum_in_zone.argmax(dim=1)             # (E,)
            info["winrate_team0"] = (winner == 0).float().mean().item()
            info["zone_balance"] = (team_cnt[:, 0] - team_cnt[:, 1]).mean().item() if self.T == 2 else 0.0
            obs, state = self.reset()
        else:
            obs, state = self._observe()
        return obs, state, reward, done_vec, info

    # ------------------------------------------------------------------ #
    def _in_zone(self) -> torch.Tensor:
        d = (self.world.pos - self.center).norm(dim=-1)         # (E, A)
        return (d <= self.cfg.zone_radius).float()

    def _team_counts(self, in_zone: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.E, self.T, device=self.device)
        for t in range(self.T):
            out[:, t] = in_zone[:, self.idx_by_team[t]].sum(dim=1)
        return out

    # ------------------------------------------------------------------ #
    def _observe(self):
        pos, vel, team = self.world.pos, self.world.vel, self.world.team
        E, A = self.E, self.A

        # paarweise Relativvektoren rel[e,i,j] = pos_j - pos_i
        rel = pos.unsqueeze(1) - pos.unsqueeze(2)               # (E,A,A,2)
        dist = rel.norm(dim=-1)                                  # (E,A,A)
        eye = torch.eye(A, device=self.device, dtype=torch.bool)
        same = (team.unsqueeze(2) == team.unsqueeze(1))         # (E,A,A)  [i,j]
        ally = same & ~eye
        enemy = ~same

        in_zone = self._in_zone()                               # (E,A)

        def centroid(mask):
            f = mask.unsqueeze(-1).float()
            c = f.sum(dim=2).clamp(min=1.0)
            return (rel * f).sum(dim=2) / c                     # (E,A,2) relativ zu i

        def nearest(mask):
            big = torch.where(mask, dist, torch.full_like(dist, 1e9))
            j = big.argmin(dim=2)                                # (E,A)
            return torch.gather(rel, 2, j.unsqueeze(-1).unsqueeze(-1).expand(E, A, 1, 2)).squeeze(2)

        own_pos = (pos - self.center) / self.half               # (E,A,2) in ~[-1,1]
        own_vel = vel / self.cfg.max_speed
        to_zone = (self.center - pos) / self.half
        dist_zone = ((pos - self.center).norm(dim=-1, keepdim=True) - self.cfg.zone_radius) / self.half[0]

        ally_in = (ally.float() * in_zone.unsqueeze(1)).sum(dim=2, keepdim=True) / self.k
        enemy_in = (enemy.float() * in_zone.unsqueeze(1)).sum(dim=2, keepdim=True) / self.k

        obs = torch.cat([
            own_pos, own_vel, to_zone, dist_zone,
            ally_in, enemy_in,
            centroid(ally) / self.half, centroid(enemy) / self.half,
            nearest(ally) / self.half, nearest(enemy) / self.half,
        ], dim=-1)                                              # (E,A,17)

        state = self._critic_state(pos, vel)
        return obs, state

    def _critic_state(self, pos, vel):
        """Zentrale, team-relative Sicht: pos+vel aller Agenten (Verbuendete zuerst,

        dann Gegner), plus Zone-Counts. Ergibt (E, T, state_dim).
        """
        E = self.E
        pos_n = (pos - self.center) / self.half                 # (E,A,2)
        vel_n = vel / self.cfg.max_speed
        in_zone = self._in_zone()
        team_cnt = self._team_counts(in_zone) / self.k          # (E,T)

        states = []
        for t in range(self.T):
            ally_idx = self.idx_by_team[t]
            enemy_idx = torch.cat([self.idx_by_team[o] for o in range(self.T) if o != t])
            order = torch.cat([ally_idx, enemy_idx])
            p = pos_n[:, order, :].reshape(E, -1)
            v = vel_n[:, order, :].reshape(E, -1)
            own = team_cnt[:, t:t + 1]
            enemy_c = (team_cnt.sum(dim=1, keepdim=True) - own)
            zone = torch.cat([own, enemy_c, torch.zeros(E, 2, device=self.device)], dim=1)
            states.append(torch.cat([p, v, zone], dim=1))
        return torch.stack(states, dim=1)                       # (E, T, state_dim)

    # ------------------------------------------------------------------ #
    def agent_team_index(self) -> torch.Tensor:
        """(A,) Team je Agent - um Agenten ihrem Team-Wert zuzuordnen."""
        return self.team_row
