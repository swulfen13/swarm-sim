"""Zonenkontrolle mit Dynamik (Phase 4): Nachschub, Ausfaelle, Zerstoerung.

Jedes Team hat feste Slots; nur ein Teil startet aktiv, der Rest ist Reserve.
Waehrend der Runde:
  * Zerstoerung: ein lebender Agent faellt aus, wenn ihn Gegner in Nahdistanz
    um mindestens destroy_margin ueberzahlen (Gegner - Verbuendete).
  * Nachschub:  alle spawn_interval Schritte aktiviert jedes Team einen
    Reserve-Agenten am eigenen Rand.

Beobachtungen sind mengenbasiert (pro Agent ein Eintrag) mit alive-Maske,
damit das Attention-Netz mit wechselnden Teamgroessen umgeht.

step() liefert:  obs(dict), reward(E,T), done(E,), info(dict)
  obs = {entities (E,A,5), alive (E,A), team (E,A),
         self_global (E,A,3), team_global (E,T,2)}
"""

from __future__ import annotations

import math

import torch

from .config import SimConfig
from .physics import PhysicsWorld

_DIR_LIST = [(0.0, 0.0)]
for _k in range(8):
    _a = math.pi * _k / 4.0
    _DIR_LIST.append((math.cos(_a), math.sin(_a)))
NUM_ACTIONS = len(_DIR_LIST)


class ZoneControlDynEnv:
    entity_dim = 5
    self_global_dim = 3
    team_global_dim = 2

    def __init__(self, cfg: SimConfig):
        assert cfg.num_agents % cfg.num_teams == 0
        self.cfg = cfg
        self.world = PhysicsWorld(cfg)
        self.device = self.world.device
        self.E, self.A, self.T = cfg.num_envs, cfg.num_agents, cfg.num_teams
        self.k = self.A // self.T

        self.dirs = torch.tensor(_DIR_LIST, device=self.device, dtype=torch.float32)
        self.center = torch.tensor([cfg.arena_w / 2, cfg.arena_h / 2], device=self.device)
        self.half = torch.tensor([cfg.arena_w / 2, cfg.arena_h / 2], device=self.device)

        self.team_row = self.world.team[0]
        self.idx_by_team = [torch.nonzero(self.team_row == t).flatten() for t in range(self.T)]
        # feste Spawn-x-Position je Team (team0 links, letztes Team rechts)
        self.spawn_x = [cfg.arena_w * (0.08 + 0.84 * (t / max(self.T - 1, 1))) for t in range(self.T)]

        self.t_step = 0
        self.cum_in_zone = torch.zeros(self.E, self.T, device=self.device)

    # ------------------------------------------------------------------ #
    def reset(self):
        self.world.reset()
        self.t_step = 0
        self.cum_in_zone.zero_()
        alive = torch.zeros(self.E, self.A, device=self.device, dtype=torch.bool)
        n0 = min(self.cfg.start_active_per_team, self.k)
        for t in range(self.T):
            alive[:, self.idx_by_team[t][:n0]] = True
        self.world.alive = alive
        self.world.vel.zero_()
        return self._observe()

    # ------------------------------------------------------------------ #
    def step(self, actions: torch.Tensor):
        thrust = self.dirs[actions] * self.cfg.max_thrust
        self.world.step(thrust)

        destroyed = self._apply_destruction()
        spawned = self._apply_reinforcements()

        in_zone = self._in_zone()
        team_cnt = self._team_counts(in_zone)
        self.cum_in_zone += team_cnt

        reward = torch.zeros(self.E, self.T, device=self.device)
        total = team_cnt.sum(dim=1)
        for t in range(self.T):
            others = (total - team_cnt[:, t]) / max(self.T - 1, 1)
            reward[:, t] = (team_cnt[:, t] - others) / self.k

        self.t_step += 1
        done = self.t_step % self.cfg.episode_len == 0
        done_vec = torch.full((self.E,), bool(done), device=self.device, dtype=torch.bool)

        info = {
            "in_zone_frac": (team_cnt.sum(dim=1) / self.A).mean().item(),
            "destroyed_per_step": destroyed.float().sum(dim=1).mean().item(),
            "spawned_per_step": float(spawned),
            "alive_frac": self.world.alive.float().mean().item(),
        }
        if done:
            winner = self.cum_in_zone.argmax(dim=1)
            info["winrate_team0"] = (winner == 0).float().mean().item()
            obs = self.reset()
        else:
            obs = self._observe()
        return obs, reward, done_vec, info

    # ------------------------------------------------------------------ #
    def _apply_destruction(self) -> torch.Tensor:
        """Markiert ueberzahlte Agenten als zerstoert. Gibt (E,A) bool zurueck."""
        cfg = self.cfg
        pos, team, alive = self.world.pos, self.world.team, self.world.alive
        A = self.A
        dist = torch.cdist(pos, pos)                              # (E,A,A)
        eye = torch.eye(A, device=self.device, dtype=torch.bool)
        near = (dist <= cfg.combat_radius) & ~eye & alive.unsqueeze(1)  # Nachbar k lebt
        same = team.unsqueeze(2) == team.unsqueeze(1)
        ally_cnt = (near & same).sum(dim=2)
        enemy_cnt = (near & ~same).sum(dim=2)
        destroyed = alive & ((enemy_cnt - ally_cnt) >= cfg.destroy_margin)
        self.world.alive = alive & ~destroyed
        return destroyed

    # ------------------------------------------------------------------ #
    def _apply_reinforcements(self) -> int:
        """Aktiviert alle spawn_interval Schritte je Team einen Reserve-Agenten."""
        if self.t_step == 0 or self.t_step % self.cfg.spawn_interval != 0:
            return 0
        spawned = 0
        alive = self.world.alive
        for t in range(self.T):
            idx = self.idx_by_team[t]
            sub = alive[:, idx]                                   # (E,k)
            has_reserve = (~sub).any(dim=1)                       # (E,)
            local = (~sub).float().argmax(dim=1)                  # erster inaktiver Slot
            gslot = idx[local]                                    # (E,)
            rows = torch.nonzero(has_reserve).flatten()
            if rows.numel() == 0:
                continue
            cols = gslot[rows]
            self.world.alive[rows, cols] = True
            y = torch.rand(rows.numel(), device=self.device, generator=self.world.gen) * self.cfg.arena_h
            self.world.pos[rows, cols, 0] = self.spawn_x[t]
            self.world.pos[rows, cols, 1] = y
            self.world.vel[rows, cols] = 0.0
            spawned += rows.numel()
        return spawned

    # ------------------------------------------------------------------ #
    def _in_zone(self) -> torch.Tensor:
        d = (self.world.pos - self.center).norm(dim=-1)
        return ((d <= self.cfg.zone_radius) & self.world.alive).float()

    def _team_counts(self, in_zone: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.E, self.T, device=self.device)
        for t in range(self.T):
            out[:, t] = in_zone[:, self.idx_by_team[t]].sum(dim=1)
        return out

    # ------------------------------------------------------------------ #
    def _observe(self):
        pos, vel, team, alive = self.world.pos, self.world.vel, self.world.team, self.world.alive
        pos_n = (pos - self.center) / self.half
        vel_n = vel / self.cfg.max_speed
        in_zone = self._in_zone().unsqueeze(-1)
        entities = torch.cat([pos_n, vel_n, in_zone], dim=-1)        # (E,A,5)

        to_zone = (self.center - pos) / self.half
        dist_zone = ((pos - self.center).norm(dim=-1, keepdim=True) - self.cfg.zone_radius) / self.half[0]
        self_global = torch.cat([to_zone, dist_zone], dim=-1)        # (E,A,3)

        team_cnt = self._team_counts(in_zone.squeeze(-1)) / self.k
        team_global = torch.zeros(self.E, self.T, 2, device=self.device)
        for t in range(self.T):
            own = team_cnt[:, t]
            enemy = (team_cnt.sum(dim=1) - own) / max(self.T - 1, 1)
            team_global[:, t, 0] = own
            team_global[:, t, 1] = enemy

        return {
            "entities": entities,
            "alive": alive,
            "team": team,
            "self_global": self_global,
            "team_global": team_global,
        }

    def agent_team_index(self) -> torch.Tensor:
        return self.team_row
