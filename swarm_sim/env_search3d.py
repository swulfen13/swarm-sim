"""3D-Such-Mission mit Zufallsereignissen (Phase 5).

Ein kooperativer Schwarm (Standard: ein Team) soll den Boden absuchen
(Suchraster bilden) und versteckte Ziele finden. Die Kamera-Abdeckung am
Boden haengt von der Flughoehe ab: hoeher = groesserer Fussabdruck, aber die
Drohne deckt dann groeber ab. Das gibt der 3. Dimension echten Sinn.

Zufallsereignisse zwingen zum Umplanen:
  * zufaellige Ausfaelle (fail_prob je Drohne/Schritt)
  * Voegel fliegen durch die Arena; Drohnen in bird_danger_radius fallen aus

Belohnung (geteilt, kooperativ):
  + neu abgedeckte Rasterzellen   + gefundene Ziele

step() -> obs(dict), reward(E,T), done(E,), info(dict)
  obs = {entities (E,A,6), alive (E,A), team (E,A),
         self_global (E,A,8), team_global (E,T,3)}
"""

from __future__ import annotations

import math

import torch

from .config import SimConfig
from .physics3d import PhysicsWorld3D

# 3D-Aktionen: stehen + 8 Horizontalrichtungen + hoch + runter = 11
_DIRS = [(0.0, 0.0, 0.0)]
for _k in range(8):
    _a = math.pi * _k / 4.0
    _DIRS.append((math.cos(_a), math.sin(_a), 0.0))
_DIRS += [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]
NUM_ACTIONS = len(_DIRS)


class SearchEnv3D:
    entity_dim = 6
    self_global_dim = 8
    team_global_dim = 3

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.world = PhysicsWorld3D(cfg)
        self.device = self.world.device
        self.E, self.A, self.T = cfg.num_envs, cfg.num_agents, cfg.num_teams
        self.k = self.A // self.T
        self.G = cfg.grid_size

        self.dirs = torch.tensor(_DIRS, device=self.device, dtype=torch.float32)
        self.hi = self.world.hi
        self.cell_w = cfg.arena_w / self.G
        self.cell_h = cfg.arena_h / self.G
        # Zellmittelpunkte am Boden (GG, 2)
        gx = (torch.arange(self.G, device=self.device) + 0.5) * self.cell_w
        gy = (torch.arange(self.G, device=self.device) + 0.5) * self.cell_h
        cx, cy = torch.meshgrid(gx, gy, indexing="ij")
        self.cell_xy = torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=-1)   # (GG,2)
        self.GG = self.G * self.G

        self.team_row = self.world.team[0]
        self.idx_by_team = [torch.nonzero(self.team_row == t).flatten() for t in range(self.T)]
        self.t_step = 0

    # ------------------------------------------------------------------ #
    def reset(self):
        self.world.reset()
        self.world.alive = torch.ones(self.E, self.A, device=self.device, dtype=torch.bool)
        self.t_step = 0
        self.covered = torch.zeros(self.E, self.GG, device=self.device, dtype=torch.bool)
        # Ziele auf zufaelligen Zellen
        self.target_cell = torch.randint(0, self.GG, (self.E, self.cfg.num_targets),
                                          device=self.device, generator=self.world.gen)
        self.found = torch.zeros(self.E, self.cfg.num_targets, device=self.device, dtype=torch.bool)
        self._spawn_birds(torch.ones(self.E, self.cfg.bird_count, device=self.device, dtype=torch.bool))
        return self._observe()[0]

    # ------------------------------------------------------------------ #
    def _spawn_birds(self, mask):
        """Setzt (neue) Voegel an einen zufaelligen Rand mit Flug quer durch."""
        E, B = self.E, self.cfg.bird_count
        if not hasattr(self, "bird_pos"):
            self.bird_pos = torch.zeros(E, B, 3, device=self.device)
            self.bird_vel = torch.zeros(E, B, 3, device=self.device)
        g = self.world.gen
        start = torch.rand(E, B, 3, device=self.device, generator=g) * self.hi
        start[..., 0] = 0.0  # am linken Rand starten
        target = torch.rand(E, B, 3, device=self.device, generator=g) * self.hi
        target[..., 0] = self.cfg.arena_w
        direction = target - start
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        m = mask.unsqueeze(-1)
        self.bird_pos = torch.where(m, start, self.bird_pos)
        self.bird_vel = torch.where(m, direction * self.cfg.bird_speed, self.bird_vel)

    def _move_birds(self):
        self.bird_pos = self.bird_pos + self.bird_vel * self.cfg.dt
        # Vogel ausserhalb -> neu spawnen
        out = (self.bird_pos[..., 0] > self.cfg.arena_w + 5) | (self.bird_pos[..., 0] < -5)
        if out.any():
            self._spawn_birds(out)

    # ------------------------------------------------------------------ #
    def step(self, actions: torch.Tensor):
        thrust = self.dirs[actions] * self.cfg.max_thrust      # (E,A,3)
        self.world.step(thrust)

        # 1) zufaellige Ausfaelle
        roll = torch.rand(self.E, self.A, device=self.device, generator=self.world.gen)
        rand_fail = self.world.alive & (roll < self.cfg.fail_prob)

        # 2) Voegel bewegen und Treffer pruefen
        self._move_birds()
        bd = torch.cdist(self.world.pos, self.bird_pos)        # (E,A,B)
        hit = self.world.alive & (bd.min(dim=2).values <= self.cfg.bird_danger_radius)

        destroyed = rand_fail | hit
        self.world.alive = self.world.alive & ~destroyed

        # 3) Abdeckung berechnen
        obs, newly_cov, new_finds = self._observe(update_coverage=True)

        reward = torch.zeros(self.E, self.T, device=self.device)
        rew = (newly_cov / self.GG) * 10.0 + new_finds * 1.0   # (E,)
        for t in range(self.T):
            reward[:, t] = rew

        self.t_step += 1
        done = self.t_step % self.cfg.episode_len == 0
        done_vec = torch.full((self.E,), bool(done), device=self.device, dtype=torch.bool)

        info = {
            "coverage_frac": self.covered.float().mean().item(),
            "targets_found_frac": self.found.float().mean().item(),
            "alive_frac": self.world.alive.float().mean().item(),
            "destroyed_per_step": destroyed.float().sum(dim=1).mean().item(),
            "bird_hits_per_step": hit.float().sum(dim=1).mean().item(),
        }
        if done:
            obs = self.reset()
        return obs, reward, done_vec, info

    # ------------------------------------------------------------------ #
    def _footprint_dist(self):
        """Abstand jeder Drohne (x,y) zu allen Bodenzellen. (E,A,GG), (E,A,1) radius."""
        xy = self.world.pos[..., :2]                           # (E,A,2)
        d = torch.cdist(xy, self.cell_xy.unsqueeze(0).expand(self.E, -1, -1))  # (E,A,GG)
        alt = (self.world.pos[..., 2] - self.cfg.min_altitude).clamp(min=0.0)
        fp = (self.cfg.footprint_base + self.cfg.footprint_alt_gain * alt).unsqueeze(-1)  # (E,A,1)
        return d, fp

    # ------------------------------------------------------------------ #
    def _observe(self, update_coverage: bool = False):
        cfg = self.cfg
        pos, vel, alive = self.world.pos, self.world.vel, self.world.alive
        d, fp = self._footprint_dist()                         # (E,A,GG), (E,A,1)

        within = (d <= fp) & alive.unsqueeze(-1)               # (E,A,GG)
        covered_now = within.any(dim=1)                        # (E,GG)

        newly_cov = torch.zeros(self.E, device=self.device)
        new_finds = torch.zeros(self.E, device=self.device)
        if update_coverage:
            fresh = covered_now & ~self.covered
            newly_cov = fresh.float().sum(dim=1)
            self.covered = self.covered | covered_now
            tgt_cov = torch.gather(self.covered.float(), 1, self.target_cell) > 0.5
            fresh_find = tgt_cov & ~self.found
            new_finds = fresh_find.float().sum(dim=1)
            self.found = self.found | tgt_cov

        # naechste unbedeckte Zelle je Drohne
        uncovered = ~self.covered                              # (E,GG)
        big = torch.where(uncovered.unsqueeze(1), d, torch.full_like(d, 1e9))
        nj = big.argmin(dim=2)                                  # (E,A)
        near_cell = self.cell_xy[nj]                            # (E,A,2)
        tgt3 = torch.cat([near_cell, torch.full_like(near_cell[..., :1], cfg.min_altitude)], dim=-1)
        to_cov = tgt3 - pos
        d_cov = to_cov.norm(dim=-1, keepdim=True)
        dir_cov = to_cov / d_cov.clamp(min=1e-6)

        # naechster Vogel je Drohne
        bd = torch.cdist(pos, self.bird_pos)                   # (E,A,B)
        bj = bd.argmin(dim=2)                                   # (E,A)
        near_bird = torch.gather(self.bird_pos, 1, bj.unsqueeze(-1).expand(-1, -1, 3))
        to_bird = near_bird - pos
        d_bird = to_bird.norm(dim=-1, keepdim=True)
        dir_bird = to_bird / d_bird.clamp(min=1e-6)

        diag = (self.hi).norm()
        pos_n = pos / self.hi
        vel_n = vel / cfg.max_speed
        entities = torch.cat([pos_n, vel_n], dim=-1)           # (E,A,6)
        self_global = torch.cat([dir_cov, d_cov / diag, dir_bird, d_bird / diag], dim=-1)  # (E,A,8)

        cov_frac = self.covered.float().mean(dim=1, keepdim=True)         # (E,1)
        found_frac = self.found.float().mean(dim=1, keepdim=True)
        alive_frac = alive.float().mean(dim=1, keepdim=True)
        tg = torch.cat([cov_frac, found_frac, alive_frac], dim=-1)       # (E,3)
        team_global = tg.unsqueeze(1).expand(self.E, self.T, 3)

        obs = {
            "entities": entities,
            "alive": alive,
            "team": self.world.team,
            "self_global": self_global,
            "team_global": team_global,
        }
        return obs, newly_cov, new_finds

    def agent_team_index(self) -> torch.Tensor:
        return self.team_row
