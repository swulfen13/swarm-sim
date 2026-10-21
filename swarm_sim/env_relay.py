"""Krisen-Relais / Netzabdeckung (Phase 7, 3D) - dynamisch + Curriculum.

Szenario: Strom und Internet sind grossflaechig ausgefallen. Ein Schwarm
Drohnen spannt ein fliegendes Funknetz auf, damit Menschen am Boden und
Rettungskraefte wieder Daten haben. Eine Bodenzelle gilt als VERSORGT, wenn

  (a) eine Drohne sie mit ihrem Funk-Fussabdruck abdeckt  UND
  (b) diese Drohne ueber die Funk-Kette mit der Basisstation
      (dem einzigen Internet-Tor) verbunden ist.

Die NACHFRAGE (Bevoelkerung) ist eine Karte aus Hotspots: wo mehr Menschen
sind, kommt das staerkste Signal und wird mehr Netz gebraucht. Diese Karte
ist DYNAMISCH - Hotspots bewegen sich, aendern ihre Staerke und verlagern
sich waehrend der Runde. Der Schwarm muss die Abdeckung live nachfuehren.

Akku zwingt zum vorausschauenden Abloesen: Energie leert beim Fliegen, laedt
nur nahe der Basis. Wer im Feld leerlaeuft, faellt aus (Strafe). Ein
Heimkehr-Anreiz (Shaping) zieht Drohnen mit niedrigem Akku zur Basis, damit
das Abloesen OHNE Versorgungsluecke gelingt.

Curriculum: set_difficulty(frac) rampt leicht -> schwer (Funkradius,
Akkuverbrauch, Nachfrage-Dynamik, Hotspot-Streuung).

step() -> obs(dict), reward(E,T), done(E,), info(dict)
  obs = {entities (E,A,7), alive (E,A), team (E,A),
         self_global (E,A,11), team_global (E,T,3)}
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


def _lerp(a: float, b: float, f: float) -> float:
    return a + (b - a) * f


class RelayEnv3D:
    entity_dim = 7        # pos(3) + vel(3) + battery(1)
    self_global_dim = 11  # dir_base(3)+dist_base(1)+battery(1)+connected(1)+dir_gap(3)+dist_gap(1)+local_demand(1)
    team_global_dim = 3   # served_frac + connected_frac + mean_battery

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.world = PhysicsWorld3D(cfg)
        self.device = self.world.device
        self.E, self.A, self.T = cfg.num_envs, cfg.num_agents, cfg.num_teams
        self.H = cfg.num_demand
        self.G = cfg.grid_size
        self.GG = self.G * self.G

        self.dirs = torch.tensor(_DIRS, device=self.device, dtype=torch.float32)
        self.hi = self.world.hi
        self.diag = self.hi.norm()
        self.cell_w = cfg.arena_w / self.G
        self.cell_h = cfg.arena_h / self.G
        gx = (torch.arange(self.G, device=self.device) + 0.5) * self.cell_w
        gy = (torch.arange(self.G, device=self.device) + 0.5) * self.cell_h
        cx, cy = torch.meshgrid(gx, gy, indexing="ij")
        self.cell_xy = torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=-1)   # (GG,2)

        # Basisstation: Mitte am Boden (das Internet-Tor).
        self.base = torch.tensor(
            [cfg.arena_w * 0.5, cfg.arena_h * 0.5, cfg.min_altitude], device=self.device
        )

        # Curriculum-Parameter (von set_difficulty ueberschrieben); Default = schwer.
        self.radio = getattr(cfg, "radio_radius", 18.0)
        self.batt_drain = cfg.battery_drain
        self.batt_move = cfg.battery_move_drain
        self.demand_speed = cfg.demand_move_speed if cfg.demand_dynamic else 0.0
        self.spawn_frac = 1.0

        self.team_row = self.world.team[0]
        self.t_step = 0
        self.reset()

    # ------------------------------------------------------------------ #
    def set_difficulty(self, frac: float) -> None:
        """Rampt die Schwierigkeit von leicht (0) nach schwer (1).

        Wird vom Trainer je Update aufgerufen. Wirkt ab dem naechsten reset()
        voll (Hotspot-Streuung/Geschwindigkeit) bzw. sofort (Funk/Akku).
        """
        cfg = self.cfg
        if not cfg.curriculum:
            frac = 1.0
        f = max(0.0, min(1.0, frac))
        base_radio = getattr(cfg, "radio_radius", 18.0)
        self.radio = base_radio * _lerp(cfg.radio_easy_mult, 1.0, f)
        self.batt_drain = cfg.battery_drain * _lerp(cfg.batt_easy_mult, 1.0, f)
        self.batt_move = cfg.battery_move_drain * _lerp(cfg.batt_easy_mult, 1.0, f)
        self.demand_speed = (cfg.demand_move_speed if cfg.demand_dynamic else 0.0) * f
        self.spawn_frac = f   # 0 = nah an der Basis, 1 = ueber die ganze Arena

    # ------------------------------------------------------------------ #
    def reset(self):
        cfg = self.cfg
        self.world.reset()
        self.world.alive = torch.ones(self.E, self.A, device=self.device, dtype=torch.bool)
        self.t_step = 0
        lo = cfg.battery_start_min
        self.battery = lo + (1.0 - lo) * torch.rand(
            self.E, self.A, device=self.device, generator=self.world.gen
        )
        self._init_demand()
        return self._observe()[0]

    # ------------------------------------------------------------------ #
    def _init_demand(self) -> None:
        """Setzt bewegliche Hotspots: Zentren, Geschwindigkeiten, Staerken."""
        cfg = self.cfg
        g = self.world.gen
        E, H = self.E, self.H
        # Zentren: leichtes Curriculum sammelt sie nahe der Basis, schweres
        # verteilt sie ueber die ganze Arena.
        span_x = _lerp(cfg.arena_w * 0.25, cfg.arena_w, self.spawn_frac)
        span_y = _lerp(cfg.arena_h * 0.25, cfg.arena_h, self.spawn_frac)
        base_xy = self.base[:2]
        off = (torch.rand(E, H, 2, device=self.device, generator=g) - 0.5)
        off[..., 0] *= span_x
        off[..., 1] *= span_y
        self.centers = (base_xy + off).clamp(
            min=torch.zeros(2, device=self.device),
            max=torch.tensor([cfg.arena_w, cfg.arena_h], device=self.device),
        )                                                               # (E,H,2)
        ang = torch.rand(E, H, device=self.device, generator=g) * 2 * math.pi
        self.center_vel = torch.stack([ang.cos(), ang.sin()], dim=-1) * self.demand_speed
        self.intensity = 0.5 + torch.rand(E, H, device=self.device, generator=g)  # [0.5,1.5]
        self.demand = self._demand_from_state()

    # ------------------------------------------------------------------ #
    def _demand_from_state(self) -> torch.Tensor:
        """Baut die normierte Nachfragekarte (E,GG) aus den aktuellen Hotspots."""
        cfg = self.cfg
        d2 = torch.cdist(self.cell_xy.unsqueeze(0).expand(self.E, -1, -1), self.centers)  # (E,GG,H)
        bump = torch.exp(-(d2 ** 2) / (2 * cfg.demand_sigma ** 2)) * self.intensity.unsqueeze(1)
        demand = bump.sum(dim=2) + 0.05                                 # (E,GG) Grundnachfrage
        return demand / demand.sum(dim=1, keepdim=True).clamp(min=1e-6)

    # ------------------------------------------------------------------ #
    def _move_demand(self) -> None:
        """Bewegt Hotspots, variiert Staerke, verlagert gelegentlich einen."""
        cfg = self.cfg
        if not cfg.demand_dynamic:
            return
        g = self.world.gen
        self.centers = self.centers + self.center_vel * cfg.dt
        # an den Raendern abprallen
        for ax, upper in ((0, cfg.arena_w), (1, cfg.arena_h)):
            c = self.centers[..., ax]
            v = self.center_vel[..., ax]
            lo = c < 0.0
            hi = c > upper
            self.center_vel[..., ax] = torch.where(lo | hi, -v, v)
            self.centers[..., ax] = c.clamp(0.0, upper)
        # Staerke schwanken lassen (Random Walk)
        jit = (torch.rand(self.E, self.H, device=self.device, generator=g) - 0.5)
        self.intensity = (self.intensity + jit * cfg.demand_intensity_jitter).clamp(0.2, 1.8)
        # gelegentlich einen Hotspot komplett neu platzieren (Lage aendert sich)
        if cfg.demand_relocate_interval > 0 and self.t_step % cfg.demand_relocate_interval == 0 \
                and self.t_step > 0:
            j = torch.randint(0, self.H, (self.E,), device=self.device, generator=g)
            new_xy = torch.rand(self.E, 2, device=self.device, generator=g)
            new_xy[:, 0] *= cfg.arena_w
            new_xy[:, 1] *= cfg.arena_h
            idx = torch.arange(self.E, device=self.device)
            self.centers[idx, j] = new_xy
        self.demand = self._demand_from_state()

    # ------------------------------------------------------------------ #
    def _connectivity(self):
        """Erreichbarkeit jeder Drohne von der Basis ueber die Funk-Kette."""
        pos, alive = self.world.pos, self.world.alive
        E, A = self.E, self.A
        dd = torch.cdist(pos, pos)
        eye = torch.eye(A, device=self.device, dtype=torch.bool)
        adj = (dd <= self.radio) & ~eye
        adj = adj & alive.unsqueeze(1) & alive.unsqueeze(2)
        base_d = torch.cdist(pos, self.base.view(1, 1, 3).expand(E, 1, 3)).squeeze(-1)
        base_link = (base_d <= self.radio) & alive
        reach = base_link.clone()
        for _ in range(A):
            new = base_link | (adj & reach.unsqueeze(1)).any(dim=2)
            if torch.equal(new, reach):
                break
            reach = new
        return reach, base_d

    # ------------------------------------------------------------------ #
    def _footprint(self):
        xy = self.world.pos[..., :2]
        d = torch.cdist(xy, self.cell_xy.unsqueeze(0).expand(self.E, -1, -1))
        alt = (self.world.pos[..., 2] - self.cfg.min_altitude).clamp(min=0.0)
        fp = (self.cfg.footprint_base + self.cfg.footprint_alt_gain * alt).unsqueeze(-1)
        return d, fp

    # ------------------------------------------------------------------ #
    def _local_demand(self):
        """Nachfragewert an der Bodenzelle direkt unter jeder Drohne (E,A)."""
        xy = self.world.pos[..., :2]
        ix = (xy[..., 0] / self.cell_w).long().clamp(0, self.G - 1)
        iy = (xy[..., 1] / self.cell_h).long().clamp(0, self.G - 1)
        idx = ix * self.G + iy                                          # (E,A)
        local = torch.gather(self.demand, 1, idx)                       # (E,A)
        return local * self.GG   # auf ~1 im Mittel skalieren (relativ zur Dichte)

    # ------------------------------------------------------------------ #
    def step(self, actions: torch.Tensor):
        cfg = self.cfg
        thrust = self.dirs[actions] * cfg.max_thrust
        self.world.step(thrust)

        _, base_d = self._connectivity()
        speed_frac = (self.world.vel.norm(dim=-1) / cfg.max_speed).clamp(0.0, 1.0)
        near_base = base_d <= cfg.charge_radius
        drain = self.batt_drain + self.batt_move * speed_frac
        charged = (self.battery + cfg.battery_charge).clamp(max=1.0)
        self.battery = torch.where(near_base, charged, self.battery - drain)
        self.battery = self.battery.clamp(min=0.0)
        self.battery = torch.where(self.world.alive, self.battery, torch.zeros_like(self.battery))

        new_empty = self.world.alive & (self.battery <= 0.0)
        self.world.alive = self.world.alive & ~new_empty

        # Nachfrage bewegt sich, DANN Versorgung bewerten
        self._move_demand()
        obs, served_w, served_raw, reach = self._observe(return_metrics=True)

        # --- Belohnung + Shaping -----------------------------------------
        alive_f = self.world.alive.float()
        empty_frac = new_empty.float().sum(dim=1) / self.A
        move_cost = speed_frac.mean(dim=1)
        low = ((cfg.low_batt_threshold - self.battery).clamp(min=0.0) / cfg.low_batt_threshold)
        home_pen = ((low * (base_d / self.diag)) * alive_f).sum(1) / alive_f.sum(1).clamp(min=1.0)
        rew = (served_w * cfg.served_reward_scale
               - cfg.w_homecome * home_pen
               - cfg.empty_penalty * empty_frac
               - 0.03 * move_cost)                                      # (E,)
        reward = rew.unsqueeze(1).expand(self.E, self.T)

        self.t_step += 1
        done = self.t_step % cfg.episode_len == 0
        done_vec = torch.full((self.E,), bool(done), device=self.device, dtype=torch.bool)

        conn_frac = (reach.float().sum(dim=1) / alive_f.sum(dim=1).clamp(min=1.0)).mean().item()
        mean_batt = (self.battery * alive_f).sum() / alive_f.sum().clamp(min=1.0)
        info = {
            "served_frac": served_w.mean().item(),
            "served_raw_frac": served_raw.mean().item(),
            "connected_frac": conn_frac,
            "alive_frac": alive_f.mean().item(),
            "mean_battery": mean_batt.item(),
            "empty_per_step": new_empty.float().sum(dim=1).mean().item(),
        }
        if done:
            obs = self.reset()
        return obs, reward, done_vec, info

    # ------------------------------------------------------------------ #
    def _observe(self, return_metrics: bool = False):
        cfg = self.cfg
        pos, vel, alive = self.world.pos, self.world.vel, self.world.alive
        reach, _ = self._connectivity()
        d, fp = self._footprint()

        usable = (alive & reach).unsqueeze(-1)
        within = (d <= fp) & usable
        served_now = within.any(dim=1)                                 # (E,GG)

        served_w = (served_now.float() * self.demand).sum(dim=1)       # (E,) gewichtet
        served_raw = served_now.float().mean(dim=1)                    # (E,) Flaeche

        # Richtung zur naechsten UNVERSORGTEN, stark nachgefragten Zelle
        gap = (~served_now) & (self.demand > self.demand.mean(dim=1, keepdim=True))
        big = torch.where(gap.unsqueeze(1), d, torch.full_like(d, 1e9))
        nj = big.argmin(dim=2)
        gap_xy = self.cell_xy[nj]
        gap3 = torch.cat([gap_xy, torch.full_like(gap_xy[..., :1], cfg.min_altitude)], dim=-1)
        to_gap = gap3 - pos
        d_gap = to_gap.norm(dim=-1, keepdim=True)
        dir_gap = to_gap / d_gap.clamp(min=1e-6)

        to_base = self.base.view(1, 1, 3) - pos
        d_base = to_base.norm(dim=-1, keepdim=True)
        dir_base = to_base / d_base.clamp(min=1e-6)

        pos_n = pos / self.hi
        vel_n = vel / cfg.max_speed
        batt = self.battery.unsqueeze(-1)
        local_dem = self._local_demand().unsqueeze(-1)                 # (E,A,1)
        entities = torch.cat([pos_n, vel_n, batt], dim=-1)             # (E,A,7)
        self_global = torch.cat(
            [dir_base, d_base / self.diag, batt, reach.float().unsqueeze(-1),
             dir_gap, d_gap / self.diag, local_dem], dim=-1,           # (E,A,11)
        )

        alive_f = alive.float()
        served_col = served_w.unsqueeze(-1)
        conn_col = (reach.float().sum(dim=1, keepdim=True) /
                    alive_f.sum(dim=1, keepdim=True).clamp(min=1.0))
        batt_col = ((self.battery * alive_f).sum(dim=1, keepdim=True) /
                    alive_f.sum(dim=1, keepdim=True).clamp(min=1.0))
        tg = torch.cat([served_col, conn_col, batt_col], dim=-1)
        team_global = tg.unsqueeze(1).expand(self.E, self.T, 3)

        obs = {
            "entities": entities,
            "alive": alive,
            "team": self.world.team,
            "self_global": self_global,
            "team_global": team_global,
        }
        return obs, served_w, served_raw, reach

    # ------------------------------------------------------------------ #
    def agent_team_index(self) -> torch.Tensor:
        return self.team_row
