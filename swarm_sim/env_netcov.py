"""Netzabdeckung (Vollmodell, 3D, echte Einheiten) - Phase 7.

Umsetzung der netzabdeckung_config.yaml. Ein Drohnenschwarm haelt ein
fliegendes Funknetz ueber einer Krisenregion aufrecht und versorgt
Bedarfs-Hotspots (Notunterkunft, Einsatzleitung, Krankenhaus, wandernde
Menge) mit Datenkapazitaet, die ueber einen UPLINK (Satellit/Netzknoten)
ins Internet gereicht wird. LADESTATIONEN (getrennt vom Uplink) laden den
Akku. WETTER (Wind/Boeen) und GELAENDE (Hindernisse, die Funk daempfen und
Kollisionen verursachen) variieren pro Episode.

Alle Zahlen stammen aus der YAML (MissionConfig) - nichts ist hart kodiert.
Variable Anzahlen (Drohnen 12-20, Hotspots 5-8, Stationen 2-3) werden ueber
feste Max-Puffer + Aktiv-Masken realisiert, damit die Tensor-Formen im Batch
konstant bleiben.

Zeitschritt = 1/steuerung_hz (eine Aktion = eine Steuerentscheidung).

Modell-Kurzform (bewusste Vereinfachungen sind kommentiert):
  Signalstaerke s(d) = max(0, 1-(d/reichweite)^2); Link wenn s>=min_signal.
  Gelaende daempft s auf dem Pfad Drohne->Boden/Uplink (Faktor je Hindernis).
  Kapazitaet einer verbundenen Drohne = kap_pro_drohne*(1-hopverlust)^Hops.
  Hotspot versorgt = gelieferte Kapazitaet / Bedarf, gedeckelt vom Uplink.

obs = {entities (E,A,7), alive (E,A), team (E,A),
       self_global (E,A,13), team_global (E,T,3)}
"""

from __future__ import annotations

import math

import torch

from .mission_config import MissionConfig, as_range

# stehen + 8 Horizontalrichtungen + hoch + runter
_DIRS = [(0.0, 0.0, 0.0)]
for _k in range(8):
    _a = math.pi * _k / 4.0
    _DIRS.append((math.cos(_a), math.sin(_a), 0.0))
_DIRS += [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]
NUM_ACTIONS = len(_DIRS)

# Hotspot-Typen in fester Reihenfolge (Index -> YAML-Schluessel)
_TYPES = ["notunterkunft", "einsatzleitung", "krankenhaus", "wandernde_menge"]


def _u(gen, dev, *shape, lo=0.0, hi=1.0):
    return lo + (hi - lo) * torch.rand(*shape, device=dev, generator=gen)


def _rand_minutes(spec, gen, dev):
    lo, hi = as_range(spec)
    return float(lo + (hi - lo) * torch.rand(1, device=dev, generator=gen).item())


class NetCovEnv:
    entity_dim = 7         # pos_xy(2)+alt(1)+vel_xy(2)+vz(1)+battery(1)
    self_global_dim = 13   # dir_up(2)+d_up(1)+dir_st(2)+d_st(1)+dir_gap(2)+d_gap(1)+batt(1)+conn(1)+wind(2)
    team_global_dim = 3

    def __init__(self, mcfg: MissionConfig, num_envs: int, device: str = "cuda", seed: int = 0):
        self.m = mcfg
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.E = num_envs
        self.T = 1
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        # feste Max-Puffer (globale Obergrenzen ueber alle Stufen)
        self.A = int(mcfg.raw["drohne"]["anzahl_training"][1])          # 20
        self.H = int(mcfg.raw["bedarf"]["hotspots_anzahl"][1])          # 8
        self.S = int(mcfg.raw["ladestation"]["anzahl"][1])             # 3
        self.U = int(mcfg.funk["uplink"]["anzahl"])                    # 1
        self.O = 10                                                    # max Hindernisse

        self.dirs = torch.tensor(_DIRS, device=self.device, dtype=torch.float32)
        self._read_static_params()
        self.t_step = 0
        self.reset()

    # ------------------------------------------------------------------ #
    def _read_static_params(self) -> None:
        m = self.m
        self.map_w, self.map_h = m.map_size_m()
        self.alt_lo, self.alt_hi = m.altitude_m()
        self.dt = m.control_dt()
        self.max_speed = float(m.drone["max_geschwindigkeit_ms"])
        self.max_acc = float(m.drone["max_beschleunigung_ms2"])
        self.mass = float(m.drone["masse_kg"])
        self.fail_rate_h = float(m.drone["ausfallrate_pro_stunde"])
        # Funk
        self.cov_range = float(m.funk["versorgungsradius_boden_m"])    # bei ref-Hoehe
        self.dd_range = float(m.funk["verbindung_drohne_drohne_m"])
        self.min_sig = float(m.funk["min_signalstaerke_fuer_verbindung"])
        self.damp = float(m.funk["hindernis_daempfung"])
        self.cap_drone = float(m.funk["kapazitaet_pro_drohne_mbit"])
        self.hop_loss = float(m.funk["kapazitaetsverlust_pro_hop_prozent"]) / 100.0
        self.uplink_cap = float(m.funk["uplink"]["kapazitaet_mbit"])
        self.ref_alt = 100.0  # Referenzhoehe fuer den Bodenradius laut YAML-Kommentar
        # Akku
        self.start_lo, self.start_hi = as_range(m.akku["start_ladung_prozent"])
        # Reward
        r = m.reward
        self.r_cov = float(r["versorgte_kapazitaet"])
        self.r_hole = float(r["funkloch_prioritaets_hotspot"])
        self.r_crash = float(r["absturz_leerer_akku"])
        self.r_coll = float(r["kollision"])
        self.r_energy = float(r["energie"])
        self.r_reserve = float(r.get("akku_reserve", 0.0))             # Strafe unter Sicherheitsreserve
        self.r_staffel = float(r.get("staffel_bonus", 0.0))            # exp1: Bonus fuer Anteil einsatzbereiter (geladener) Drohnen
        self.reserve_pct = float(m.akku["sicherheitsreserve_prozent"])
        # Station
        self.charge_radius = 150.0  # m, Annahme: Naehe zur Station zum Laden
        self.slots = int(m.station["plaetze_pro_station"])
        self.charge_full_min = float(m.station["ladezeit_0_auf_100_min"])

    # ------------------------------------------------------------------ #
    def set_stage(self, stage: int) -> None:
        self.m.set_stage(stage)

    # ------------------------------------------------------------------ #
    def reset(self):
        m, dev, g = self.m, self.device, self.gen
        E, A, H, S, U, O = self.E, self.A, self.H, self.S, self.U, self.O
        self.t_step = 0

        # --- Episoden-Schwierigkeit / Flags aus Stufe ---
        self.flight_s = m.battery_minutes() * 60.0                     # Akku-Laufzeit, ENTKOPPELT von der Episode
        self.batt_basis = 100.0 / self.flight_s                        # %/s Grundverbrauch
        self.charge_rate = 100.0 / (m.charge_minutes() * 60.0)         # %/s (Modus voll/schnell/tausch)
        self.ep_steps = max(64, int(round(m.episode_minutes() * 60.0 / self.dt)))
        self.use_obstacles = m.obstacles_enabled()
        self.use_failures = m.failures_enabled()
        self.use_gusts = m.gust_enabled()
        self.shift_demand = m.demand_shift_enabled()

        # --- Drohnen (variable Anzahl pro Env) ---
        dlo, dhi = m.drones_range()
        n_drones = torch.randint(dlo, dhi + 1, (E,), device=dev, generator=g)
        self.alive = torch.arange(A, device=dev).expand(E, A) < n_drones.unsqueeze(1)
        self.pos = torch.zeros(E, A, 3, device=dev)
        self.pos[..., 0] = _u(g, dev, E, A, hi=self.map_w)
        self.pos[..., 1] = _u(g, dev, E, A, hi=self.map_h)
        self.pos[..., 2] = _u(g, dev, E, A, lo=self.alt_lo, hi=self.alt_hi)
        self.vel = torch.zeros(E, A, 3, device=dev)
        self.battery = _u(g, dev, E, A, lo=self.start_lo, hi=self.start_hi)

        # --- Hotspot-Positionen schon hier sampeln (fuer den Modus "beim_hotspot") ---
        self.hs_pos = torch.stack([_u(g, dev, E, H, hi=self.map_w),
                                   _u(g, dev, E, H, hi=self.map_h)], dim=-1)

        # --- Ladestationen (Platzierung config-gesteuert) ---
        slo, shi = m.stations_range()
        n_st = torch.randint(slo, shi + 1, (E,), device=dev, generator=g)
        self.st_active = torch.arange(S, device=dev).expand(E, S) < n_st.unsqueeze(1)
        pos_mode = str(m.station.get("position", "kartenrand_zufaellig"))
        if pos_mode.startswith("kartenrand"):          # Rand der Karte (realistisch)
            self.st_pos = self._edge_points(E, S)
        elif pos_mode == "beim_hotspot":               # Station direkt beim Hotspot -> laden beim Versorgen
            idx = torch.arange(S, device=dev) % H
            jit = (torch.rand(E, S, 2, device=dev, generator=g) - 0.5) * (0.4 * self.charge_radius)
            self.st_pos = (self.hs_pos[:, idx, :] + jit)
            self.st_pos[..., 0] = self.st_pos[..., 0].clamp(0.0, self.map_w)
            self.st_pos[..., 1] = self.st_pos[..., 1].clamp(0.0, self.map_h)
        else:                                          # zufaellig ueber die ganze Karte
            self.st_pos = torch.stack([_u(g, dev, E, S, hi=self.map_w),
                                       _u(g, dev, E, S, hi=self.map_h)], dim=-1)

        # --- Uplink (zufaellige Position, voll aktiv) ---
        self.up_pos = torch.zeros(E, U, 2, device=dev)
        self.up_pos[..., 0] = _u(g, dev, E, U, hi=self.map_w)
        self.up_pos[..., 1] = _u(g, dev, E, U, hi=self.map_h)

        # --- Gelaende / Hindernisse ---
        if self.use_obstacles:
            n_ob = torch.randint(3, O + 1, (E,), device=dev, generator=g)
        else:
            n_ob = torch.zeros(E, dtype=torch.long, device=dev)
        self.ob_active = torch.arange(O, device=dev).expand(E, O) < n_ob.unsqueeze(1)
        self.ob_pos = torch.stack([_u(g, dev, E, O, hi=self.map_w),
                                   _u(g, dev, E, O, hi=self.map_h)], dim=-1)
        self.ob_r = _u(g, dev, E, O, lo=self.map_w * 0.02, hi=self.map_w * 0.06)

        # --- Wetter: Grundwind pro Env (Betrag + Richtung) ---
        wlo, whi = m.wind_ms()
        wmag = _u(g, dev, E, lo=wlo, hi=whi)
        wang = _u(g, dev, E, lo=0.0, hi=2 * math.pi)
        self.wind = torch.stack([wmag * wang.cos(), wmag * wang.sin()], dim=-1)   # (E,2)
        self.gust = torch.zeros(E, 2, device=dev)

        # --- Bedarf (Hotspots mit Typ/Prioritaet/Bedarf) ---
        hlo, hhi = m.hotspots_range()
        n_hs = torch.randint(hlo, hhi + 1, (E,), device=dev, generator=g)
        self.hs_active = torch.arange(H, device=dev).expand(E, H) < n_hs.unsqueeze(1)
        # hs_pos wurde bereits oben gesampelt (fuer die Stationsplatzierung)
        self._sample_hotspot_types()
        self.relocate_every = max(1, int(round(
            _rand_minutes(m.bedarf["neu_verteilen_alle_min"], g, dev) / (self.dt / 60.0))))

        return self._observe()[0]

    # ------------------------------------------------------------------ #
    def _edge_points(self, E, N) -> torch.Tensor:
        """N Punkte je Env an einem zufaelligen Kartenrand (in Metern)."""
        g, dev = self.gen, self.device
        side = torch.randint(0, 4, (E, N), device=dev, generator=g)
        t = _u(g, dev, E, N)
        zeros = torch.zeros_like(t)
        x = torch.where(side <= 1, t * self.map_w,
                        torch.where(side == 2, zeros, torch.full_like(t, self.map_w)))
        y = torch.where(side == 0, zeros,
                        torch.where(side == 1, torch.full_like(t, self.map_h), t * self.map_h))
        return torch.stack([x, y], dim=-1)

    # ------------------------------------------------------------------ #
    def _sample_hotspot_types(self) -> None:
        """Weist jedem Hotspot einen Typ zu und berechnet Bedarf/Prioritaet."""
        m, g, dev = self.m, self.gen, self.device
        E, H = self.E, self.H
        tsel = torch.randint(0, len(_TYPES), (E, H), device=dev, generator=g)
        self.hs_type = tsel
        demand = torch.zeros(E, H, device=dev)
        prio = torch.ones(E, H, device=dev)
        vel = torch.zeros(E, H, 2, device=dev)
        for ti, name in enumerate(_TYPES):
            spec = m.bedarf["typen"][name]
            ulo, uhi = as_range(spec["nutzer"])
            users = _u(g, dev, E, H, lo=ulo, hi=uhi)
            mbit = float(spec["mbit_pro_nutzer"])
            p = float(spec["prioritaet"])
            sel = (tsel == ti)
            demand = torch.where(sel, users * mbit, demand)
            prio = torch.where(sel, torch.full_like(prio, p), prio)
            if "geschwindigkeit_ms" in spec:   # wandernde Menge bewegt sich
                vlo, vhi = as_range(spec["geschwindigkeit_ms"])
                ang = _u(g, dev, E, H, lo=0.0, hi=2 * math.pi)
                spd = _u(g, dev, E, H, lo=vlo, hi=vhi)
                mv = torch.stack([spd * ang.cos(), spd * ang.sin()], dim=-1)
                vel = torch.where(sel.unsqueeze(-1), mv, vel)
        self.hs_demand = demand.clamp(min=1e-3)
        self.hs_prio = prio
        self.hs_vel = vel

    # ------------------------------------------------------------------ #
    def _blocking(self, src_xy, dst_xy):
        """Daempfungsfaktor auf dem Pfad src->dst durch Hindernisse.

        src_xy (E,A,2), dst_xy (E,K,2) -> faktor (E,A,K) in (0,1].
        Faktor = damp^(#Hindernisse, deren Footprint das 2D-Segment schneidet).
        """
        if not self.use_obstacles:
            return None
        E, A, _ = src_xy.shape
        K = dst_xy.shape[1]
        P = src_xy.unsqueeze(2)                          # (E,A,1,2)
        Q = dst_xy.unsqueeze(1)                          # (E,1,K,2)
        seg = Q - P                                      # (E,A,K,2)
        seg_len2 = (seg ** 2).sum(-1, keepdim=True).clamp(min=1e-6)
        C = self.ob_pos.view(E, 1, 1, self.O, 2)         # (E,1,1,O,2)
        t = ((C - P.unsqueeze(3)) * seg.unsqueeze(3)).sum(-1, keepdim=True) / seg_len2.unsqueeze(3)
        t = t.clamp(0.0, 1.0)
        closest = P.unsqueeze(3) + t * seg.unsqueeze(3)  # (E,A,K,O,2)
        dist = (closest - C).norm(dim=-1)                # (E,A,K,O)
        hit = (dist < self.ob_r.view(E, 1, 1, self.O)) & self.ob_active.view(E, 1, 1, self.O)
        n_hit = hit.float().sum(-1)                      # (E,A,K)
        return self.damp ** n_hit

    # ------------------------------------------------------------------ #
    def _connectivity(self):
        """Mesh-Hops von jeder Drohne zum naechsten Uplink.

        return: hops (E,A) float, connected (E,A) bool, cap_factor (E,A)
        """
        pos_xy = self.pos[..., :2]
        E, A = self.E, self.A
        dd = torch.cdist(pos_xy, pos_xy)                 # (E,A,A)
        sig = (1 - (dd / self.dd_range) ** 2).clamp(min=0.0)
        eye = torch.eye(A, device=self.device, dtype=torch.bool)
        link = (sig >= self.min_sig) & ~eye & self.alive.unsqueeze(1) & self.alive.unsqueeze(2)

        du = torch.cdist(pos_xy, self.up_pos)            # (E,A,U)
        sig_u = (1 - (du / self.dd_range) ** 2).clamp(min=0.0)
        damp_u = self._blocking(pos_xy, self.up_pos)
        if damp_u is not None:
            sig_u = sig_u * damp_u
        up_link = ((sig_u >= self.min_sig).any(dim=2)) & self.alive

        INF = 1e9
        depth = torch.where(up_link, torch.zeros(E, A, device=self.device),
                            torch.full((E, A), INF, device=self.device))
        for _ in range(A):
            nb = torch.where(link, depth.unsqueeze(1), torch.full_like(depth.unsqueeze(1), INF))
            nb_min = nb.min(dim=2).values
            new = torch.minimum(depth, nb_min + 1.0)
            new = torch.where(up_link, torch.zeros_like(new), new)
            if torch.equal(new, depth):
                break
            depth = new
        connected = depth < INF / 2
        cap_factor = torch.where(connected, (1 - self.hop_loss) ** depth.clamp(max=A),
                                 torch.zeros_like(depth))
        return depth, connected, cap_factor

    # ------------------------------------------------------------------ #
    def _serve(self, connected, cap_factor):
        """Belieferte Kapazitaet je Hotspot + gewichteter Versorgungsgrad.

        return: served_frac (E,), delivered (E,H), hole (E,H) bool, covers (E,A,H)
        """
        pos_xy = self.pos[..., :2]
        alt = self.pos[..., 2]
        cov = self.cov_range * (alt / self.ref_alt).clamp(min=0.3)     # (E,A) hoehenabh. Bodenradius
        d_hs = torch.cdist(pos_xy, self.hs_pos)                        # (E,A,H)
        sig = (1 - (d_hs / cov.unsqueeze(-1)) ** 2).clamp(min=0.0)
        damp = self._blocking(pos_xy, self.hs_pos)
        if damp is not None:
            sig = sig * damp
        covers = (sig >= self.min_sig) & connected.unsqueeze(-1) & self.alive.unsqueeze(-1)  # (E,A,H)

        cap_ah = covers.float() * (self.cap_drone * cap_factor).unsqueeze(-1)   # (E,A,H)
        supply = cap_ah.sum(dim=1)                                     # (E,H)
        delivered = torch.minimum(self.hs_demand, supply) * self.hs_active.float()
        total = delivered.sum(dim=1, keepdim=True)
        scale = (self.U * self.uplink_cap / total.clamp(min=1e-6)).clamp(max=1.0)
        delivered = delivered * scale

        ratio = (delivered / self.hs_demand).clamp(0.0, 1.0) * self.hs_active.float()
        w = self.hs_prio * self.hs_active.float()
        served_frac = (ratio * w).sum(dim=1) / w.sum(dim=1).clamp(min=1e-6)
        hole = self.hs_active & (supply <= 1e-6)
        return served_frac, delivered, hole, covers

    # ------------------------------------------------------------------ #
    def step(self, actions: torch.Tensor):
        dev = self.device
        E, A = self.E, self.A
        alive_f = self.alive.float().unsqueeze(-1)

        # --- Boeen aktualisieren ---
        if self.use_gusts:
            p = float(self.m.umwelt["boeen_wahrscheinlichkeit_pro_min"]) * (self.dt / 60.0)
            trigger = (torch.rand(E, device=dev, generator=self.gen) < p)
            glo, ghi = as_range(self.m.umwelt["boeen_ms"])
            gmag = _u(self.gen, dev, E, lo=glo, hi=ghi)
            gang = _u(self.gen, dev, E, lo=0.0, hi=2 * math.pi)
            new_g = torch.stack([gmag * gang.cos(), gmag * gang.sin()], dim=-1)
            self.gust = torch.where(trigger.unsqueeze(-1), new_g, self.gust * 0.9)
        wind_vec = self.wind + self.gust                               # (E,2)
        wind_mag = wind_vec.norm(dim=-1)                               # (E,)

        # --- Bewegung (Kinematik + Winddrift) ---
        acc = self.dirs[actions] * self.max_acc                        # (E,A,3)
        acc[..., :2] = acc[..., :2] + wind_vec.unsqueeze(1)
        self.vel = self.vel + acc * self.dt
        sp = self.vel.norm(dim=-1, keepdim=True)
        self.vel = self.vel * (self.max_speed / sp.clamp(min=1e-6)).clamp(max=1.0)
        self.vel = self.vel * alive_f
        self.pos = self.pos + self.vel * self.dt
        self.pos[..., 0] = self.pos[..., 0].clamp(0.0, self.map_w)
        self.pos[..., 1] = self.pos[..., 1].clamp(0.0, self.map_h)
        self.pos[..., 2] = self.pos[..., 2].clamp(self.alt_lo, self.alt_hi)

        # --- Kollisionen mit Hindernissen ---
        collided = torch.zeros(E, A, dtype=torch.bool, device=dev)
        if self.use_obstacles:
            d_ob = torch.cdist(self.pos[..., :2], self.ob_pos)         # (E,A,O)
            inside = (d_ob < self.ob_r.unsqueeze(1)) & self.ob_active.unsqueeze(1) & self.alive.unsqueeze(-1)
            collided = inside.any(dim=2)
            nearest = d_ob.argmin(dim=2)
            near_xy = torch.gather(self.ob_pos, 1, nearest.unsqueeze(-1).expand(E, A, 2))
            cvec = self.pos[..., :2] - near_xy
            cnorm = cvec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            r_near = torch.gather(self.ob_r, 1, nearest).unsqueeze(-1)
            push = (r_near - cnorm).clamp(min=0.0)
            self.pos[..., :2] = torch.where(collided.unsqueeze(-1),
                                            self.pos[..., :2] + (cvec / cnorm) * push, self.pos[..., :2])

        # --- Akku: Verbrauch nach YAML-Formel, Laden an Stationen ---
        v = self.vel.norm(dim=-1)                                      # m/s
        cons_s = self.batt_basis * (1 + 0.004 * v ** 2) * (1 + 0.05 * wind_mag.unsqueeze(1)) * 1.1
        drain = cons_s * self.dt
        charging = self._charging_mask()
        self.battery = torch.where(charging, (self.battery + self.charge_rate * self.dt).clamp(max=100.0),
                                   self.battery - drain)
        self.battery = (self.battery * self.alive.float()).clamp(min=0.0, max=100.0)

        # --- Ausfaelle: leerer Akku + zufaelliger Totalausfall ---
        empty = self.alive & (self.battery <= float(self.m.akku["absturz_bei_prozent"]))
        rand_fail = torch.zeros_like(empty)
        if self.use_failures:
            pstep = self.fail_rate_h * (self.dt / 3600.0)
            rand_fail = self.alive & (torch.rand(E, A, device=dev, generator=self.gen) < pstep)
        crashed = empty | rand_fail
        self.alive = self.alive & ~crashed

        # --- Bedarf bewegt/verlagert sich ---
        self._move_demand()

        # --- Funk/Versorgung bewerten ---
        _, connected, cap_factor = self._connectivity()
        served_frac, delivered, hole, covers = self._serve(connected, cap_factor)

        # --- Belohnung PRO AGENT (lokale Kreditzuweisung gegen "lazy agents") ---
        # Statt EINER geteilten Team-Belohnung bekommt jede Drohne ihren eigenen
        # Deckungsbeitrag + eigene Straffaktoren. Funkloch bleibt geteilt (Kooperation).
        prio_norm = self.hs_prio / self.hs_prio.clamp(min=1e-6).amax(dim=1, keepdim=True)   # (E,H)
        hs_w = (prio_norm * self.hs_active.float()).unsqueeze(1)                            # (E,1,H)
        local_cov = (covers.float() * hs_w).sum(dim=2)                                      # (E,A) eigener Beitrag
        holes = (hole.float() * prio_norm).sum(dim=1, keepdim=True)                         # (E,1) geteilt
        shortfall = (self.reserve_pct - self.battery).clamp(min=0.0) / max(self.reserve_pct, 1e-6)  # (E,A)
        # exp1 Staffel-Bonus: Anteil ALIVE-Drohnen UEBER der Reserve (= einsatzbereit), geteilt pro Arena.
        # Belohnt, immer ein paar geladene Drohnen in Reserve zu halten -> gegen kollektives Leerlaufen.
        ready = (self.battery >= self.reserve_pct) & self.alive                                   # (E,A)
        ready_frac = ready.float().sum(1, keepdim=True) / self.alive.float().sum(1, keepdim=True).clamp(min=1)  # (E,1) geteilt
        reward = (self.r_cov * local_cov                      # + eigener prioritaets-gew. Deckungsbeitrag
                  + self.r_hole * holes                       # - geteilte Funkloch-Strafe
                  + self.r_crash * crashed.float()            # - eigener Absturz
                  + self.r_coll * collided.float()            # - eigene Kollision
                  + self.r_reserve * shortfall * self.alive.float()   # - eigene Reserve-Unterschreitung
                  + self.r_staffel * ready_frac * self.alive.float()  # + geteilter Staffel-Bonus (exp1)
                  + self.r_energy * self.alive.float())       # - eigene Energie            -> (E,A)

        self.t_step += 1
        done = self.t_step % self.ep_steps == 0
        done_vec = torch.full((E,), bool(done), device=dev, dtype=torch.bool)

        alive_fr = self.alive.float()
        info = {
            "served_pct": served_frac.mean().item() * 100.0,
            "connected_frac": (connected.float().sum(1) / alive_fr.sum(1).clamp(min=1)).mean().item(),
            "crashes_per_step": crashed.float().sum(1).mean().item(),
            "collisions_per_step": collided.float().sum(1).mean().item(),
            "charging_now": charging.float().sum(1).mean().item(),
            "mean_battery": ((self.battery * alive_fr).sum() / alive_fr.sum().clamp(min=1)).item(),
            "ready_frac": ready_frac.mean().item(),
            "alive_frac": alive_fr.mean().item(),
            "holes_frac": (hole.float().sum(1) / self.hs_active.float().sum(1).clamp(min=1)).mean().item(),
            "wind_ms": wind_mag.mean().item(),
            "stage": self.m.stage,
        }
        obs = self._observe()[0]
        if done:
            obs = self.reset()
        return obs, reward, done_vec, info

    # ------------------------------------------------------------------ #
    def _charging_mask(self):
        """True fuer Drohnen, die gerade laden (nahe Station + freier Slot)."""
        d_st = torch.cdist(self.pos[..., :2], self.st_pos)            # (E,A,S)
        near = (d_st < self.charge_radius) & self.st_active.unsqueeze(1) & self.alive.unsqueeze(-1)
        charging = torch.zeros(self.E, self.A, dtype=torch.bool, device=self.device)
        for s in range(self.S):
            near_s = near[:, :, s]
            d = torch.where(near_s, d_st[:, :, s], torch.full_like(d_st[:, :, s], 1e9))
            order = d.argsort(dim=1)
            rank = torch.empty_like(order)
            rank.scatter_(1, order, torch.arange(self.A, device=self.device).expand(self.E, self.A))
            charging = charging | (near_s & (rank < self.slots))
        return charging

    # ------------------------------------------------------------------ #
    def _move_demand(self):
        if not self.shift_demand:
            return
        self.hs_pos = self.hs_pos + self.hs_vel * self.dt
        for ax, up in ((0, self.map_w), (1, self.map_h)):
            c = self.hs_pos[..., ax]
            v = self.hs_vel[..., ax]
            self.hs_vel[..., ax] = torch.where((c < 0) | (c > up), -v, v)
            self.hs_pos[..., ax] = c.clamp(0.0, up)
        if self.relocate_every > 0 and self.t_step > 0 and self.t_step % self.relocate_every == 0:
            j = torch.randint(0, self.H, (self.E,), device=self.device, generator=self.gen)
            idx = torch.arange(self.E, device=self.device)
            self.hs_pos[idx, j, 0] = _u(self.gen, self.device, self.E, hi=self.map_w)
            self.hs_pos[idx, j, 1] = _u(self.gen, self.device, self.E, hi=self.map_h)

    # ------------------------------------------------------------------ #
    def _nearest(self, targets_xy, active=None):
        """Richtung+Distanz zum naechsten (aktiven) Ziel je Drohne."""
        pos_xy = self.pos[..., :2]
        d = torch.cdist(pos_xy, targets_xy)                            # (E,A,K)
        if active is not None:
            d = torch.where(active.unsqueeze(1), d, torch.full_like(d, 1e12))
        j = d.argmin(dim=2)
        tgt = torch.gather(targets_xy, 1, j.unsqueeze(-1).expand(-1, -1, 2))
        vec = tgt - pos_xy
        dist = vec.norm(dim=-1, keepdim=True)
        return vec / dist.clamp(min=1e-6), dist

    # ------------------------------------------------------------------ #
    def _observe(self):
        _, connected, cap_factor = self._connectivity()
        served_frac, delivered, hole, _ = self._serve(connected, cap_factor)

        pos_xy = self.pos[..., :2]
        diag = math.hypot(self.map_w, self.map_h)
        dir_up, d_up = self._nearest(self.up_pos)
        dir_st, d_st = self._nearest(self.st_pos, self.st_active)
        unmet = self.hs_active & (hole | ((delivered / self.hs_demand) < 0.9))
        dir_gap, d_gap = self._nearest(self.hs_pos, unmet)

        alt_n = (self.pos[..., 2:3] - self.alt_lo) / max(self.alt_hi - self.alt_lo, 1e-6)
        pos_n = pos_xy / torch.tensor([self.map_w, self.map_h], device=self.device)
        vel_n = self.vel[..., :2] / self.max_speed
        vz_n = self.vel[..., 2:3] / self.max_speed
        batt_n = (self.battery / 100.0).unsqueeze(-1)
        wind_n = (self.wind + self.gust).unsqueeze(1).expand(self.E, self.A, 2) / max(self.max_speed, 1e-6)

        entities = torch.cat([pos_n, alt_n, vel_n, vz_n, batt_n], dim=-1)           # (E,A,7)
        self_global = torch.cat([
            dir_up, d_up / diag, dir_st, d_st / diag, dir_gap, d_gap / diag,
            batt_n, connected.float().unsqueeze(-1), wind_n,
        ], dim=-1)                                                                  # (E,A,13)

        alive_f = self.alive.float()
        conn_col = (connected.float().sum(1, keepdim=True) / alive_f.sum(1, keepdim=True).clamp(min=1))
        batt_col = ((self.battery / 100.0 * alive_f).sum(1, keepdim=True) / alive_f.sum(1, keepdim=True).clamp(min=1))
        tg = torch.cat([served_frac.unsqueeze(-1), conn_col, batt_col], dim=-1)     # (E,3)
        team_global = tg.unsqueeze(1).expand(self.E, self.T, 3)

        obs = {
            "entities": entities,
            "alive": self.alive,
            "team": torch.zeros(self.E, self.A, dtype=torch.long, device=self.device),
            "self_global": self_global,
            "team_global": team_global,
        }
        return obs, served_frac

    # ------------------------------------------------------------------ #
    def agent_team_index(self) -> torch.Tensor:
        return torch.zeros(self.A, dtype=torch.long, device=self.device)
