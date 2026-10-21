"""Standbild der Krisen-Relais-Mission (Top-Down, headless).

Laesst eine Policy (oder Zufall) eine Weile fliegen und rendert dann einen
aussagekraeftigen Frame:

  * Hintergrund  = Bevoelkerungsdichte (Demand-Hotspots)
  * gruene Tönung = aktuell VERSORGTE Bodenflaeche
  * Stern        = Basisstation (Internet-Tor)
  * Punkte       = Drohnen; gruen = mit Basis verbunden, rot = abgehaengt
                   Groesse ~ Flughoehe, Transparenz ~ Akkustand
  * graue Linien = aktive Funkverbindungen (Drohne-Drohne und Drohne-Basis)

  python show_relay.py                         # Zufallspolicy
  python show_relay.py checkpoints/relay_final.pt
  python show_relay.py checkpoints/relay_final.pt --steps 120 --out out/relay.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from swarm_sim import SimConfig, TrainConfig
from swarm_sim.env_relay import RelayEnv3D, NUM_ACTIONS
from swarm_sim.attn import AttnActorCritic


def _links(pos, alive, base, radio):
    """Liniensegmente aller aktiven Funkverbindungen fuer einen Frame.

    pos (A,3) numpy, alive (A,) bool, base (3,) numpy. Rueckgabe (segments, 2, 2)
    in (x,y). Drohne-Drohne und Drohne-Basis innerhalb radio.
    """
    segs = []
    A = pos.shape[0]
    for i in range(A):
        if not alive[i]:
            continue
        if np.linalg.norm(pos[i] - base) <= radio:
            segs.append([pos[i, :2], base[:2]])
        for j in range(i + 1, A):
            if alive[j] and np.linalg.norm(pos[i] - pos[j]) <= radio:
                segs.append([pos[i, :2], pos[j, :2]])
    return np.array(segs) if segs else np.zeros((0, 2, 2))


def render(checkpoint: str | None, steps: int, agents: int, out_png: str | Path, seed: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    cfg = SimConfig(num_envs=1, num_agents=agents, num_teams=1, seed=seed)
    env = RelayEnv3D(cfg)
    dev = env.device

    net = None
    if checkpoint:
        net = AttnActorCritic(env.entity_dim, env.self_global_dim, env.team_global_dim,
                              NUM_ACTIONS, hidden=TrainConfig().hidden).to(dev)
        net.load_state_dict(torch.load(checkpoint, map_location=dev))
        net.eval()

    obs = env.reset()
    for _ in range(steps):
        if net is not None:
            with torch.no_grad():
                dist = net.policy(obs["entities"], obs["alive"], obs["team"], obs["self_global"])
                action = dist.probs.argmax(dim=-1)   # deterministisch: beste Aktion
        else:
            action = torch.randint(0, NUM_ACTIONS, (env.E, env.A), device=dev)
        obs, _, _, info = env.step(action)

    # Zustand des ersten (einzigen) Environments auslesen
    reach, _ = env._connectivity()
    d, fp = env._footprint()
    usable = (env.world.alive & reach).unsqueeze(-1)
    served_now = ((d <= fp) & usable).any(dim=1)[0].cpu().numpy()        # (GG,)
    demand = env.demand[0].cpu().numpy()                                 # (GG,)
    pos = env.world.pos[0].cpu().numpy()                                 # (A,3)
    alive = env.world.alive[0].cpu().numpy()                             # (A,)
    batt = env.battery[0].cpu().numpy()                                  # (A,)
    reach0 = reach[0].cpu().numpy()                                      # (A,)
    base = env.base.cpu().numpy()
    G = env.G

    dem_img = demand.reshape(G, G).T       # (y,x) fuer imshow
    srv_img = served_now.reshape(G, G).T

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, cfg.arena_w); ax.set_ylim(0, cfg.arena_h); ax.set_aspect("equal")
    extent = [0, cfg.arena_w, 0, cfg.arena_h]
    ax.imshow(dem_img, extent=extent, origin="lower", cmap="Greys", alpha=0.85)
    # versorgte Flaeche gruen ueberlagern
    srv_rgba = np.zeros((G, G, 4))
    srv_rgba[..., 1] = 1.0
    srv_rgba[..., 3] = srv_img * 0.35
    ax.imshow(srv_rgba, extent=extent, origin="lower")

    segs = _links(pos, alive, base, env.radio)
    if len(segs):
        ax.add_collection(LineCollection(segs, colors="#555555", linewidths=0.8, alpha=0.6, zorder=2))

    ax.scatter(base[0], base[1], marker="*", s=600, c="gold", edgecolors="black",
               linewidths=1.2, zorder=4, label="Basisstation")
    for status, col, lab in ((True, "#2ca02c", "verbunden"), (False, "#d62728", "abgehaengt")):
        m = alive & (reach0 == status)
        if m.any():
            sizes = 60 + 300 * (pos[m, 2] - cfg.min_altitude) / max(cfg.arena_d - cfg.min_altitude, 1e-6)
            ax.scatter(pos[m, 0], pos[m, 1], s=sizes, c=col, edgecolors="black",
                       linewidths=0.6, alpha=0.5 + 0.5 * batt[m], zorder=3, label=lab)
    dead = ~alive
    if dead.any():
        ax.scatter(pos[dead, 0], pos[dead, 1], marker="x", s=70, c="black", zorder=3, label="ausgefallen")

    served_w = float((served_now * demand).sum())
    ax.set_title(f"Krisen-Relais 3D  |  versorgt (gewichtet) {served_w:.0%}  "
                 f"|  verbunden {reach0[alive].mean() if alive.any() else 0:.0%}  "
                 f"|  O Akku {batt[alive].mean() if alive.any() else 0:.0%}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> None:
    p = argparse.ArgumentParser(description="Standbild der Krisen-Relais-Mission")
    p.add_argument("checkpoint", nargs="?", default=None, help="Pfad zu relay_*.pt (optional)")
    p.add_argument("--steps", type=int, default=100, help="Schritte vor dem Rendern")
    p.add_argument("--agents", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="out/relay.png")
    args = p.parse_args()
    out = render(args.checkpoint, args.steps, args.agents, args.out, args.seed)
    print(f"Standbild gespeichert: {out}")


if __name__ == "__main__":
    main()
