"""Standbild des Netzabdeckungs-Vollmodells (Top-Down, headless).

Laesst eine Policy (oder Zufall) fliegen und rendert einen Frame mit allem:

  * graue Kreise  = Gelaende/Hindernisse (daempfen Funk, Kollision)
  * blaue Quadrate = Ladestationen (Akku)
  * goldener Stern = Uplink (Internet-Tor)
  * farbige Kreise = Bedarfs-Hotspots (Farbe=Typ, Groesse=Bedarf,
                     roter Ring = aktuell unversorgt); versorgte heller
  * Punkte        = Drohnen (gruen=mit Uplink verbunden, rot=abgehaengt,
                    Groesse ~ Flughoehe, Transparenz ~ Akku)
  * graue Linien  = aktive Funkverbindungen
  * Pfeil         = Windrichtung/-staerke

  python show_netcov.py                          # Zufallspolicy, Stufe 4
  python show_netcov.py checkpoints/netcov_final.pt --stage 4 --steps 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from swarm_sim import TrainConfig
from swarm_sim.mission_config import MissionConfig
from swarm_sim.env_netcov import NetCovEnv, NUM_ACTIONS, _TYPES
from swarm_sim.attn import AttnActorCritic

_TYPE_COLORS = {"notunterkunft": "#1f77b4", "einsatzleitung": "#d62728",
                "krankenhaus": "#2ca02c", "wandernde_menge": "#ff7f0e"}


def _links_xy(pos, alive, up, radius):
    segs = []
    A = pos.shape[0]
    for i in range(A):
        if not alive[i]:
            continue
        for u in range(up.shape[0]):
            if np.linalg.norm(pos[i, :2] - up[u]) <= radius:
                segs.append([pos[i, :2], up[u]])
        for j in range(i + 1, A):
            if alive[j] and np.linalg.norm(pos[i, :2] - pos[j, :2]) <= radius:
                segs.append([pos[i, :2], pos[j, :2]])
    return np.array(segs) if segs else np.zeros((0, 2, 2))


def render(checkpoint, stage, steps, out_png, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    mcfg = MissionConfig.load("netzabdeckung_config.yaml", stage=stage)
    env = NetCovEnv(mcfg, num_envs=1, device="cuda", seed=seed)
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
                action = dist.probs.argmax(dim=-1)
        else:
            action = torch.randint(0, NUM_ACTIONS, (env.E, env.A), device=dev)
        obs, _, _, info = env.step(action)

    _, connected, cap_factor = env._connectivity()
    served_frac, delivered, hole, _ = env._serve(connected, cap_factor)

    pos = env.pos[0].cpu().numpy()
    alive = env.alive[0].cpu().numpy()
    batt = (env.battery[0] / 100.0).cpu().numpy()
    conn = connected[0].cpu().numpy()
    up = env.up_pos[0].cpu().numpy()
    st = env.st_pos[0].cpu().numpy(); st_a = env.st_active[0].cpu().numpy()
    ob = env.ob_pos[0].cpu().numpy(); ob_r = env.ob_r[0].cpu().numpy(); ob_a = env.ob_active[0].cpu().numpy()
    hs = env.hs_pos[0].cpu().numpy(); hs_a = env.hs_active[0].cpu().numpy()
    hs_t = env.hs_type[0].cpu().numpy(); hs_d = env.hs_demand[0].cpu().numpy()
    hole0 = hole[0].cpu().numpy()
    ratio0 = (delivered[0] / env.hs_demand[0]).clamp(0, 1).cpu().numpy()
    wind = (env.wind[0] + env.gust[0]).cpu().numpy()
    W, H = env.map_w, env.map_h

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")

    for o in range(len(ob)):
        if ob_a[o]:
            ax.add_patch(plt.Circle(ob[o], ob_r[o], color="#999999", alpha=0.35, zorder=1))

    segs = _links_xy(pos, alive, up, env.dd_range)
    if len(segs):
        ax.add_collection(LineCollection(segs, colors="#777777", linewidths=0.6, alpha=0.4, zorder=2))

    # Hotspots
    for h in range(len(hs)):
        if not hs_a[h]:
            continue
        name = _TYPES[int(hs_t[h])]
        col = _TYPE_COLORS[name]
        size = 120 + 500 * (hs_d[h] / max(hs_d.max(), 1e-6))
        edge = "red" if hole0[h] else "black"
        lw = 2.2 if hole0[h] else 0.7
        ax.scatter(hs[h, 0], hs[h, 1], s=size, c=col, alpha=0.3 + 0.6 * ratio0[h],
                   edgecolors=edge, linewidths=lw, marker="o", zorder=3)

    # Stationen + Uplink
    for s in range(len(st)):
        if st_a[s]:
            ax.scatter(st[s, 0], st[s, 1], marker="s", s=260, c="#1f77b4",
                       edgecolors="black", linewidths=1.0, zorder=4)
    ax.scatter(up[:, 0], up[:, 1], marker="*", s=700, c="gold",
               edgecolors="black", linewidths=1.2, zorder=5)

    # Drohnen
    for status, col in ((True, "#2ca02c"), (False, "#d62728")):
        m = alive & (conn == status)
        if m.any():
            sizes = 40 + 160 * (pos[m, 2] - env.alt_lo) / max(env.alt_hi - env.alt_lo, 1e-6)
            ax.scatter(pos[m, 0], pos[m, 1], s=sizes, c=col, edgecolors="black",
                       linewidths=0.5, alpha=0.4 + 0.6 * batt[m], zorder=6)

    # Windpfeil
    wn = float(np.linalg.norm(wind))
    if wn > 0.1:
        ax.annotate("", xy=(0.12 * W + wind[0] / max(wn, 1e-6) * 0.07 * W,
                            0.92 * H + wind[1] / max(wn, 1e-6) * 0.07 * H),
                    xytext=(0.12 * W, 0.92 * H),
                    arrowprops=dict(facecolor="black", width=2, headwidth=9))
    ax.text(0.12 * W, 0.95 * H, f"Wind {wn:.1f} m/s", fontsize=9)

    served_pct = served_frac[0].item() * 100
    conn_pct = conn[alive].mean() * 100 if alive.any() else 0
    batt_pct = batt[alive].mean() * 100 if alive.any() else 0
    ax.set_title(f"Netzabdeckung (Stufe {stage})  |  Abdeckung {served_pct:.0f}%  "
                 f"|  verbunden {conn_pct:.0f}%  |  O Akku {batt_pct:.0f}%  "
                 f"|  aktiv {int(alive.sum())} Drohnen")
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    p = argparse.ArgumentParser(description="Standbild Netzabdeckung-Vollmodell")
    p.add_argument("checkpoint", nargs="?", default=None)
    p.add_argument("--stage", type=int, default=4)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="out/netcov.png")
    args = p.parse_args()
    out = render(args.checkpoint, args.stage, args.steps, args.out, args.seed)
    print(f"Standbild gespeichert: {out}")


if __name__ == "__main__":
    main()
