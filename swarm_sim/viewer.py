"""Viewer mit Mesh-Overlay (matplotlib).

Darstellung:
  * Form    = Team      (Kreis, Quadrat, Dreieck, Raute)
  * Farbe   = Gruppe     (faerbt sich live um, wenn Schwaerme sich teilen/treffen)
  * schwarzer Ring = Anfuehrer der Gruppe
  * Linien  = Funkverbindungen (Mesh)

Drei Einstiege:
  * live(...)        - laufende Simulation in Echtzeit
  * play_replay(...) - gespeichertes .npz-Replay abspielen
  * snapshot(...)    - einen Frame als PNG speichern (headless / Thumbnail)

Das Mesh wird in jedem Frame aus den Positionen neu berechnet, damit auch
Replays (die nur Positionen speichern) das Netz zeigen.

CLI:  python -m swarm_sim.viewer <replay.npz>
      python -m swarm_sim.viewer <replay.npz> --snapshot bild.png --frame 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import SimConfig
from .mesh import MeshNetwork

# Form je Team, Farbpalette je Gruppe.
_TEAM_MARKERS = ["o", "s", "^", "D"]
_GROUP_CMAP = "tab20"


def _group_colors(labels: np.ndarray):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(_GROUP_CMAP)
    return np.array([cmap(int(l) % cmap.N) for l in labels])


def _mesh_for(pos_f: np.ndarray, team_f: np.ndarray, net: MeshNetwork):
    """Berechnet Mesh fuer einen einzelnen Frame (A,2)/(A,) -> numpy-Arrays."""
    pos = torch.from_numpy(pos_f).unsqueeze(0).float()
    team = torch.from_numpy(team_f).unsqueeze(0).long()
    st = net.compute(pos, team)
    return (
        st.adjacency[0].numpy(),
        st.labels[0].numpy(),
        st.leader[0].numpy(),
    )


def _edge_segments(pos_f: np.ndarray, adj: np.ndarray):
    """Liniensegmente fuer alle Verbindungen (nur obere Dreiecksmatrix)."""
    iu, ju = np.triu_indices(adj.shape[0], k=1)
    sel = adj[iu, ju]
    i, j = iu[sel], ju[sel]
    return np.stack([pos_f[i], pos_f[j]], axis=1)  # (num_edges, 2, 2)


def _draw_frame(ax, pos_f, team_f, adj, labels, leader, teams, w, h, title):
    """Zeichnet einen kompletten Frame (ax wird vorher geleert)."""
    from matplotlib.collections import LineCollection

    ax.clear()
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_title(title)

    # Mesh-Kanten
    segs = _edge_segments(pos_f, adj)
    if len(segs):
        ax.add_collection(LineCollection(segs, colors="#888888", linewidths=0.8, alpha=0.5, zorder=1))

    colors = _group_colors(labels)
    leaders = set(int(x) for x in np.unique(leader))
    for t in teams:
        idx = np.where(team_f == t)[0]
        if len(idx) == 0:
            continue
        marker = _TEAM_MARKERS[int(t) % len(_TEAM_MARKERS)]
        sizes = np.array([170 if int(k) in leaders else 55 for k in idx])
        edges = np.array(["black" if int(k) in leaders else "white" for k in idx])
        lws = np.array([1.6 if int(k) in leaders else 0.4 for k in idx])
        ax.scatter(
            pos_f[idx, 0], pos_f[idx, 1],
            marker=marker, c=colors[idx], s=sizes,
            edgecolors=edges, linewidths=lws, zorder=2,
        )


def play_replay(path: str | Path, interval_ms: int = 30, show_mesh: bool = True) -> None:
    """Spielt ein .npz-Replay ab und berechnet das Mesh pro Frame neu."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    from .recorder import ReplayRecorder

    data = ReplayRecorder.load(path)
    pos = data["pos"]                      # (T, A, 2)
    team = data["team"]                    # (A,)
    w, h = data["arena"]
    T, A, _ = pos.shape
    teams = sorted(set(int(t) for t in team))

    cfg = SimConfig(num_envs=1, num_agents=A, device="cpu", radio_radius=data["radio_radius"])
    net = MeshNetwork(cfg, torch.device("cpu"))

    fig, ax = plt.subplots(figsize=(7, 7))

    def update(f: int):
        adj, labels, leader = _mesh_for(pos[f], team, net)
        if not show_mesh:
            adj = np.zeros_like(adj)
        _draw_frame(ax, pos[f], team, adj, labels, leader, teams, w, h,
                    f"Replay  Frame {f+1}/{T}")

    _anim = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=False)
    plt.show()


def live(cfg: SimConfig, world, controller, steps: int = 3000, interval_ms: int = 20,
         show_mesh: bool = True):
    """Zeigt eine laufende Simulation (Environment 0) live mit Mesh-Overlay."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    net = MeshNetwork(cfg, world.device)
    teams = list(range(cfg.num_teams))
    team0 = world.team[0].detach().to("cpu").numpy()

    fig, ax = plt.subplots(figsize=(7, 7))
    counter = {"i": 0}

    def update(_f: int):
        i = counter["i"]
        controller.wander_goals(i)
        actions = controller.act(world.pos, world.vel, world.team)
        world.step(actions)
        counter["i"] = i + 1

        st = net.compute(world.pos, world.team)
        pos_f = world.pos[0].detach().to("cpu").numpy()
        adj = st.adjacency[0].detach().to("cpu").numpy() if show_mesh else \
            np.zeros((cfg.num_agents, cfg.num_agents), bool)
        labels = st.labels[0].detach().to("cpu").numpy()
        leader = st.leader[0].detach().to("cpu").numpy()
        _draw_frame(ax, pos_f, team0, adj, labels, leader, teams,
                    cfg.arena_w, cfg.arena_h, "Schwarm-Sandbox (live, Mesh)")

    _anim = FuncAnimation(fig, update, frames=steps, interval=interval_ms, blit=False)
    plt.show()


def snapshot(path: str | Path, frame: int, out_png: str | Path, show_mesh: bool = True) -> Path:
    """Rendert einen einzelnen Replay-Frame als PNG (headless-tauglich)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .recorder import ReplayRecorder

    data = ReplayRecorder.load(path)
    pos, team = data["pos"], data["team"]
    w, h = data["arena"]
    T, A, _ = pos.shape
    frame = max(0, min(frame, T - 1))
    teams = sorted(set(int(t) for t in team))

    cfg = SimConfig(num_envs=1, num_agents=A, device="cpu", radio_radius=data["radio_radius"])
    net = MeshNetwork(cfg, torch.device("cpu"))
    adj, labels, leader = _mesh_for(pos[frame], team, net)
    if not show_mesh:
        adj = np.zeros_like(adj)

    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_frame(ax, pos[frame], team, adj, labels, leader, teams, w, h,
                f"Frame {frame+1}/{T}")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _main() -> None:
    p = argparse.ArgumentParser(description="Replay-Viewer mit Mesh-Overlay")
    p.add_argument("replay", help="Pfad zur .npz-Replaydatei")
    p.add_argument("--no-mesh", action="store_true", help="Mesh-Verbindungen ausblenden")
    p.add_argument("--snapshot", type=str, default=None, help="statt Abspielen einen Frame als PNG")
    p.add_argument("--frame", type=int, default=0, help="Frame fuer --snapshot")
    args = p.parse_args()

    if args.snapshot:
        out = snapshot(args.replay, args.frame, args.snapshot, show_mesh=not args.no_mesh)
        print(f"Snapshot gespeichert: {out}")
    else:
        play_replay(args.replay, show_mesh=not args.no_mesh)


if __name__ == "__main__":
    _main()
