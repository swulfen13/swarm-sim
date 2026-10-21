"""Phase 1 - Sandbox ohne ML: Demo-Entry.

Startet die Schwarm-Simulation mit regelbasiertem Verhalten. Drei Modi:

  python run_sandbox.py live        # Simulation live zuschauen (matplotlib)
  python run_sandbox.py record      # eine Runde aufzeichnen -> replays/*.npz
  python run_sandbox.py bench       # GPU-Durchsatz messen (viele Arenen)

Nach "record" das Replay ansehen mit:
  python -m swarm_sim.viewer replays/<datei>.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from swarm_sim import (
    SimConfig,
    PhysicsWorld,
    BoidsController,
    ReplayRecorder,
    MeshNetwork,
    MetricsLogger,
)
from swarm_sim import viewer


def _build(cfg: SimConfig):
    world = PhysicsWorld(cfg)
    controller = BoidsController(cfg, world.device)
    return world, controller


def run_live(cfg: SimConfig, steps: int) -> None:
    cfg.num_envs = 1  # live zeigt genau eine Arena
    world, controller = _build(cfg)
    print(f"Live-Ansicht auf {world.device}. Fenster schliessen zum Beenden.")
    viewer.live(cfg, world, controller, steps=steps)


def run_record(cfg: SimConfig, steps: int, out: Path) -> Path:
    cfg.num_envs = max(cfg.num_envs, 1)
    world, controller = _build(cfg)
    rec = ReplayRecorder(cfg, env_index=0)
    rec.capture(world.state_cpu())
    for i in range(steps):
        controller.wander_goals(i)
        actions = controller.act(world.pos, world.vel, world.team)
        world.step(actions)
        rec.capture(world.state_cpu())
    path = rec.save(out)
    print(f"Replay gespeichert: {path}  ({rec.num_frames} Frames)")
    print(f"Ansehen mit:  python -m swarm_sim.viewer {path}")
    return path


def run_bench(cfg: SimConfig, steps: int) -> None:
    world, controller = _build(cfg)
    dev = world.device
    # Aufwaermen (JIT/CUDA-Kontext), dann messen.
    for _ in range(10):
        a = controller.act(world.pos, world.vel, world.team)
        world.step(a)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(steps):
        a = controller.act(world.pos, world.vel, world.team)
        world.step(a)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    env_steps = cfg.num_envs * steps
    agent_steps = env_steps * cfg.num_agents
    print(f"Geraet:              {dev}")
    print(f"Arenen (num_envs):   {cfg.num_envs}")
    print(f"Agenten/Arena:       {cfg.num_agents}")
    print(f"Schritte:            {steps}")
    print(f"Zeit:                {dt:.3f} s")
    print(f"Env-Steps/s:         {env_steps / dt:,.0f}")
    print(f"Agenten-Steps/s:     {agent_steps / dt:,.0f}")
    if dev.type == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"VRAM (max alloc):    {mem:.1f} MB")


def run_mesh(cfg: SimConfig, steps: int, log_every: int) -> None:
    """Phase 2: simuliert + berechnet das Mesh-Netz und loggt live ins

    lokale TensorBoard. Gruppen, Verbindungen und System-Stats landen unter
    runs/<zeit>/ und sind im Browser (auch am Handy) sichtbar.
    """
    world, controller = _build(cfg)
    mesh = MeshNetwork(cfg, world.device)
    agg = {}
    with MetricsLogger(run_name=f"mesh_{int(time.time())}") as log:
        print(f"Mesh-Lauf auf {world.device}, {cfg.num_envs} Arenen x {cfg.num_agents} Agenten.")
        print(f"TensorBoard: tensorboard --logdir runs   ->  http://localhost:6006")
        for i in range(steps):
            controller.wander_goals(i)
            actions = controller.act(world.pos, world.vel, world.team)
            world.step(actions)
            state = mesh.compute(world.pos, world.team)
            if i % log_every == 0:
                agg = MeshNetwork.group_stats(state)
                log.log(agg, i)
                log.log_throughput(i, cfg.num_envs)
                log.log_gpu(i)
        print(f"Fertig. Letzte Mesh-Werte (Mittel ueber Arenen): {agg}")
        print(f"Logs: {log.path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Schwarm-Sandbox Phase 1-2")
    p.add_argument("mode", choices=["live", "record", "bench", "mesh"], nargs="?", default="live")
    p.add_argument("--steps", type=int, default=None, help="Anzahl Simulationsschritte")
    p.add_argument("--envs", type=int, default=None, help="parallele Arenen (bench)")
    p.add_argument("--agents", type=int, default=None, help="Agenten pro Arena")
    p.add_argument("--cpu", action="store_true", help="auf CPU statt CUDA rechnen")
    p.add_argument("--out", type=str, default=None, help="Replay-Ausgabepfad (record)")
    p.add_argument("--log-every", type=int, default=5, help="Log-Intervall in Schritten (mesh)")
    args = p.parse_args()

    cfg = SimConfig()
    if args.cpu:
        cfg.device = "cpu"
    if args.agents is not None:
        cfg.num_agents = args.agents

    if args.mode == "live":
        run_live(cfg, steps=args.steps or 3000)
    elif args.mode == "record":
        out = Path(args.out) if args.out else Path("replays") / f"sandbox_{int(time.time())}.npz"
        run_record(cfg, steps=args.steps or 1200, out=out)
    elif args.mode == "mesh":
        if args.envs is not None:
            cfg.num_envs = args.envs
        run_mesh(cfg, steps=args.steps or 1000, log_every=args.log_every)
    else:  # bench
        cfg.num_envs = args.envs if args.envs is not None else 4096
        run_bench(cfg, steps=args.steps or 300)


if __name__ == "__main__":
    main()
