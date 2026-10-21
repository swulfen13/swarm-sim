"""Phase 3 - MAPPO-Training fuer Zonenkontrolle (Self-Play).

  python train.py                         # Standardlauf auf der GPU
  python train.py --updates 500 --envs 512
  python train.py --cpu                   # zum Debuggen auf der CPU

Waehrend des Trainings laufen alle Kennzahlen live ins lokale TensorBoard:
  tensorboard --logdir runs   ->  http://localhost:6006

Geloggt:
  loss/*      Policy-, Value-Loss, Entropie
  ppo/*       Approx-KL
  progress/*  Anteil Agenten in der Zone, Siegquote, Zonen-Balance
  system/*    Steps/s, GPU-Auslastung, VRAM, Temperatur
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from swarm_sim import SimConfig, TrainConfig, ZoneControlEnv, NUM_ACTIONS, MAPPO, MetricsLogger
from swarm_sim.mappo import compute_gae


def train(cfg: SimConfig, tcfg: TrainConfig, run_name: str | None = None) -> None:
    env = ZoneControlEnv(cfg)
    dev = env.device
    algo = MAPPO(env.obs_dim, env.state_dim, NUM_ACTIONS, tcfg, dev)
    team_of_agent = env.agent_team_index().to(dev)   # (A,)

    E, A, T, R = cfg.num_envs, cfg.num_agents, cfg.num_teams, tcfg.rollout_len
    ckpt_dir = Path(tcfg.ckpt_dir)

    obs, state = env.reset()
    recent = {"winrate_team0": 0.5, "zone_balance": 0.0}

    with MetricsLogger(run_name=run_name or f"mappo_{int(time.time())}") as log:
        print(f"Training auf {dev}: {E} Arenen, {A} Agenten ({A//T} je Team), {T} Teams.")
        print(f"TensorBoard: tensorboard --logdir runs  ->  http://localhost:6006")
        global_step = 0
        t_start = time.perf_counter()

        for update in range(1, tcfg.total_updates + 1):
            obs_buf = torch.empty(R, E, A, env.obs_dim, device=dev)
            act_buf = torch.empty(R, E, A, dtype=torch.long, device=dev)
            logp_buf = torch.empty(R, E, A, device=dev)
            state_buf = torch.empty(R, E, T, env.state_dim, device=dev)
            val_buf = torch.empty(R, E, T, device=dev)
            rew_buf = torch.empty(R, E, T, device=dev)
            in_zone_acc = 0.0

            # --- Rollout sammeln ---
            for r in range(R):
                action, logp, _ = algo.act(obs)               # (E,A)
                value = algo.value(state)                      # (E,T)

                obs_buf[r] = obs
                act_buf[r] = action
                logp_buf[r] = logp
                state_buf[r] = state
                val_buf[r] = value

                obs, state, reward, done, info = env.step(action)
                rew_buf[r] = reward
                in_zone_acc += info["in_zone_frac"]
                if "winrate_team0" in info:
                    recent["winrate_team0"] = info["winrate_team0"]
                    recent["zone_balance"] = info["zone_balance"]
                global_step += E

            # --- Vorteile (GAE) pro Team, dann auf Agenten mappen ---
            last_value = algo.value(state)                     # (E,T)
            adv, returns = compute_gae(rew_buf, val_buf, last_value, tcfg.gamma, tcfg.gae_lambda)
            adv_agent = adv[:, :, team_of_agent]               # (R,E,A)

            batch = {
                "obs": obs_buf.reshape(-1, env.obs_dim),
                "actions": act_buf.reshape(-1),
                "old_logp": logp_buf.reshape(-1),
                "advantages": adv_agent.reshape(-1),
                "state": state_buf.reshape(-1, env.state_dim),
                "returns": returns.reshape(-1),
                "old_values": val_buf.reshape(-1),
            }
            stats = algo.update(batch)

            # --- Logging ---
            if update % tcfg.log_every == 0:
                stats["progress/in_zone_frac"] = in_zone_acc / R
                stats["progress/winrate_team0"] = recent["winrate_team0"]
                stats["progress/zone_balance"] = recent["zone_balance"]
                stats["progress/reward_abs"] = rew_buf.abs().mean().item()
                log.log(stats, global_step)
                log.log_throughput(global_step, E)
                log.log_gpu(global_step)

            if update % 25 == 0 or update == 1:
                sps = global_step / (time.perf_counter() - t_start)
                print(f"Update {update:4d} | in_zone {stats.get('progress/in_zone_frac', 0):.3f} "
                      f"| policy {stats['loss/policy']:+.3f} | value {stats['loss/value']:.3f} "
                      f"| entropy {stats['loss/entropy']:.3f} | {sps:,.0f} env-steps/s")

            if update % tcfg.ckpt_every == 0:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                p = ckpt_dir / f"mappo_update{update}.pt"
                algo.save(p)
                print(f"  Checkpoint: {p}")

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        final = ckpt_dir / "mappo_final.pt"
        algo.save(final)
        print(f"Fertig. Finaler Checkpoint: {final}  |  Logs: {log.path}")


def main() -> None:
    p = argparse.ArgumentParser(description="MAPPO Zonenkontrolle (Phase 3)")
    p.add_argument("--updates", type=int, default=None)
    p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--agents", type=int, default=10)
    p.add_argument("--rollout", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--name", type=str, default=None)
    args = p.parse_args()

    cfg = SimConfig(num_envs=args.envs, num_agents=args.agents, num_teams=2)
    if args.cpu:
        cfg.device = "cpu"
    tcfg = TrainConfig()
    if args.updates is not None:
        tcfg.total_updates = args.updates
    if args.rollout is not None:
        tcfg.rollout_len = args.rollout

    train(cfg, tcfg, run_name=args.name)


if __name__ == "__main__":
    main()
