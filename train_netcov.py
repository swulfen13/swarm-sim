"""Phase 7 - MAPPO-Training des Netzabdeckungs-Vollmodells (YAML-getrieben).

Liest netzabdeckung_config.yaml, trainiert den Schwarm ueber das 4-stufige
Curriculum (Aufstieg, sobald die gewichtete Abdeckung laenger ueber 90 %
liegt) und loggt die Auswertungs-Kennzahlen lokal nach TensorBoard.

  python train_netcov.py                       # Standardlauf auf der GPU
  python train_netcov.py --updates 4000 --envs 256
  python train_netcov.py --config netzabdeckung_config.yaml --start-stage 1

Dashboard (tensorboard --logdir runs  ->  http://localhost:6006):
  netcov/served_pct           gewichtete Abdeckung %
  netcov/connected_frac       Anteil mit Uplink verbundener Drohnen
  netcov/holes_frac           Anteil Hotspots ohne Versorgung (Funkloch)
  netcov/crashes_per_step     Abstuerze (leerer Akku / Ausfall)
  netcov/collisions_per_step  Kollisionen mit Gelaende
  netcov/charging_now         gleichzeitig ladende Drohnen
  netcov/mean_battery_pct     mittlerer Akkustand
  netcov/wind_ms              aktueller Wind
  curriculum/stage            aktuelle Stufe (1..4)
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn

from swarm_sim import TrainConfig, MetricsLogger
from swarm_sim.mission_config import MissionConfig
from swarm_sim.env_netcov import NetCovEnv, NUM_ACTIONS
from swarm_sim.attn import AttnActorCritic
from swarm_sim.mappo import compute_gae


def train(cfg_path: str, num_envs: int, updates: int, start_stage: int,
          device: str, run_name: str | None, resume: str | None = None) -> None:
    mcfg = MissionConfig.load(cfg_path, stage=start_stage)
    env = NetCovEnv(mcfg, num_envs=num_envs, device=device)
    dev = env.device
    tcfg = TrainConfig()
    tcfg.total_updates = updates
    tcfg.gamma = 0.997        # run10: laengerer Kredit-Horizont (~33s statt ~10s) fuer die Lade-Rotation

    net = AttnActorCritic(env.entity_dim, env.self_global_dim, env.team_global_dim,
                          NUM_ACTIONS, hidden=tcfg.hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=tcfg.lr)
    if resume:
        net.load_state_dict(torch.load(resume, map_location=dev))
        print(f"Resume: Netz-Gewichte geladen aus {resume} (Optimizer startet frisch)")
    team_row = env.agent_team_index().to(dev)

    E, A, T, R = env.E, env.A, env.T, tcfg.rollout_len
    ckpt_dir = Path(tcfg.ckpt_dir)

    obs = env.reset()
    stage = start_stage
    served_hist = deque(maxlen=100)   # Aufstiegskriterium: Abdeckung > 90 %

    def policy_value(o):
        dist = net.policy(o["entities"], o["alive"], o["team"], o["self_global"])
        val = net.agent_values(o["entities"], o["alive"], o["team"], o["team_global"])  # (E,A) pro Agent
        return dist, val

    with MetricsLogger(run_name=run_name or f"netcov_{int(time.time())}") as log:
        print(f"Netzabdeckung-Training auf {dev}: {E} Arenen, bis {A} Drohnen, Start-Stufe {stage}.")
        print(f"Karte {env.map_w:.0f}x{env.map_h:.0f} m, dt={env.dt}s. TensorBoard: tensorboard --logdir runs")
        global_step = 0
        t_start = time.perf_counter()

        for update in range(1, updates + 1):
            ent_buf = torch.empty(R, E, A, env.entity_dim, device=dev)
            alive_buf = torch.empty(R, E, A, dtype=torch.bool, device=dev)
            sg_buf = torch.empty(R, E, A, env.self_global_dim, device=dev)
            tg_buf = torch.empty(R, E, T, env.team_global_dim, device=dev)
            act_buf = torch.empty(R, E, A, dtype=torch.long, device=dev)
            logp_buf = torch.empty(R, E, A, device=dev)
            val_buf = torch.empty(R, E, A, device=dev)    # Wert pro Agent
            rew_buf = torch.empty(R, E, A, device=dev)    # Belohnung pro Agent
            acc = {}

            for r in range(R):
                with torch.no_grad():
                    dist, val = policy_value(obs)
                    action = dist.sample()
                    logp = dist.log_prob(action)
                ent_buf[r] = obs["entities"]; alive_buf[r] = obs["alive"]
                sg_buf[r] = obs["self_global"]; tg_buf[r] = obs["team_global"]
                act_buf[r] = action; logp_buf[r] = logp; val_buf[r] = val

                obs, reward, done, info = env.step(action)
                rew_buf[r] = reward
                for k, vv in info.items():
                    acc[k] = acc.get(k, 0.0) + (vv if isinstance(vv, (int, float)) else 0.0)
                global_step += E

            with torch.no_grad():
                _, last_val = policy_value(obs)
            adv, returns = compute_gae(rew_buf, val_buf, last_val, tcfg.gamma, tcfg.gae_lambda)
            adv_agent = adv                                # Vorteil ist bereits pro Agent (R,E,A)

            M = R * E
            ent = ent_buf.reshape(M, A, env.entity_dim)
            alive = alive_buf.reshape(M, A)
            sg = sg_buf.reshape(M, A, env.self_global_dim)
            tg = tg_buf.reshape(M, T, env.team_global_dim)
            acts = act_buf.reshape(M, A)
            old_logp = logp_buf.reshape(M, A)
            old_val = val_buf.reshape(M, A)
            rets = returns.reshape(M, A)
            adv_a = adv_agent.reshape(M, A)
            team_b = team_row.unsqueeze(0).expand(M, A)

            af = alive.float()
            mean = (adv_a * af).sum() / af.sum().clamp(min=1)
            var = ((adv_a - mean) ** 2 * af).sum() / af.sum().clamp(min=1)
            adv_a = (adv_a - mean) / (var.sqrt() + 1e-8)

            stats = {"loss/policy": 0.0, "loss/value": 0.0, "loss/entropy": 0.0, "ppo/approx_kl": 0.0}
            count = 0
            mb = max(1, M // tcfg.minibatches)
            for _ in range(tcfg.epochs):
                perm = torch.randperm(M, device=dev)
                for b in range(tcfg.minibatches):
                    sl = perm[b * mb:(b + 1) * mb]
                    if sl.numel() == 0:
                        continue
                    o = {"entities": ent[sl], "alive": alive[sl], "team": team_b[sl],
                         "self_global": sg[sl], "team_global": tg[sl]}
                    dist, val = policy_value(o)
                    logp = dist.log_prob(acts[sl])
                    am = alive[sl].float()
                    denom = am.sum().clamp(min=1)
                    ratio = torch.exp(logp - old_logp[sl])
                    a = adv_a[sl]
                    surr1 = ratio * a
                    surr2 = torch.clamp(ratio, 1 - tcfg.clip, 1 + tcfg.clip) * a
                    policy_loss = -(torch.min(surr1, surr2) * am).sum() / denom
                    entropy = (dist.entropy() * am).sum() / denom
                    # Kein Value-Clipping: clip=0.2 ist fuer unnormierte Wert-Ziele (~O(10)) zu eng
                    # und laesst den Critic divergieren. Einfacher maskierter MSE.
                    v_loss = (((val - rets[sl]) ** 2) * am).sum() / denom   # nur lebende Agenten
                    loss = policy_loss - tcfg.entropy_coef * entropy + tcfg.value_coef * v_loss
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), tcfg.max_grad_norm)
                    opt.step()
                    with torch.no_grad():
                        stats["loss/policy"] += policy_loss.item()
                        stats["loss/value"] += v_loss.item()
                        stats["loss/entropy"] += entropy.item()
                        stats["ppo/approx_kl"] += ((old_logp[sl] - logp) * am).sum().item() / denom.item()
                    count += 1
            for key in stats:
                stats[key] /= max(count, 1)

            served = acc.get("served_pct", 0.0) / R
            served_hist.append(served)
            stats["netcov/served_pct"] = served
            stats["netcov/connected_frac"] = acc.get("connected_frac", 0.0) / R
            stats["netcov/holes_frac"] = acc.get("holes_frac", 0.0) / R
            stats["netcov/crashes_per_step"] = acc.get("crashes_per_step", 0.0) / R
            stats["netcov/collisions_per_step"] = acc.get("collisions_per_step", 0.0) / R
            stats["netcov/charging_now"] = acc.get("charging_now", 0.0) / R
            stats["netcov/mean_battery_pct"] = acc.get("mean_battery", 0.0) / R
            stats["netcov/ready_frac"] = acc.get("ready_frac", 0.0) / R
            stats["netcov/wind_ms"] = acc.get("wind_ms", 0.0) / R
            stats["curriculum/stage"] = float(stage)
            log.log(stats, global_step)
            log.log_throughput(global_step, E)
            log.log_gpu(global_step)

            # --- Curriculum-Aufstieg: Abdeckung > 90 % ueber 100 Updates ---
            if stage < env.m.num_stages and len(served_hist) == served_hist.maxlen \
                    and sum(served_hist) / len(served_hist) > 90.0:
                stage += 1
                env.set_stage(stage)
                obs = env.reset()
                served_hist.clear()
                print(f"  >>> Curriculum-Aufstieg zu Stufe {stage}")

            if update % 25 == 0 or update == 1:
                sps = global_step / (time.perf_counter() - t_start)
                print(f"Update {update:4d} | Stufe {stage} | served {served:5.1f}% "
                      f"| conn {stats['netcov/connected_frac']:.2f} | holes {stats['netcov/holes_frac']:.2f} "
                      f"| crash {stats['netcov/crashes_per_step']:.3f} | batt {stats['netcov/mean_battery_pct']:.0f}% "
                      f"| ready {stats['netcov/ready_frac']:.2f} | {sps:,.0f} steps/s")

            if update % tcfg.ckpt_every == 0:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), ckpt_dir / f"netcov_update{update}.pt")

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        final = ckpt_dir / "netcov_final.pt"
        torch.save(net.state_dict(), final)
        print(f"Fertig. Finaler Checkpoint: {final}  |  Logs: {log.path}")


def main() -> None:
    p = argparse.ArgumentParser(description="MAPPO Netzabdeckung Vollmodell (Phase 7)")
    p.add_argument("--config", type=str, default="netzabdeckung_config.yaml")
    p.add_argument("--updates", type=int, default=3000)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--start-stage", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint-Pfad: Gewichte laden und weitertrainieren")
    args = p.parse_args()
    train(args.config, args.envs, args.updates, args.start_stage,
          "cpu" if args.cpu else "cuda", args.name, args.resume)


if __name__ == "__main__":
    main()
