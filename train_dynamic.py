"""Phase 4 - MAPPO-Training mit Dynamik (Nachschub, Ausfaelle, Attention).

  python train_dynamic.py                    # Standardlauf auf der GPU
  python train_dynamic.py --updates 500 --envs 512
  python train_dynamic.py --cpu              # Debuggen auf der CPU

Nutzt das Attention-Netz (variable Teamgroessen) und das dynamische Env.
Loggt zusaetzlich zu Phase 3 die Schwarm-Dynamik ins lokale TensorBoard:
  swarm/destroyed_per_step, swarm/alive_frac, swarm/spawned
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from swarm_sim import SimConfig, TrainConfig, MetricsLogger
from swarm_sim.env_dynamic import ZoneControlDynEnv, NUM_ACTIONS
from swarm_sim.attn import AttnActorCritic
from swarm_sim.mappo import compute_gae


def train(cfg: SimConfig, tcfg: TrainConfig, run_name: str | None = None) -> None:
    env = ZoneControlDynEnv(cfg)
    dev = env.device
    net = AttnActorCritic(env.entity_dim, env.self_global_dim, env.team_global_dim,
                          NUM_ACTIONS, hidden=tcfg.hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=tcfg.lr)
    team_row = env.agent_team_index().to(dev)                 # (A,)

    E, A, T, R = cfg.num_envs, cfg.num_agents, cfg.num_teams, tcfg.rollout_len
    ckpt_dir = Path(tcfg.ckpt_dir)

    obs = env.reset()
    recent_win = 0.5

    def policy_value(o):
        dist = net.policy(o["entities"], o["alive"], o["team"], o["self_global"])
        val = net.critic_value(o["entities"], o["alive"], o["team"], o["team_global"])
        return dist, val

    with MetricsLogger(run_name=run_name or f"mappo_dyn_{int(time.time())}") as log:
        print(f"Dynamik-Training auf {dev}: {E} Arenen, bis {A} Agenten ({A//T}/Team), {T} Teams.")
        print(f"TensorBoard: tensorboard --logdir runs  ->  http://localhost:6006")
        global_step = 0
        t_start = time.perf_counter()

        for update in range(1, tcfg.total_updates + 1):
            ent_buf = torch.empty(R, E, A, env.entity_dim, device=dev)
            alive_buf = torch.empty(R, E, A, dtype=torch.bool, device=dev)
            sg_buf = torch.empty(R, E, A, env.self_global_dim, device=dev)
            tg_buf = torch.empty(R, E, T, env.team_global_dim, device=dev)
            act_buf = torch.empty(R, E, A, dtype=torch.long, device=dev)
            logp_buf = torch.empty(R, E, A, device=dev)
            val_buf = torch.empty(R, E, T, device=dev)
            rew_buf = torch.empty(R, E, T, device=dev)
            acc = {"in_zone_frac": 0.0, "destroyed": 0.0, "alive_frac": 0.0, "spawned": 0.0}

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
                acc["in_zone_frac"] += info["in_zone_frac"]
                acc["destroyed"] += info["destroyed_per_step"]
                acc["alive_frac"] += info["alive_frac"]
                acc["spawned"] += info["spawned_per_step"]
                if "winrate_team0" in info:
                    recent_win = info["winrate_team0"]
                global_step += E

            with torch.no_grad():
                _, last_val = policy_value(obs)
            adv, returns = compute_gae(rew_buf, val_buf, last_val, tcfg.gamma, tcfg.gae_lambda)
            adv_agent = adv[:, :, team_row]                       # (R,E,A)

            # auf Env-Schritte (M = R*E) zusammenfassen; Mengen bleiben erhalten
            M = R * E
            ent = ent_buf.reshape(M, A, env.entity_dim)
            alive = alive_buf.reshape(M, A)
            sg = sg_buf.reshape(M, A, env.self_global_dim)
            tg = tg_buf.reshape(M, T, env.team_global_dim)
            acts = act_buf.reshape(M, A)
            old_logp = logp_buf.reshape(M, A)
            old_val = val_buf.reshape(M, T)
            rets = returns.reshape(M, T)
            adv_a = adv_agent.reshape(M, A)
            team_b = team_row.unsqueeze(0).expand(M, A)

            alive_f = alive.float()
            mean = (adv_a * alive_f).sum() / alive_f.sum().clamp(min=1)
            var = ((adv_a - mean) ** 2 * alive_f).sum() / alive_f.sum().clamp(min=1)
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

                    v_clip = old_val[sl] + (val - old_val[sl]).clamp(-tcfg.clip, tcfg.clip)
                    v_loss = torch.max((val - rets[sl]) ** 2, (v_clip - rets[sl]) ** 2).mean()

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
            stats["progress/in_zone_frac"] = acc["in_zone_frac"] / R
            stats["progress/winrate_team0"] = recent_win
            stats["swarm/destroyed_per_step"] = acc["destroyed"] / R
            stats["swarm/alive_frac"] = acc["alive_frac"] / R
            stats["swarm/spawned_per_rollout"] = acc["spawned"]
            log.log(stats, global_step)
            log.log_throughput(global_step, E)
            log.log_gpu(global_step)

            if update % 25 == 0 or update == 1:
                sps = global_step / (time.perf_counter() - t_start)
                print(f"Update {update:4d} | in_zone {stats['progress/in_zone_frac']:.3f} "
                      f"| alive {stats['swarm/alive_frac']:.2f} | destr/step {stats['swarm/destroyed_per_step']:.2f} "
                      f"| entropy {stats['loss/entropy']:.2f} | {sps:,.0f} steps/s")

            if update % tcfg.ckpt_every == 0:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), ckpt_dir / f"dyn_update{update}.pt")

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        final = ckpt_dir / "dyn_final.pt"
        torch.save(net.state_dict(), final)
        print(f"Fertig. Finaler Checkpoint: {final}  |  Logs: {log.path}")


def main() -> None:
    p = argparse.ArgumentParser(description="MAPPO Dynamik (Phase 4)")
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
