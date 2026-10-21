"""MAPPO: geteilter Actor (dezentral) + zentraler Critic, mit PPO-Update.

Ein einziges Netzwerk spielt beide Teams (symmetrisches Self-Play), weil die
Beobachtungen team-relativ sind. Der Actor sieht nur die Agenten-Beobachtung,
der Critic die zentrale Team-Sicht und schaetzt einen Wert pro Team.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import TrainConfig


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, state_dim: int, num_actions: int, hidden: int):
        super().__init__()
        self.actor = _mlp(obs_dim, hidden, num_actions)
        self.critic = _mlp(state_dim, hidden, 1)

    def policy(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.actor(obs)
        return torch.distributions.Categorical(logits=logits)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)


class MAPPO:
    def __init__(self, obs_dim: int, state_dim: int, num_actions: int,
                 tcfg: TrainConfig, device: torch.device):
        self.net = ActorCritic(obs_dim, state_dim, num_actions, tcfg.hidden).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=tcfg.lr)
        self.tcfg = tcfg
        self.device = device

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        """obs (..., obs_dim) -> action, logprob, entropy (gesamplet)."""
        dist = self.net.policy(obs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    @torch.no_grad()
    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.net.value(state)

    # ------------------------------------------------------------------ #
    def update(self, batch: dict) -> dict:
        """PPO-Update. batch enthaelt flache Tensoren (N, ...):

        obs, actions, old_logp, advantages (pro Agent); state, returns,
        old_values (pro Team-Sample).
        """
        t = self.tcfg
        obs = batch["obs"]
        actions = batch["actions"]
        old_logp = batch["old_logp"]
        adv = batch["advantages"]            # pro Agent (schon vom Team gemappt)
        state = batch["state"]               # pro Team-Sample
        returns = batch["returns"]           # pro Team-Sample
        old_values = batch["old_values"]     # pro Team-Sample

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n_agents = obs.shape[0]
        n_team = state.shape[0]
        mb_a = max(1, n_agents // t.minibatches)
        mb_s = max(1, n_team // t.minibatches)

        stats = {"loss/policy": 0.0, "loss/value": 0.0, "loss/entropy": 0.0, "ppo/approx_kl": 0.0}
        count = 0
        for _ in range(t.epochs):
            idx_a = torch.randperm(n_agents, device=self.device)
            idx_s = torch.randperm(n_team, device=self.device)
            for b in range(t.minibatches):
                a_sl = idx_a[b * mb_a:(b + 1) * mb_a]
                s_sl = idx_s[b * mb_s:(b + 1) * mb_s]
                if len(a_sl) == 0 or len(s_sl) == 0:
                    continue

                # --- Actor ---
                dist = self.net.policy(obs[a_sl])
                logp = dist.log_prob(actions[a_sl])
                ratio = torch.exp(logp - old_logp[a_sl])
                a = adv[a_sl]
                surr1 = ratio * a
                surr2 = torch.clamp(ratio, 1 - t.clip, 1 + t.clip) * a
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy = dist.entropy().mean()

                # --- Critic (clipped value loss) ---
                v = self.net.value(state[s_sl])
                v_clipped = old_values[s_sl] + (v - old_values[s_sl]).clamp(-t.clip, t.clip)
                v_loss = torch.max((v - returns[s_sl]) ** 2, (v_clipped - returns[s_sl]) ** 2).mean()

                loss = policy_loss - t.entropy_coef * entropy + t.value_coef * v_loss
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), t.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    stats["loss/policy"] += policy_loss.item()
                    stats["loss/value"] += v_loss.item()
                    stats["loss/entropy"] += entropy.item()
                    stats["ppo/approx_kl"] += (old_logp[a_sl] - logp).mean().item()
                count += 1

        for key in stats:
            stats[key] /= max(count, 1)
        return stats

    # ------------------------------------------------------------------ #
    def save(self, path) -> None:
        torch.save({"net": self.net.state_dict(), "opt": self.opt.state_dict()}, path)

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.opt.load_state_dict(ckpt["opt"])


def compute_gae(rewards, values, last_value, gamma, lam):
    """GAE pro Team ueber die Zeit.

    rewards   (Tsteps, E, T)
    values    (Tsteps, E, T)
    last_value(E, T)          Bootstrap-Wert nach dem letzten Schritt
    return    advantages, returns  je (Tsteps, E, T)
    """
    Tsteps = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    for step in reversed(range(Tsteps)):
        next_v = last_value if step == Tsteps - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_v - values[step]
        gae = delta + gamma * lam * gae
        adv[step] = gae
    returns = adv + values
    return adv, returns
