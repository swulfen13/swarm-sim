"""Attention-basiertes Actor-Critic fuer wechselnde Teamgroessen (Phase 4).

Statt einer festen Verkettung aller Agenten (Phase 3) nutzt dieses Netz
maskierte Attention ueber die Menge der *lebenden* Agenten. Dadurch ist es
permutationsinvariant und kommt mit beliebig vielen aktiven Agenten klar -
genau was Nachschub und Ausfaelle erfordern.

Der Actor bildet pro Agent zwei maskierte Attention-Pools: einen ueber die
lebenden Verbuendeten, einen ueber die lebenden Gegner. Der Critic poolt pro
Team ueber lebende Verbuendete und Gegner (maskierter Mittelwert).

Eingaben (vom dynamischen Env erzeugt):
  entities    (E, A, De)   pro Agent: [pos_n(2), vel_n(2), in_zone(1)]
  alive       (E, A) bool
  team        (E, A) long
  self_global (E, A, Dg)   pro Agent: [vec_zu_zone(2), dist_zone(1)]
  team_global (E, T, Dtg)  pro Team: [eigene_frac, gegner_frac]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _masked_attention(query, keys, values, mask):
    """Single-Head Attention mit Maskierung ungueltiger Keys.

    query (E,A,H), keys/values (E,K,H), mask (E,A,K) bool.
    Zeilen ohne gueltigen Key liefern einen Null-Kontext.
    """
    H = query.shape[-1]
    scores = torch.einsum("eah,ekh->eak", query, keys) / math.sqrt(H)  # (E,A,K)
    neg = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~mask, neg)
    valid = mask.any(dim=-1, keepdim=True)                              # (E,A,1)
    attn = torch.softmax(scores, dim=-1)
    attn = torch.where(valid, attn, torch.zeros_like(attn))
    return torch.einsum("eak,ekh->eah", attn, values)                  # (E,A,H)


class AttnActorCritic(nn.Module):
    def __init__(self, entity_dim: int, self_global_dim: int, team_global_dim: int,
                 num_actions: int, hidden: int = 64):
        super().__init__()
        H = hidden
        self.embed = nn.Sequential(nn.Linear(entity_dim, H), nn.Tanh(), nn.Linear(H, H), nn.Tanh())
        self.q = nn.Linear(H, H)
        self.k = nn.Linear(H, H)
        self.v = nn.Linear(H, H)
        # Actor-Kopf: eigenes Embedding + Ally-Pool + Enemy-Pool + Self-Global
        self.actor_head = nn.Sequential(
            nn.Linear(H * 3 + self_global_dim, H), nn.Tanh(),
            nn.Linear(H, num_actions),
        )
        # Critic-Kopf: Ally-Pool + Enemy-Pool + Team-Global
        self.critic_head = nn.Sequential(
            nn.Linear(H * 2 + team_global_dim, H), nn.Tanh(),
            nn.Linear(H, 1),
        )

    # ------------------------------------------------------------------ #
    def _masks(self, alive, team):
        same = team.unsqueeze(2) == team.unsqueeze(1)        # (E,A,A) [i,k]
        alive_k = alive.unsqueeze(1)                          # (E,1,A)
        ally = same & alive_k                                 # inkl. self -> >=1 Key
        enemy = (~same) & alive_k
        return ally, enemy

    # ------------------------------------------------------------------ #
    def actor_logits(self, entities, alive, team, self_global):
        emb = self.embed(entities)                            # (E,A,H)
        q, k, v = self.q(emb), self.k(emb), self.v(emb)
        ally, enemy = self._masks(alive, team)
        pool_a = _masked_attention(q, k, v, ally)
        pool_e = _masked_attention(q, k, v, enemy)
        x = torch.cat([emb, pool_a, pool_e, self_global], dim=-1)
        return self.actor_head(x)                             # (E,A,num_actions)

    # ------------------------------------------------------------------ #
    def critic_value(self, entities, alive, team, team_global):
        """Ein Wert je Team aus maskiertem Mittelwert ueber lebende Agenten."""
        E, A, _ = entities.shape
        T = team_global.shape[1]
        emb = self.embed(entities)                            # (E,A,H)

        values = []
        for t in range(T):
            ally_m = ((team == t) & alive).unsqueeze(-1).float()   # (E,A,1)
            enemy_m = ((team != t) & alive).unsqueeze(-1).float()
            pool_a = (emb * ally_m).sum(1) / ally_m.sum(1).clamp(min=1.0)
            pool_e = (emb * enemy_m).sum(1) / enemy_m.sum(1).clamp(min=1.0)
            x = torch.cat([pool_a, pool_e, team_global[:, t]], dim=-1)
            values.append(self.critic_head(x).squeeze(-1))         # (E,)
        return torch.stack(values, dim=1)                          # (E,T)

    # ------------------------------------------------------------------ #
    def agent_values(self, entities, alive, team, team_global):
        """Ein Wert PRO AGENT (E,A) statt pro Team - fuer lokale Kreditzuweisung.

        Jeder Agent: eigenes Embedding + zentralisierter Pool ueber alle lebenden
        Agenten (CTDE) + globale Team-Features. Nutzt den vorhandenen critic_head
        wieder (gleiche Eingangsdimension H*2 + team_global_dim), also keine neuen
        Parameter -> andere Modelle (relay/search) bleiben ladbar.
        """
        emb = self.embed(entities)                             # (E,A,H)
        am = alive.unsqueeze(-1).float()                       # (E,A,1)
        pool = (emb * am).sum(1, keepdim=True) / am.sum(1, keepdim=True).clamp(min=1.0)  # (E,1,H)
        pool = pool.expand(-1, emb.shape[1], -1)               # (E,A,H)
        g = team_global[:, 0:1, :].expand(-1, emb.shape[1], -1)  # (E,A,Dtg)
        x = torch.cat([emb, pool, g], dim=-1)                  # (E,A, 2H+Dtg)
        return self.critic_head(x).squeeze(-1)                 # (E,A)

    # ------------------------------------------------------------------ #
    def policy(self, entities, alive, team, self_global):
        logits = self.actor_logits(entities, alive, team, self_global)
        return torch.distributions.Categorical(logits=logits)
