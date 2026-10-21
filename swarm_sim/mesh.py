"""Mesh-Netz: Funkradius, dynamische Gruppenbildung, Anfuehrer-Wahl.

Jeder Agent hat einen Funkradius. Zwei Agenten desselben Teams sind direkt
verbunden, wenn ihr Abstand <= radio_radius ist. Ueber diese direkten
Verbindungen bilden sich Gruppen (Zusammenhangskomponenten). Teilt sich ein
Schwarm auf, entstehen mehrere Gruppen; treffen sie sich wieder, verschmelzen
die Gruppen - das ergibt sich automatisch aus den Positionen.

Alles vollvektorisiert ueber alle Environments:
    pos, team -> Verbindungen (E,A,A) -> Gruppen-Label (E,A) -> Anfuehrer (E,A)

Die Gruppen-Label werden per Label-Propagation berechnet: jeder Knoten
uebernimmt wiederholt das kleinste Label in seiner Nachbarschaft, bis sich
nichts mehr aendert. Das Label einer Gruppe ist damit der kleinste
Agenten-Index in ihr.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import SimConfig


@dataclass
class MeshState:
    adjacency: torch.Tensor   # (E, A, A) bool  direkte Funkverbindungen (ohne Selbst)
    labels: torch.Tensor      # (E, A) int64    Gruppen-Label (= kleinster Index der Gruppe)
    leader: torch.Tensor      # (E, A) int64    Anfuehrer-Index der eigenen Gruppe
    degree: torch.Tensor      # (E, A) int64    Anzahl direkter Verbindungen


class MeshNetwork:
    def __init__(self, cfg: SimConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.radio_radius = getattr(cfg, "radio_radius", 18.0)
        self.same_team_only = getattr(cfg, "mesh_same_team", True)

    # ------------------------------------------------------------------ #
    def compute(self, pos: torch.Tensor, team: torch.Tensor) -> MeshState:
        """Berechnet das komplette Mesh fuer den aktuellen Zustand.

        pos  : (E, A, 2)   team : (E, A)
        """
        E, A, _ = pos.shape
        dist = torch.cdist(pos, pos)                       # (E, A, A)
        eye = torch.eye(A, device=self.device, dtype=torch.bool)

        adj = (dist <= self.radio_radius) & ~eye
        if self.same_team_only:
            adj = adj & (team.unsqueeze(1) == team.unsqueeze(2))

        labels = self._connected_components(adj)
        degree = adj.sum(dim=2).to(torch.int64)
        leader = self._elect_leaders(labels, degree)
        return MeshState(adjacency=adj, labels=labels, leader=leader, degree=degree)

    # ------------------------------------------------------------------ #
    def _connected_components(self, adj: torch.Tensor) -> torch.Tensor:
        """Label-Propagation: Label = kleinster Index der Zusammenhangskomponente.

        adj: (E, A, A) bool. Selbstschleife wird hier hinzugefuegt, damit ein
        isolierter Agent sein eigenes Label behaelt. Konvergiert in <= A
        Iterationen; wir brechen frueh ab, wenn sich nichts mehr aendert.
        """
        E, A, _ = adj.shape
        eye = torch.eye(A, device=self.device, dtype=torch.bool)
        adj_self = adj | eye                               # (E, A, A)

        labels = torch.arange(A, device=self.device).expand(E, A).clone()
        big = A + 1
        for _ in range(A):
            # fuer jeden Knoten i: kleinstes Label unter seinen Nachbarn j
            cand = torch.where(adj_self, labels.unsqueeze(1), torch.full_like(labels.unsqueeze(1), big))
            new_labels = cand.min(dim=2).values
            if torch.equal(new_labels, labels):
                break
            labels = new_labels
        return labels

    # ------------------------------------------------------------------ #
    def _elect_leaders(self, labels: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
        """Pro Gruppe gewinnt der Agent mit den meisten Verbindungen.

        Gleichstand wird durch den kleinsten Index entschieden. Rueckgabe:
        fuer jeden Agenten der Index seines Gruppen-Anfuehrers.
        """
        E, A = labels.shape
        idx = torch.arange(A, device=self.device).expand(E, A)
        # Score so, dass hoher Grad gewinnt, bei Gleichstand kleiner Index:
        score = degree.to(torch.float32) - idx.to(torch.float32) / (A + 1.0)

        same_group = labels.unsqueeze(1) == labels.unsqueeze(2)   # (E, A, A)
        neg_inf = torch.finfo(torch.float32).min
        masked = torch.where(same_group, score.unsqueeze(1), torch.full_like(score.unsqueeze(1), neg_inf))
        leader = masked.argmax(dim=2).to(torch.int64)             # (E, A)
        return leader

    # ------------------------------------------------------------------ #
    @staticmethod
    def group_stats(state: MeshState) -> dict:
        """Aggregierte Kennzahlen fuer das Dashboard (Mittel ueber Environments).

        Gruppenanzahl, mittlere Gruppengroesse, Mesh-Verbindungen (Kanten),
        isolierte Agenten.
        """
        labels = state.labels                      # (E, A)
        E, A = labels.shape
        idx = torch.arange(A, device=labels.device).expand(E, A)

        is_root = labels == idx                    # Gruppenrepraesentant je Gruppe
        num_groups = is_root.sum(dim=1).float()    # (E,) Gruppen pro Arena
        edges = state.adjacency.sum(dim=(1, 2)).float() / 2.0   # ungerichtete Kanten
        isolated = (state.degree == 0).sum(dim=1).float()
        avg_group_size = A / num_groups.clamp(min=1.0)

        return {
            "mesh/num_groups": num_groups.mean().item(),
            "mesh/avg_group_size": avg_group_size.mean().item(),
            "mesh/edges": edges.mean().item(),
            "mesh/isolated_agents": isolated.mean().item(),
        }
