# FARP-Seg segmentation kernels.

from __future__ import annotations
import heapq
from collections import defaultdict
from typing import Any
import numpy as np

class PFComponentGraph:
    """
    Replicates the fixed topology-fallback rule of Exp6-2-2.
    """

    def __init__(
        self,
        surviving_tracks: set[int],
        protected_tracks: set[int],
        retained_tracks: set[int],
        ff_contact: dict[tuple[int, int], int],
        fa_z_contact: dict[tuple[int, int], int],
    ):
        self.members: dict[int, set[int]] = {int(t): {int(t)} for t in sorted(surviving_tracks)}
        self.track_to_root: dict[int, int] = {int(t): int(t) for t in surviving_tracks}
        self.ff: dict[int, dict[int, int]] = {int(t): {} for t in surviving_tracks}
        self.fa: dict[int, dict[int, int]] = {int(t): {} for t in surviving_tracks}

        for (a, b), count in ff_contact.items():
            a = int(a)
            b = int(b)
            if a not in surviving_tracks or b not in surviving_tracks or a == b:
                continue
            self.ff[a][b] = int(self.ff[a].get(b, 0) + int(count))
            self.ff[b][a] = int(self.ff[b].get(a, 0) + int(count))

        for (f_id, a_id), count in fa_z_contact.items():
            f_id = int(f_id)
            if f_id not in surviving_tracks:
                continue
            self.fa[f_id][int(a_id)] = int(self.fa[f_id].get(int(a_id), 0) + int(count))

        self.terminal: dict[int, tuple[str, int] | None] = {}
        for t in surviving_tracks:
            if int(t) in retained_tracks:
                self.terminal[int(t)] = ("F", int(t))
            elif int(t) in protected_tracks:
                self.terminal[int(t)] = None
            else:
                raise RuntimeError(f"Unexpected PF surviving F track {t}")

    def _best_action_for_root(self, root: int) -> tuple[int, int, str, int] | None:
        if self.terminal[root] is not None:
            return None

        options: list[tuple[int, int, str, int]] = []
        for other, contact in self.ff[root].items():
            if other == root or other not in self.members or contact <= 0:
                continue
            options.append((int(contact), 1, "F", int(other)))
        for a_label, contact in self.fa[root].items():
            if contact <= 0:
                continue
            options.append((int(contact), 0, "A", int(a_label)))

        if not options:
            return None
        options.sort(key=lambda x: (-x[0], -x[1], x[3]))
        return options[0]

    def _merge_root_into_f(self, source: int, target: int) -> None:
        source_members = set(self.members[source])
        source_ff = dict(self.ff[source])

        for neighbor, weight in source_ff.items():
            if neighbor == target or neighbor not in self.members:
                continue
            self.ff[neighbor].pop(source, None)
            self.ff[neighbor][target] = int(self.ff[neighbor].get(target, 0) + int(weight))
            self.ff[target][neighbor] = int(self.ff[target].get(neighbor, 0) + int(weight))
        self.ff[target].pop(source, None)
        self.ff[target].pop(target, None)

        for a_label, weight in self.fa[source].items():
            self.fa[target][a_label] = int(self.fa[target].get(a_label, 0) + int(weight))

        target_terminal = self.terminal[target]
        source_terminal = self.terminal[source]
        if target_terminal is None and source_terminal is not None:
            self.terminal[target] = source_terminal

        self.members[target].update(source_members)
        for track_id in source_members:
            self.track_to_root[track_id] = target

        del self.members[source]
        del self.ff[source]
        del self.fa[source]
        del self.terminal[source]

    def _resolve_root_to_a(self, root: int, a_label: int) -> None:
        self.terminal[root] = ("A", int(a_label))

    def run(self) -> None:
        while True:
            proposals: list[tuple[int, int, int, str, int]] = []
            for root in sorted(self.members):
                best = self._best_action_for_root(root)
                if best is None:
                    continue
                contact, type_priority, target_type, target_id = best
                proposals.append((-int(contact), int(root), -int(type_priority), target_type, int(target_id)))
            if not proposals:
                break
            proposals.sort()
            neg_contact, source, _neg_priority, target_type, target_id = proposals[0]
            if source not in self.members or self.terminal[source] is not None:
                continue
            if target_type == "F":
                if target_id not in self.members:
                    continue
                self._merge_root_into_f(source, target_id)
            else:
                self._resolve_root_to_a(source, target_id)
