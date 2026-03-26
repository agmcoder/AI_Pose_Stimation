"""
src/exercises/alternating_mixin.py

Mixin for exercises that alternate between left and right sides
(skipping, punch, kick).

Provides per-side state tracking so each subclass only needs to
implement the metric computation and threshold comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SideState:
    """Per-side phase tracking for alternating exercises."""
    phase: str = "idle"       # "idle" | "active"
    valid_active: bool = False


class AlternatingSideMixin:
    """
    Tracks independent left/right sub-states keyed by (track_id, side).

    Subclasses call `get_side_state` / `set_side_state` and implement
    their own metric + transition logic in `update()`.
    """

    def __init__(self) -> None:
        # {track_id: {"left": SideState, "right": SideState}}
        self._side_states: dict[int, dict[str, SideState]] = {}

    def get_side_state(self, track_id: int, side: str) -> SideState:
        per_id = self._side_states.setdefault(track_id, {})
        return per_id.setdefault(side, SideState())

    def set_side_state(self, track_id: int, side: str, state: SideState) -> None:
        self._side_states.setdefault(track_id, {})[side] = state
