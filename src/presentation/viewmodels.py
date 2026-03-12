"""
src/presentation/viewmodels.py

Immutable DTOs that the presentation layer consumes.
These are the only data types that cross from domain to presentation;
the view never imports Person, ExerciseState, or any domain type directly.

Frozen dataclasses enforce immutability so the Presenter builds a fresh
snapshot each frame and widgets simply replace their state.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ── Fine-grained ViewModels ───────────────────────────────────────────────────


@dataclass(frozen=True)
class ExerciseStatsVM:
    """State of ONE exercise for ONE person — presentation-only."""

    name: str
    rep_count: int
    phase: str                                    # "up" | "down"
    feedback: str
    angles: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PersonStatsVM:
    """All exercise states for ONE tracked person."""

    track_id: int
    exercises: dict[str, ExerciseStatsVM] = field(default_factory=dict)


# ── Top-level ViewModels ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class DashboardVM:
    """
    Complete snapshot of the left stats panel.
    Built by DashboardPresenter from IExerciseCounter + Person[].
    """

    num_tracked: int
    exercise_totals: dict[str, int]                # {exercise_name: total_reps_all_ids}
    exercise_by_id: dict[str, dict[int, int]]      # {exercise_name: {track_id: reps}}
    persons: list[PersonStatsVM] = field(default_factory=list)


@dataclass
class VideoFrameVM:
    """
    Annotated video frame ready to be displayed.
    NOT frozen: numpy arrays are mutable by nature.
    """

    frame: np.ndarray    # BGR, uint8 — already annotated with skeleton/overlays
    width: int
    height: int
