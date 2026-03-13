"""
src/presentation/presenter.py

DashboardPresenter — the only component that knows both domain and ViewModels.

Responsibilities:
  - Receive domain objects (Person[], IExerciseCounter, annotated frame)
  - Transform them into ViewModels for the view
  - Know NOTHING about Qt, widgets, or layout

SOLID:
  SRP  — only builds ViewModels, no rendering, no widget management
  DIP  — depends on IExerciseCounter abstraction, not ExerciseCounter directly
  OCP  — adding new ViewModel fields does not require changes to widgets
"""
from __future__ import annotations

import numpy as np

from src.core.interfaces import IExerciseCounter
from src.core.types import Person

from .viewmodels import (
    DashboardVM,
    ExerciseChecklistItemVM,
    ExerciseChecklistVM,
    ExerciseStatsVM,
    PersonStatsVM,
    VideoFrameVM,
)


class DashboardPresenter:
    """
    Bridges the domain pipeline and the presentation layer.

    Usage (each frame):
        video_vm     = presenter.build_video_vm(annotated_frame)
        dashboard_vm = presenter.build_dashboard_vm(persons)
    """

    def __init__(self, counter: IExerciseCounter) -> None:
        self._counter = counter

    # ── Public API ───────────────────────────────────────────────────────────

    def build_video_vm(self, annotated_frame: np.ndarray) -> VideoFrameVM:
        """Wrap the already-annotated frame in a ViewModel."""
        h, w = annotated_frame.shape[:2]
        return VideoFrameVM(frame=annotated_frame, width=w, height=h)

    def build_dashboard_vm(self, persons: list[Person], current_fps: float = 0.0) -> DashboardVM:
        """Build the full dashboard snapshot from counter + persons."""
        all_exercises = self._counter.all_exercises()
        return DashboardVM(
            num_tracked=len(persons),
            current_fps=current_fps,
            exercise_totals={ex: self._counter.total(ex) for ex in all_exercises},
            exercise_by_id=all_exercises,
            persons=[self._build_person_vm(p) for p in persons],
        )

    def build_checklist_vm(
        self,
        available: list[str],
        active: set[str],
    ) -> ExerciseChecklistVM:
        """Build checklist VM from pipeline state."""
        items = tuple(
            ExerciseChecklistItemVM(
                name=name,
                display_name=name.replace("_", " ").upper(),
                is_active=name in active,
            )
            for name in available
        )
        return ExerciseChecklistVM(items=items)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_person_vm(self, person: Person) -> PersonStatsVM:
        exercises: dict[str, ExerciseStatsVM] = {}
        for name, state in person.exercises.items():
            exercises[name] = ExerciseStatsVM(
                name=state.name,
                rep_count=state.rep_count,
                phase=state.phase,
                feedback=state.feedback,
                angles=dict(state.angles),
            )
        return PersonStatsVM(track_id=person.track_id, exercises=exercises)
