"""
src/exercises/jumping_jacks.py

Jumping Jacks detector — bilateral symmetric exercise.

States: CLOSED → OPEN → CLOSED = 1 rep
Metrics:
  - arm_angle:  average shoulder-elbow-wrist angle (L+R)
  - leg_spread: ankle horizontal distance / hip width
"""
from __future__ import annotations

import numpy as np

from ..core.interfaces import IExerciseCounter, IExerciseDetector
from ..core.types import ExerciseState, Person
from ..utils.vector_math import angle_at_vertex, hip_width

_NAME = "jumping_jacks"

# COCO indices
_LSH, _RSH = 5, 6
_LEL, _REL = 7, 8
_LWR, _RWR = 9, 10
_LHI, _RHI = 11, 12
_LAN, _RAN = 15, 16

_REQUIRED = [_LSH, _RSH, _LEL, _REL, _LWR, _RWR, _LHI, _RHI, _LAN, _RAN]


class JumpingJacksDetector(IExerciseDetector):
    """
    Detector de jumping jacks.

    Fase OPEN válida: brazos arriba Y piernas separadas simultáneamente.
    """

    def __init__(self, cfg: dict) -> None:
        self.min_conf       = cfg["min_confidence"]
        self.arm_cfg        = cfg["arm"]
        self.leg_cfg        = cfg["leg"]
        self.fb             = cfg["feedback"]

    def update(self, person: Person, counter: IExerciseCounter) -> ExerciseState:
        kps   = person.keypoints
        state = person.exercises.get(_NAME, ExerciseState(name=_NAME))

        if kps is None:
            return state

        if kps.scores[_REQUIRED].min() < self.min_conf:
            return state

        arm_angle  = self._avg_arm_angle(kps.coords)
        leg_spread = self._leg_spread(kps.coords, kps.scores)

        if leg_spread is None:
            return state

        state.angles["arm"] = arm_angle
        state.angles["leg_spread"] = leg_spread

        state = self._update_phase(state, arm_angle, leg_spread, person, counter)
        return state

    # ── Metrics ────────────────────────────────────────────────────────────

    def _avg_arm_angle(self, coords: np.ndarray) -> float:
        left  = angle_at_vertex(coords[_LSH], coords[_LEL], coords[_LWR])
        right = angle_at_vertex(coords[_RSH], coords[_REL], coords[_RWR])
        return (left + right) * 0.5

    def _leg_spread(self, coords: np.ndarray, scores: np.ndarray) -> float | None:
        hw = hip_width(coords, scores, self.min_conf)
        if hw is None or hw < 1e-3:
            return None
        ankle_dist = abs(float(coords[_LAN][0] - coords[_RAN][0]))
        return ankle_dist / hw

    # ── State machine: CLOSED → OPEN → CLOSED ─────────────────────────────

    def _update_phase(
        self,
        state: ExerciseState,
        arm: float,
        leg: float,
        person: Person,
        counter: IExerciseCounter,
    ) -> ExerciseState:

        arms_open = arm >= self.arm_cfg["open_threshold"]
        legs_open = leg >= self.leg_cfg["open_threshold"]

        arms_closed = arm <= self.arm_cfg["closed_threshold"]
        legs_closed = leg <= self.leg_cfg["closed_threshold"]

        if state.phase == "closed" or state.phase == "up":
            # Detect opening
            if arms_open or legs_open:
                state.phase      = "open"
                state.valid_down = arms_open and legs_open
                state.feedback   = self._open_feedback(arms_open, legs_open)

        elif state.phase == "open":
            if arms_open and legs_open:
                state.valid_down = True
                state.feedback   = ""

            elif arms_closed and legs_closed:
                state.phase = "closed"

                if state.valid_down:
                    counter.record(person.track_id, _NAME)
                    state.feedback = self.fb["valid_rep"]
                else:
                    state.feedback = self.fb["bad_form_rep"]

                state.valid_down = False
            else:
                state.feedback = self._open_feedback(arms_open, legs_open)

        return state

    def _open_feedback(self, arms_ok: bool, legs_ok: bool) -> str:
        if arms_ok and legs_ok:
            return ""
        issues = []
        if not arms_ok:
            issues.append(self.fb["arms_low"])
        if not legs_ok:
            issues.append(self.fb["legs_narrow"])
        return " | ".join(issues)
