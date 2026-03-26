"""
src/exercises/lateral_jump.py

Lateral Jump detector — positional exercise.

States: CENTER → SIDE → CENTER = 1 rep
Metrics:
  - lateral_delta: horizontal displacement of hip center / hip width
  - vertical_lift: vertical displacement / torso length (confirms real jump)
"""
from __future__ import annotations

from ..core.interfaces import IExerciseCounter, IExerciseDetector
from ..core.types import ExerciseState, Person
from ..utils.vector_math import hip_width, midpoint, torso_length

_NAME = "lateral_jump"

# COCO indices
_LHI, _RHI = 11, 12


class LateralJumpDetector(IExerciseDetector):
    """
    Detector de saltos laterales.

    Usa línea base adaptativa (EMA) del centro de caderas en X.
    Requiere componente vertical para distinguir de caminata.
    """

    def __init__(self, cfg: dict) -> None:
        self.min_conf = cfg["min_confidence"]
        self.jump_cfg = cfg["jump"]
        self.fb       = cfg["feedback"]
        # Per-track baselines: {track_id: (baseline_x, baseline_y)}
        self._baselines: dict[int, tuple[float, float]] = {}

    def update(self, person: Person, counter: IExerciseCounter) -> ExerciseState:
        kps   = person.keypoints
        state = person.exercises.get(_NAME, ExerciseState(name=_NAME))

        if kps is None:
            return state

        tlen = torso_length(kps.coords, kps.scores, self.min_conf)
        hw   = hip_width(kps.coords, kps.scores, self.min_conf)
        if tlen is None or tlen < 1e-3 or hw is None or hw < 1e-3:
            return state

        hip_center = midpoint(kps.coords[_LHI], kps.coords[_RHI])
        hip_x = float(hip_center[0])
        hip_y = float(hip_center[1])

        tid = person.track_id
        if tid not in self._baselines:
            self._baselines[tid] = (hip_x, hip_y)

        base_x, base_y = self._baselines[tid]

        lateral_delta = abs(hip_x - base_x) / hw
        vertical_lift = (base_y - hip_y) / tlen   # positive = upward

        state.angles["lateral_delta"] = lateral_delta
        state.angles["vertical_lift"] = vertical_lift

        state = self._update_phase(
            state, lateral_delta, vertical_lift, hip_x, hip_y, person, counter,
        )
        return state

    def _update_phase(
        self,
        state: ExerciseState,
        lat: float,
        vert: float,
        hip_x: float,
        hip_y: float,
        person: Person,
        counter: IExerciseCounter,
    ) -> ExerciseState:

        lat_thr    = self.jump_cfg["lateral_threshold"]
        ret_thr    = self.jump_cfg["return_threshold"]
        min_vert   = self.jump_cfg["min_vertical_lift"]
        ema_alpha  = self.jump_cfg["ema_alpha"]
        tid        = person.track_id

        if state.phase == "center" or state.phase == "up":
            has_lateral  = lat >= lat_thr
            has_vertical = vert >= min_vert

            if has_lateral and has_vertical:
                state.phase      = "side"
                state.valid_down = True
                state.feedback   = ""
            elif has_lateral and not has_vertical:
                state.phase      = "side"
                state.valid_down = False
                state.feedback   = self.fb["no_vertical_lift"]
            else:
                # Update baseline with EMA while centered
                bx, by = self._baselines[tid]
                bx += ema_alpha * (hip_x - bx)
                by += ema_alpha * (hip_y - by)
                self._baselines[tid] = (bx, by)

                if lat >= ret_thr:
                    state.feedback = self.fb["jump_too_short"]

        elif state.phase == "side":
            if lat <= ret_thr:
                state.phase = "center"

                if state.valid_down:
                    counter.record(tid, _NAME)
                    state.feedback = self.fb["valid_rep"]
                else:
                    state.feedback = self.fb["bad_form_rep"]

                state.valid_down = False
                # Reset baseline to landing position
                self._baselines[tid] = (hip_x, hip_y)

        return state
