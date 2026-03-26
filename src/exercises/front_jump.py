"""
src/exercises/front_jump.py

Front Jump / Box Jump detector — positional exercise.

States: GROUND → AIR → GROUND = 1 rep
Metric: vertical displacement of hip center relative to an adaptive
        baseline (EMA), normalized by torso length.
"""
from __future__ import annotations

from ..core.interfaces import IExerciseCounter, IExerciseDetector
from ..core.types import ExerciseState, Person
from ..utils.vector_math import midpoint, torso_length

_NAME = "front_jump"

# COCO indices
_LHI, _RHI = 11, 12


class FrontJumpDetector(IExerciseDetector):
    """
    Detector de saltos frontales.

    Usa línea base adaptativa (EMA) del centro de caderas.
    Un salto se cuenta cuando el cuerpo sube por encima del umbral
    y luego regresa.
    """

    def __init__(self, cfg: dict) -> None:
        self.min_conf  = cfg["min_confidence"]
        self.jump_cfg  = cfg["jump"]
        self.fb        = cfg["feedback"]
        # Per-track baselines: {track_id: baseline_y}
        self._baselines: dict[int, float] = {}

    def update(self, person: Person, counter: IExerciseCounter) -> ExerciseState:
        kps   = person.keypoints
        state = person.exercises.get(_NAME, ExerciseState(name=_NAME))

        if kps is None:
            return state

        tlen = torso_length(kps.coords, kps.scores, self.min_conf)
        if tlen is None or tlen < 1e-3:
            return state

        if kps.scores[_LHI] < self.min_conf or kps.scores[_RHI] < self.min_conf:
            return state

        hip_center_y = float(midpoint(kps.coords[_LHI], kps.coords[_RHI])[1])

        # Initialize or update baseline
        tid = person.track_id
        if tid not in self._baselines:
            self._baselines[tid] = hip_center_y

        baseline = self._baselines[tid]
        # Y crece hacia abajo: baseline - current > 0 significa que subió
        lift = (baseline - hip_center_y) / tlen

        state.angles["lift"] = lift

        state = self._update_phase(state, lift, hip_center_y, person, counter)
        return state

    def _update_phase(
        self,
        state: ExerciseState,
        lift: float,
        hip_y: float,
        person: Person,
        counter: IExerciseCounter,
    ) -> ExerciseState:

        jump_thr  = self.jump_cfg["jump_threshold"]
        land_thr  = self.jump_cfg["land_threshold"]
        ema_alpha = self.jump_cfg["ema_alpha"]
        tid       = person.track_id

        if state.phase == "ground" or state.phase == "up":
            if lift >= jump_thr:
                state.phase      = "air"
                state.valid_down = True
                state.feedback   = ""
            else:
                # Update baseline with EMA while on ground
                self._baselines[tid] += ema_alpha * (hip_y - self._baselines[tid])

                if lift >= land_thr:
                    state.feedback = self.fb["jump_too_low"]

        elif state.phase == "air":
            if lift <= land_thr:
                state.phase = "ground"

                if state.valid_down:
                    counter.record(tid, _NAME)
                    state.feedback = self.fb["valid_rep"]
                else:
                    state.feedback = self.fb["bad_form_rep"]

                state.valid_down = False
                # Reset baseline to current position after landing
                self._baselines[tid] = hip_y

        return state
