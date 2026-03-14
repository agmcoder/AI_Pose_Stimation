"""
src/exercises/skipping.py

Skipping (high knees) detector — alternating L/R exercise.

Each knee elevation that reaches the threshold counts as 1 rep.
Metric: knee_height_ratio = (hip.y - knee.y) / torso_length
(positive when knee is above hip in screen coords where Y grows down).
"""
from __future__ import annotations

from ..core.interfaces import IExerciseCounter, IExerciseDetector
from ..core.types import ExerciseState, Person
from ..utils.vector_math import torso_length
from .alternating_mixin import AlternatingSideMixin, SideState

_NAME = "skipping"

# COCO indices
_LHI, _RHI = 11, 12
_LKN, _RKN = 13, 14

_SIDES = {
    "left":  {"hip": _LHI, "knee": _LKN},
    "right": {"hip": _RHI, "knee": _RKN},
}


class SkippingDetector(AlternatingSideMixin, IExerciseDetector):
    """
    Detector de skipping (rodillas altas).

    Cada elevación individual (L o R) que supera up_threshold cuenta como 1 rep.
    """

    def __init__(self, cfg: dict) -> None:
        AlternatingSideMixin.__init__(self)
        self.min_conf     = cfg["min_confidence"]
        self.knee_cfg     = cfg["knee"]
        self.fb           = cfg["feedback"]

    def update(self, person: Person, counter: IExerciseCounter) -> ExerciseState:
        kps   = person.keypoints
        state = person.exercises.get(_NAME, ExerciseState(name=_NAME))

        if kps is None:
            return state

        tlen = torso_length(kps.coords, kps.scores, self.min_conf)
        if tlen is None or tlen < 1e-3:
            return state

        for side, idxs in _SIDES.items():
            hip_idx  = idxs["hip"]
            knee_idx = idxs["knee"]

            if kps.scores[hip_idx] < self.min_conf or kps.scores[knee_idx] < self.min_conf:
                continue

            # Y crece hacia abajo: hip.y - knee.y > 0 si la rodilla está arriba
            ratio = (kps.coords[hip_idx][1] - kps.coords[knee_idx][1]) / tlen

            state.angles[f"knee_{side}"] = ratio

            ss = self.get_side_state(person.track_id, side)
            ss, counted, valid = self._update_side(ss, ratio, person, counter)
            self.set_side_state(person.track_id, side, ss)

            if counted:
                state.phase = f"up_{side}"
                state.feedback = self.fb["valid_rep"] if valid else self.fb["bad_form_rep"]
            elif ss.phase == "active":
                state.phase = f"up_{side}"

        return state

    def _update_side(
        self,
        ss: SideState,
        ratio: float,
        person: Person,
        counter: IExerciseCounter,
    ) -> tuple[SideState, bool, bool]:
        """Returns (updated_state, rep_counted, was_valid)."""
        up_thr   = self.knee_cfg["up_threshold"]
        down_thr = self.knee_cfg["down_threshold"]

        counted = False
        valid   = False

        if ss.phase == "idle":
            if ratio >= up_thr:
                ss.phase        = "active"
                ss.valid_active = True
            elif ratio >= down_thr:
                # Partial lift — track but not valid yet
                ss.phase        = "active"
                ss.valid_active = False

        elif ss.phase == "active":
            if ratio >= up_thr:
                ss.valid_active = True

            elif ratio <= down_thr:
                # Knee returned down — cycle complete
                counted = True
                valid   = ss.valid_active
                if valid:
                    counter.record(person.track_id, _NAME)
                ss.phase        = "idle"
                ss.valid_active = False

        return ss, counted, valid
