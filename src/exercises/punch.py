"""
src/exercises/punch.py

Punch detector — alternating L/R exercise.

Each arm extension/retraction cycle counts as 1 rep.
Metric: arm_extension = shoulder-elbow-wrist angle.
"""
from __future__ import annotations

from ..core.interfaces import IExerciseCounter, IExerciseDetector
from ..core.types import ExerciseState, Person
from ..utils.vector_math import angle_at_vertex
from .alternating_mixin import AlternatingSideMixin, SideState

_NAME = "punch"

# COCO indices
_LSH, _RSH = 5, 6
_LEL, _REL = 7, 8
_LWR, _RWR = 9, 10

_SIDES = {
    "left":  {"shoulder": _LSH, "elbow": _LEL, "wrist": _LWR},
    "right": {"shoulder": _RSH, "elbow": _REL, "wrist": _RWR},
}


class PunchDetector(AlternatingSideMixin, IExerciseDetector):
    """
    Detector de lanzamiento de puños.

    Cada extensión completa + retracción de un brazo cuenta como 1 rep.
    """

    def __init__(self, cfg: dict) -> None:
        AlternatingSideMixin.__init__(self)
        self.min_conf = cfg["min_confidence"]
        self.arm_cfg  = cfg["arm"]
        self.fb       = cfg["feedback"]

    def update(self, person: Person, counter: IExerciseCounter) -> ExerciseState:
        kps   = person.keypoints
        state = person.exercises.get(_NAME, ExerciseState(name=_NAME))

        if kps is None:
            return state

        for side, idxs in _SIDES.items():
            sh_idx = idxs["shoulder"]
            el_idx = idxs["elbow"]
            wr_idx = idxs["wrist"]

            if kps.scores[[sh_idx, el_idx, wr_idx]].min() < self.min_conf:
                continue

            arm_angle = angle_at_vertex(
                kps.coords[sh_idx], kps.coords[el_idx], kps.coords[wr_idx],
            )

            state.angles[f"arm_{side}"] = arm_angle

            ss = self.get_side_state(person.track_id, side)
            ss, counted, valid = self._update_side(ss, arm_angle, person, counter)
            self.set_side_state(person.track_id, side, ss)

            if counted:
                state.phase = f"retracted_{side}"
                state.feedback = self.fb["valid_rep"] if valid else self.fb["bad_form_rep"]
            elif ss.phase == "active":
                state.phase = f"extended_{side}"

        return state

    def _update_side(
        self,
        ss: SideState,
        arm_angle: float,
        person: Person,
        counter: IExerciseCounter,
    ) -> tuple[SideState, bool, bool]:
        """Returns (updated_state, rep_counted, was_valid)."""
        ext_thr = self.arm_cfg["extend_threshold"]
        ret_thr = self.arm_cfg["retract_threshold"]

        counted = False
        valid   = False

        if ss.phase == "idle":
            if arm_angle >= ext_thr:
                ss.phase        = "active"
                ss.valid_active = True
            elif arm_angle >= ret_thr:
                ss.phase        = "active"
                ss.valid_active = False

        elif ss.phase == "active":
            if arm_angle >= ext_thr:
                ss.valid_active = True

            elif arm_angle <= ret_thr:
                counted = True
                valid   = ss.valid_active
                if valid:
                    counter.record(person.track_id, _NAME)
                ss.phase        = "idle"
                ss.valid_active = False

        return ss, counted, valid
