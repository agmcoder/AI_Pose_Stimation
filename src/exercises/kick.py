"""
src/exercises/kick.py

Kick detector — alternating L/R exercise.

Each leg extension/retraction cycle counts as 1 rep.
Metrics:
  - kick_angle: hip-knee-ankle angle
  - knee_lift:  (hip.y - knee.y) / torso_length  (validates real elevation)
"""
from __future__ import annotations

from ..core.interfaces import IExerciseCounter, IExerciseDetector
from ..core.types import ExerciseState, Person
from ..utils.vector_math import angle_at_vertex, torso_length
from .alternating_mixin import AlternatingSideMixin, SideState

_NAME = "kick"

# COCO indices
_LHI, _RHI = 11, 12
_LKN, _RKN = 13, 14
_LAN, _RAN = 15, 16

_SIDES = {
    "left":  {"hip": _LHI, "knee": _LKN, "ankle": _LAN},
    "right": {"hip": _RHI, "knee": _RKN, "ankle": _RAN},
}


class KickDetector(AlternatingSideMixin, IExerciseDetector):
    """
    Detector de lanzamiento de patadas.

    Cada extensión completa + retracción de una pierna cuenta como 1 rep.
    Requiere elevación mínima de rodilla para validar que es patada real.
    """

    def __init__(self, cfg: dict) -> None:
        AlternatingSideMixin.__init__(self)
        self.min_conf = cfg["min_confidence"]
        self.leg_cfg  = cfg["leg"]
        self.fb       = cfg["feedback"]

    def update(self, person: Person, counter: IExerciseCounter) -> ExerciseState:
        kps   = person.keypoints
        state = person.exercises.get(_NAME, ExerciseState(name=_NAME))

        if kps is None:
            return state

        tlen = torso_length(kps.coords, kps.scores, self.min_conf)
        if tlen is None or tlen < 1e-3:
            return state

        for side, idxs in _SIDES.items():
            hi_idx = idxs["hip"]
            kn_idx = idxs["knee"]
            an_idx = idxs["ankle"]

            if kps.scores[[hi_idx, kn_idx, an_idx]].min() < self.min_conf:
                continue

            kick_angle = angle_at_vertex(
                kps.coords[hi_idx], kps.coords[kn_idx], kps.coords[an_idx],
            )
            knee_lift = (kps.coords[hi_idx][1] - kps.coords[kn_idx][1]) / tlen

            state.angles[f"kick_{side}"] = kick_angle
            state.angles[f"knee_lift_{side}"] = knee_lift

            ss = self.get_side_state(person.track_id, side)
            ss, counted, valid, fb = self._update_side(
                ss, kick_angle, knee_lift, person, counter,
            )
            self.set_side_state(person.track_id, side, ss)

            if counted:
                state.phase    = f"retracted_{side}"
                state.feedback = fb
            elif ss.phase == "active":
                state.phase = f"extended_{side}"

        return state

    def _update_side(
        self,
        ss: SideState,
        kick_angle: float,
        knee_lift: float,
        person: Person,
        counter: IExerciseCounter,
    ) -> tuple[SideState, bool, bool, str]:
        """Returns (updated_state, rep_counted, was_valid, feedback)."""
        ext_thr     = self.leg_cfg["extend_threshold"]
        ret_thr     = self.leg_cfg["retract_threshold"]
        min_lift    = self.leg_cfg["min_knee_lift"]

        counted = False
        valid   = False
        fb      = ""

        if ss.phase == "idle":
            if kick_angle >= ext_thr:
                ss.phase        = "active"
                ss.valid_active = knee_lift >= min_lift
            elif kick_angle >= ret_thr:
                ss.phase        = "active"
                ss.valid_active = False

        elif ss.phase == "active":
            if kick_angle >= ext_thr and knee_lift >= min_lift:
                ss.valid_active = True

            elif kick_angle <= ret_thr:
                counted = True
                valid   = ss.valid_active

                if valid:
                    counter.record(person.track_id, _NAME)
                    fb = self.fb["valid_rep"]
                elif knee_lift < min_lift:
                    fb = self.fb["knee_too_low"]
                else:
                    fb = self.fb["leg_not_extended"]

                ss.phase        = "idle"
                ss.valid_active = False

        return ss, counted, valid, fb
