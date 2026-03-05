import numpy as np
from ..core.interfaces import IExerciseDetector
from ..core.types import Person, ExerciseState

# Índices COCO keypoints
KEYPOINTS = {
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

def _angle(a, b, c) -> float:
    """Ángulo en B dado vectores BA y BC."""
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

class SquatDetector(IExerciseDetector):
    def __init__(self, cfg: dict):
        self.down_thr = cfg["down_threshold"]   # 90°
        self.up_thr   = cfg["up_threshold"]     # 160°
        self.min_conf = cfg["min_confidence"]
        self._states: dict[int, ExerciseState] = {}

    def update(self, person: Person) -> ExerciseState:
        tid  = person.track_id
        kps  = person.keypoints
        state = self._states.setdefault(
            tid, ExerciseState(name="squat"))

        if kps is None:
            return state

        # Usar lado con mayor confianza promedio
        for side in ("left", "right"):
            hi = KEYPOINTS[f"{side}_hip"]
            kn = KEYPOINTS[f"{side}_knee"]
            an = KEYPOINTS[f"{side}_ankle"]
            if min(kps.scores[[hi, kn, an]]) < self.min_conf:
                continue

            angle = _angle(kps.coords[hi], kps.coords[kn], kps.coords[an])
            state.angle = angle

            if angle < self.down_thr and state.phase == "up":
                state.phase = "down"
            elif angle > self.up_thr and state.phase == "down":
                state.phase = "up"
                state.rep_count += 1
                state.feedback = f"✅ Rep {state.rep_count}"
            break

        self._states[tid] = state
        return state
