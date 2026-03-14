import numpy as np
from ..core.interfaces import IExerciseDetector, IExerciseCounter
from ..core.types import Person, ExerciseState
from ..utils.vector_math import angle_at_vertex

# Índices COCO keypoints
_KP = {
    "left_shoulder":  5,  "right_shoulder":  6,
    "left_hip":      11,  "right_hip":       12,
    "left_knee":     13,  "right_knee":      14,
    "left_ankle":    15,  "right_ankle":     16,
}

_SIDES = ("left", "right")


def _best_side(kps, min_conf: float) -> str | None:
    """Devuelve el lado con mayor confianza media en las 4 articulaciones clave."""
    best, best_score = None, -1.0
    for side in _SIDES:
        indices = [_KP[f"{side}_{j}"] for j in ("shoulder", "hip", "knee", "ankle")]
        scores  = kps.scores[indices]
        if scores.min() >= min_conf and scores.mean() > best_score:
            best, best_score = side, float(scores.mean())
    return best


class SquatDetector(IExerciseDetector):
    """
    Detector de sentadillas con validación dual de ángulos.

    Posición DOWN válida:
      - Rodilla: knee.down_min ≤ θ ≤ knee.down_max  (80°–120°)
      - Cadera:  hip.down_min  ≤ θ ≤ hip.down_max   (50°–75°)

    Una rep se cuenta al completar el ciclo DOWN → UP.
    Si la bajada no fue válida, se cuenta con feedback de técnica.
    """

    def __init__(self, cfg: dict):
        self.min_conf   = cfg["min_confidence"]
        self.knee_cfg   = cfg["knee"]
        self.hip_cfg    = cfg["hip"]
        self.fb         = cfg["feedback"]

    # ------------------------------------------------------------------
    # IExerciseDetector interface
    # ------------------------------------------------------------------

    def update(self, person: Person, counter: IExerciseCounter) -> ExerciseState:
        kps   = person.keypoints
        state = person.exercises.get("squat", ExerciseState(name="squat"))

        if kps is None:
            return state

        side = _best_side(kps, self.min_conf)
        if side is None:
            return state

        knee_angle, hip_angle = self._compute_angles(kps, side)

        # Actualizar ángulos en el estado para la UI
        state.angle             = knee_angle
        state.angles["knee"]    = knee_angle
        state.angles["hip"]     = hip_angle

        state = self._update_phase(state, knee_angle, hip_angle, person, counter)
        return state

    # ------------------------------------------------------------------
    # Cálculo de ángulos
    # ------------------------------------------------------------------

    def _compute_angles(self, kps, side: str) -> tuple[float, float]:
        sh = kps.coords[_KP[f"{side}_shoulder"]]
        hi = kps.coords[_KP[f"{side}_hip"]]
        kn = kps.coords[_KP[f"{side}_knee"]]
        an = kps.coords[_KP[f"{side}_ankle"]]

        knee_angle = angle_at_vertex(hi, kn, an)   # cadera → rodilla → tobillo
        hip_angle  = angle_at_vertex(sh, hi, kn)   # hombro → cadera → rodilla
        return knee_angle, hip_angle

    # ------------------------------------------------------------------
    # Máquina de estados: UP → DOWN → UP
    # ------------------------------------------------------------------

    def _update_phase(
        self,
        state: ExerciseState,
        knee: float,
        hip: float,
        person: Person,
        counter: IExerciseCounter,
    ) -> ExerciseState:

        knee_in_range = self.knee_cfg["down_min"] <= knee <= self.knee_cfg["down_max"]
        hip_in_range  = self.hip_cfg["down_min"]  <= hip  <= self.hip_cfg["down_max"]

        knee_up = knee >= self.knee_cfg["up_threshold"]
        hip_up  = hip  >= self.hip_cfg["up_threshold"]

        if state.phase == "up":
            # Detectar inicio de bajada — al menos uno de los dos en rango
            if knee_in_range or hip_in_range:
                state.phase      = "down"
                state.valid_down = knee_in_range and hip_in_range
                state.feedback   = self._down_feedback(knee_in_range, hip_in_range, knee, hip)

        elif state.phase == "down":
            # Actualizar validez mientras sigue bajando
            if knee_in_range and hip_in_range:
                state.valid_down = True
                state.feedback   = ""

            # Detectar subida completa
            elif knee_up and hip_up:
                state.phase = "up"

                if state.valid_down:
                    counter.record(person.track_id, "squat")
                    state.feedback = self.fb["valid_rep"]
                else:
                    state.feedback = self.fb["bad_form_rep"]

                state.valid_down = False

            # Fuera de rango durante la bajada → feedback en tiempo real
            else:
                state.feedback = self._down_feedback(knee_in_range, hip_in_range, knee, hip)

        return state

    # ------------------------------------------------------------------
    # Feedback específico por articulación
    # ------------------------------------------------------------------

    def _down_feedback(
        self, knee_ok: bool, hip_ok: bool, knee: float, hip: float
    ) -> str:
        if knee_ok and hip_ok:
            return ""
        issues = []
        if not knee_ok:
            issues.append(
                self.fb["knee_low"] if knee < self.knee_cfg["down_min"]
                else self.fb["knee_high"]
            )
        if not hip_ok:
            issues.append(
                self.fb["hip_low"] if hip < self.hip_cfg["down_min"]
                else self.fb["hip_high"]
            )
        return " | ".join(issues)
