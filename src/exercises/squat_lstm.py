"""
src/exercises/squat_lstm.py

Detector de sentadillas basado en LSTM con 77 features biomecánicas.

Usa una ventana deslizante de features (ángulos, velocidad, calidad,
keypoints normalizados por bbox) para predecir la probabilidad de estar
en posición DOWN (sentadilla).

Misma interfaz IExerciseDetector que SquatDetector (ángulos).

Soporta inferencia batched: FrameProcessor llama prepare_batch(persons)
antes de procesar cada persona, ejecutando UNA sola llamada
para todas las personas del frame.

Inferencia compilada con tf.function (elimina overhead eager).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from loguru import logger

from ..core.interfaces import IExerciseDetector, IExerciseCounter
from ..core.types import Person, ExerciseState
from ..utils.body_angles import ANGLE_NAMES, compute_body_angles
from ..utils.keypoint_normalizer import normalize_keypoints
from ..utils.pose_quality import compute_pose_quality
from ..utils.velocity_tracker import VELOCITY_NAMES, VelocityTracker


class SquatLstmDetector(IExerciseDetector):
    """
    Hybrid squat detector: LSTM for activity + angles for phase/reps.

    Architecture:
        - LSTM binary classifier: P(squat) — detects IF the person is
          doing squats (activity detection). Stays high (~0.99) during
          the entire exercise.
        - Knee angle state machine: detects UP/DOWN phases and counts
          reps using biomechanical thresholds from exercises.yml.

    The LSTM does NOT predict up/down phases — it was trained as a
    binary squat-vs-none classifier. Phase detection relies on knee
    angle oscillation (180° standing → ~90° squatting → 180° standing).

    Features (77 total):
        - Quality (2): mean_kpt_conf, visible_kpt_count
        - Body angles (19): elbow/shoulder/hip/knee/ankle L/R, trunk, neck,
          pelvis, shoulder/hip line, arm raise, leg abduction
        - Velocity (5): hip_center_y_norm, hip_center_vy,
          knee_left/right_vy, mean_joint_speed
        - Keypoints normalized (51): 17 × (x_norm, y_norm, conf)
    """

    def __init__(self, cfg: dict):
        self.min_conf       = cfg["min_confidence"]
        self.window_size    = cfg.get("window_size", 24)
        self.down_threshold = cfg.get("down_prob_threshold", 0.7)
        self.up_threshold   = cfg.get("up_prob_threshold", 0.3)
        self.min_buffer     = cfg.get("min_buffer_frames", 5)
        self.fb             = cfg["feedback"]

        # ── Knee-angle thresholds for phase detection ─────────────────
        knee_cfg = cfg.get("knee", {})
        self.knee_down_max   = knee_cfg.get("down_max", 120)    # ≤ this → entering DOWN
        self.knee_down_min   = knee_cfg.get("down_min", 80)     # valid range lower bound
        self.knee_up_thresh  = knee_cfg.get("up_threshold", 155) # ≥ this → back to UP

        # ── Resolve model & side-file paths ──────────────────────────────
        model_path = Path(cfg["model_path"])
        stem = model_path.stem  # "squat_final"
        model_dir = model_path.parent

        meta_path = model_dir / f"{stem}_meta.json"
        scaler_path = model_dir / f"{stem}_scaler.npz"

        # ── Load feature column order from metadata ──────────────────────
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._feature_cols: list[str] = meta["feature_cols"]
            # Override window_size from meta if present
            if "window_size" in meta:
                self.window_size = meta["window_size"]
            logger.info(
                f"📋 LSTM meta: {len(self._feature_cols)} features, "
                f"window_size={self.window_size}"
            )
        else:
            raise FileNotFoundError(
                f"Metadata no encontrada: {meta_path}. "
                f"Necesaria para conocer las features del modelo."
            )

        self.n_features = len(self._feature_cols)

        # ── Load StandardScaler ──────────────────────────────────────────
        if scaler_path.exists():
            scaler_data = np.load(scaler_path, allow_pickle=True)
            self._scaler_mean = scaler_data["mean"].astype(np.float32)
            self._scaler_scale = scaler_data["scale"].astype(np.float32)
            # Avoid division by zero
            self._scaler_scale[self._scaler_scale == 0] = 1.0
            logger.info(f"📊 Scaler cargado: {scaler_path}")
        else:
            logger.warning(
                f"⚠️ Scaler no encontrado: {scaler_path}. "
                f"Inferencia sin normalización."
            )
            self._scaler_mean = np.zeros(self.n_features, dtype=np.float32)
            self._scaler_scale = np.ones(self.n_features, dtype=np.float32)

        # ── Velocity tracker (stateful, per-track) ───────────────────────
        self._velocity_tracker = VelocityTracker(min_conf=self.min_conf)

        # ── TF: ocultar GPU para evitar contención CUDA con YOLO ─────────
        gpus_available = tf.config.list_physical_devices("GPU")
        device_str = "CPU"
        if gpus_available:
            tf.config.set_visible_devices([], "GPU")
            device_str = "CPU (oculta GPU para evitar contención con YOLO)"

        self._model = tf.keras.models.load_model(str(model_path), compile=False)
        self._predict_fn = tf.function(
            lambda x: self._model(x, training=False),
            input_signature=[
                tf.TensorSpec((None, self.window_size, self.n_features), tf.float32)
            ],
        )
        # Warmup: fuerza el trazado del grafo fuera del bucle de frames
        self._predict_fn(tf.zeros((1, self.window_size, self.n_features)))

        logger.info(
            f"✅ LSTM squat model cargado en {device_str} (tf.function): "
            f"{model_path} | {self.n_features} features, ws={self.window_size}"
        )

        # Buffer deslizante por persona (track_id → feature vectors)
        self._buffers: dict[int, list[np.ndarray]] = defaultdict(list)
        # Caché de probabilidades pre-computadas por prepare_batch
        self._cached_probs: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Feature extraction — replica la lógica de record_builder + CSV
    # ------------------------------------------------------------------

    def _extract_features(self, person: Person) -> Optional[np.ndarray]:
        """
        Extrae el vector de 77 features para una persona en el frame actual.

        Reutiliza las mismas utilidades que record_builder.py para mantener
        coherencia entrenamiento ↔ inferencia.

        Returns None if keypoints are not available.
        """
        if person.keypoints is None:
            return None

        coords = person.keypoints.coords
        scores = person.keypoints.scores

        # 1. Quality metrics
        quality = compute_pose_quality(scores)

        # 2. Body angles (19 values, None → 0.0)
        body_angles = compute_body_angles(coords, scores) if quality.is_valid_pose else {}

        # 3. Velocity (5 values, None → 0.0)
        velocity = self._velocity_tracker.update(
            person.track_id, coords, scores, person.bbox,
        )

        # 4. Keypoints normalized by bbox (17 × 2)
        kp_norm = normalize_keypoints(coords, person.bbox)

        # ── Build feature vector in the exact column order from meta ─────
        feature_map: dict[str, float] = {}

        # Quality
        feature_map["mean_kpt_conf"] = quality.mean_kpt_conf
        feature_map["visible_kpt_count"] = float(quality.visible_kpt_count)

        # Angles
        for name in ANGLE_NAMES:
            val = body_angles.get(name)
            feature_map[name] = float(val) if val is not None else 0.0

        # Velocity
        for name in VELOCITY_NAMES:
            val = velocity.get(name)
            feature_map[name] = float(val) if val is not None else 0.0

        # Keypoints normalized + confidence
        for i in range(17):
            feature_map[f"kp{i}_x_norm"] = float(kp_norm[i, 0])
            feature_map[f"kp{i}_y_norm"] = float(kp_norm[i, 1])
            feature_map[f"kp{i}_conf"] = float(scores[i])

        # Assemble in the exact order the model expects
        vec = np.array(
            [feature_map.get(col, 0.0) for col in self._feature_cols],
            dtype=np.float32,
        )
        return vec

    # ------------------------------------------------------------------
    # Batch inference — llamado por FrameProcessor antes del per-person
    # ------------------------------------------------------------------

    def prepare_batch(self, persons: list[Person]) -> None:
        """
        Pre-computa predicciones LSTM para TODAS las personas en una sola
        llamada. update() luego lee del caché sin tocar el modelo.
        """
        self._cached_probs.clear()

        sequences = []
        track_ids = []

        for person in persons:
            if person.keypoints is None:
                continue
            if min(person.keypoints.scores[11], person.keypoints.scores[12]) < self.min_conf:
                continue

            features = self._extract_features(person)
            if features is None:
                continue

            buf = self._buffers[person.track_id]
            buf.append(features)
            if len(buf) > self.window_size:
                buf.pop(0)

            if len(buf) < self.min_buffer:
                continue

            padded = list(buf)
            while len(padded) < self.window_size:
                padded.insert(0, padded[0])

            sequences.append(np.array(padded, dtype=np.float32))
            track_ids.append(person.track_id)

        if not sequences:
            return

        batch = np.stack(sequences)  # (N, window_size, n_features)

        # Apply StandardScaler: (x - mean) / scale
        batch = (batch - self._scaler_mean) / self._scaler_scale

        preds = self._predict_fn(tf.constant(batch))

        for tid, pred in zip(track_ids, preds.numpy()):
            self._cached_probs[tid] = float(pred[0])

    # ------------------------------------------------------------------
    # IExerciseDetector interface
    # ------------------------------------------------------------------

    def update(self, person: Person, counter: IExerciseCounter) -> ExerciseState:
        state = person.exercises.get("squat", ExerciseState(name="squat"))

        if person.track_id in self._cached_probs:
            prob = self._cached_probs.pop(person.track_id)
            # Store LSTM probability for downstream use (DeferredLabelBuffer)
            state.angle = prob
            return self._update_phase(state, prob, person, counter)

        # Persona sin keypoints o bajo min_buffer (ya procesada en prepare_batch)
        return state

    # ------------------------------------------------------------------
    # Hybrid state machine: LSTM for activity + knee angles for phase
    # ------------------------------------------------------------------

    def _get_knee_angle(self, person: Person) -> float | None:
        """
        Compute average knee angle (left + right) from keypoints.
        Returns None if keypoints are unavailable or low confidence.
        """
        if person.keypoints is None:
            return None

        coords = person.keypoints.coords
        scores = person.keypoints.scores

        # Keypoint indices: hip=11/12, knee=13/14, ankle=15/16
        # Check minimum confidence for knee angle computation
        required = [11, 12, 13, 14, 15, 16]
        if any(scores[i] < self.min_conf for i in required):
            return None

        body_angles = compute_body_angles(coords, scores)
        knee_l = body_angles.get("knee_left_deg")
        knee_r = body_angles.get("knee_right_deg")

        if knee_l is not None and knee_r is not None:
            return (knee_l + knee_r) / 2.0
        return knee_l or knee_r

    def _update_phase(
        self,
        state: ExerciseState,
        prob: float,
        person: Person,
        counter: IExerciseCounter,
    ) -> ExerciseState:
        """
        Hybrid phase detection:
          - LSTM prob > down_threshold → we're in a squat activity
          - Within active squat: knee angle drives UP/DOWN transitions
          - Rep counted on DOWN → UP transition (if valid_down)

        The LSTM is a binary classifier that outputs ~0.99 during the
        entire squat session. It does NOT oscillate between phases.
        Knee angles (180° standing ↔ ~90° squatting) provide the
        actual phase signal.
        """
        # ── Gate: only run phase detection if LSTM says "squat" ───────
        # Use down_threshold as activity gate (prob >= 0.85 → active squat)
        if prob < self.up_threshold:
            # LSTM says clearly NOT a squat — reset to "up" (standing)
            state.phase = "up"
            state.valid_down = False
            return state

        # ── Phase detection via knee angles ───────────────────────────
        knee_angle = self._get_knee_angle(person)
        if knee_angle is None:
            # Can't compute angle — keep current phase, don't transition
            return state

        # Store the knee angle for downstream consumers
        state.angles["knee_avg"] = knee_angle

        if state.phase == "up":
            # Transition UP → DOWN when knees bend past threshold
            if knee_angle <= self.knee_down_max:
                state.phase = "down"
                # Validate depth: knee angle within acceptable range
                if self.knee_down_min <= knee_angle <= self.knee_down_max:
                    state.valid_down = True
                state.feedback = ""

        elif state.phase == "down":
            # Continue validating depth while in DOWN
            if self.knee_down_min <= knee_angle <= self.knee_down_max:
                state.valid_down = True

            # Transition DOWN → UP when knees extend past threshold
            if knee_angle >= self.knee_up_thresh:
                state.phase = "up"
                if state.valid_down:
                    counter.record(person.track_id, "squat")
                    state.rep_count = counter.by_id("squat").get(
                        person.track_id, 0
                    )
                    state.feedback = self.fb["valid_rep"]
                else:
                    state.feedback = self.fb["bad_form_rep"]
                state.valid_down = False

        return state

