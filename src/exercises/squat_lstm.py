import numpy as np
import tensorflow as tf
from collections import defaultdict
from loguru import logger

from ..core.interfaces import IExerciseDetector, IExerciseCounter
from ..core.types import Person, ExerciseState
from ..utils.normalization import normalize_keypoints


class SquatLstmDetector(IExerciseDetector):
    """
    Detector de sentadillas basado en LSTM.

    Usa una ventana deslizante de keypoints normalizados para predecir
    la probabilidad de estar en posición DOWN (sentadilla).
    Misma interfaz IExerciseDetector que SquatDetector (ángulos).

    Soporta inferencia batched: FrameProcessor llama prepare_batch(persons)
    antes de procesar cada persona, ejecutando UNA sola llamada
    para todas las personas del frame.

    Inferencia compilada con tf.function (elimina overhead eager ~135ms → ~1ms).
    """

    def __init__(self, cfg: dict):
        self.min_conf        = cfg["min_confidence"]
        self.window_size     = cfg.get("window_size", 30)
        self.down_threshold  = cfg.get("down_prob_threshold", 0.7)
        self.up_threshold    = cfg.get("up_prob_threshold", 0.3)
        self.min_buffer      = cfg.get("min_buffer_frames", 5)
        self.fb              = cfg["feedback"]

        # Determinar device: ocultar GPU a TensorFlow para evitar contención CUDA
        # con PyTorch/YOLO. Modelo de 503KB donde CPU compilado (~2ms para N personas)
        # supera a GPU (cuDNN mismatch + overhead + contención con YOLO).
        # TF reconstruye LSTM con ops genéricos. PyTorch/YOLO no se ve afectado.
        gpus_available = tf.config.list_physical_devices("GPU")
        device_str = "CPU"  # default
        if gpus_available:
            tf.config.set_visible_devices([], "GPU")
            device_str = "CPU (oculta GPU para evitar contención con YOLO)"

        model_path = cfg["model_path"]
        self._model = tf.keras.models.load_model(model_path, compile=False)
        self._predict_fn = tf.function(
            lambda x: self._model(x, training=False),
            input_signature=[tf.TensorSpec((None, self.window_size, 34), tf.float32)],
        )
        # Warmup: fuerza el trazado del grafo fuera del bucle de frames
        self._predict_fn(tf.zeros((1, self.window_size, 34)))

        logger.info(f"✅ LSTM squat model cargado en {device_str} (tf.function): {model_path}")

        # Buffer deslizante por persona (track_id → frames normalizados)
        self._buffers: dict[int, list[np.ndarray]] = defaultdict(list)
        # Caché de probabilidades pre-computadas por prepare_batch
        self._cached_probs: dict[int, float] = {}

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
            kps = person.keypoints
            if kps is None:
                continue
            if min(kps.scores[11], kps.scores[12]) < self.min_conf:
                continue

            normalized = normalize_keypoints(kps.coords)
            buf = self._buffers[person.track_id]
            buf.append(normalized)
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

        batch = np.stack(sequences)  # (N, window_size, 34)
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
            state.angle = prob
            return self._update_phase(state, prob, person, counter)

        # Persona sin keypoints o bajo min_buffer (ya procesada en prepare_batch)
        return state

    # ------------------------------------------------------------------
    # Máquina de estados: UP → DOWN → UP
    # ------------------------------------------------------------------

    def _update_phase(
        self,
        state: ExerciseState,
        prob: float,
        person: Person,
        counter: IExerciseCounter,
    ) -> ExerciseState:

        if state.phase == "up":
            if prob >= self.down_threshold:
                state.phase      = "down"
                state.valid_down = True
                state.feedback   = ""

        elif state.phase == "down":
            if prob <= self.up_threshold:
                state.phase = "up"
                if state.valid_down:
                    counter.record(person.track_id, "squat")
                    state.feedback = self.fb["valid_rep"]
                else:
                    state.feedback = self.fb["bad_form_rep"]
                state.valid_down = False

        return state
