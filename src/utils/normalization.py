import numpy as np


def normalize_keypoints(coords: np.ndarray) -> np.ndarray:
    """
    Normalización biomecánica: centra en caderas y escala por longitud de torso.

    Misma lógica que data_processing.py para mantener coherencia entrenamiento ↔ inferencia.
    Retorna array aplanado de 34 valores (17 puntos × 2 coordenadas).
    """
    kp = np.asarray(coords, dtype=np.float32)

    # Centro de caderas (COCO indices: 11, 12)
    hip_center = (kp[11] + kp[12]) / 2
    kp_centered = kp - hip_center

    # Longitud de torso (centro hombros → centro caderas)
    shoulder_center = (kp[5] + kp[6]) / 2
    torso_size = float(np.linalg.norm(shoulder_center - hip_center))

    if torso_size == 0:
        torso_size = 1.0

    kp_normalized = kp_centered / torso_size
    return kp_normalized.flatten()
