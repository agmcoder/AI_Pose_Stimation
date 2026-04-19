"""
src/utils/body_angles.py

Canonical body angle computation — exercise-agnostic.

Computes 19 angles from COCO-17 keypoints.  Each angle is returned only
when ALL required keypoints exceed a minimum confidence.  Otherwise the
value is ``None``.

Reuses ``angle_at_vertex`` and ``midpoint`` from ``vector_math``.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .vector_math import angle_at_vertex, midpoint

# ── COCO-17 keypoint indices ────────────────────────────────────────────────
_NOSE = 0
_LSH, _RSH = 5, 6      # shoulders
_LEL, _REL = 7, 8      # elbows
_LWR, _RWR = 9, 10     # wrists
_LHI, _RHI = 11, 12    # hips
_LKN, _RKN = 13, 14    # knees
_LAN, _RAN = 15, 16    # ankles

# Canonical angle names — order matches the CSV spec
ANGLE_NAMES: list[str] = [
    "elbow_left_deg",   "elbow_right_deg",
    "shoulder_left_deg", "shoulder_right_deg",
    "hip_left_deg",      "hip_right_deg",
    "knee_left_deg",     "knee_right_deg",
    "ankle_left_deg",    "ankle_right_deg",
    "trunk_deg",         "neck_deg",        "pelvis_deg",
    "shoulder_line_deg", "hip_line_deg",
    "left_arm_raise_deg",  "right_arm_raise_deg",
    "left_leg_abduction_deg", "right_leg_abduction_deg",
]


def _ok(scores: np.ndarray, indices: list[int], min_conf: float) -> bool:
    """All listed keypoints exceed *min_conf*."""
    return bool(scores[indices].min() >= min_conf)


def _angle_to_vertical(a: np.ndarray, b: np.ndarray) -> float:
    """
    Angle (degrees) of vector AB w.r.t. the downward vertical (0 = straight down).
    Positive = tilted rightward in image space.
    """
    d = b - a
    # vertical reference: pointing downward (0, 1) in image coords
    cos_val = d[1] / (np.linalg.norm(d) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def _angle_to_horizontal(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (degrees) of vector AB w.r.t. the horizontal axis."""
    d = b - a
    return float(np.degrees(np.arctan2(abs(d[1]), abs(d[0]) + 1e-6)))


def compute_body_angles(
    coords: np.ndarray,
    scores: np.ndarray,
    min_conf: float = 0.3,
) -> Dict[str, Optional[float]]:
    """
    Compute canonical body angles from COCO-17 keypoints.

    Parameters
    ----------
    coords : np.ndarray, shape (17, 2)
    scores : np.ndarray, shape (17,)
    min_conf : float
        Minimum keypoint confidence to include in a given angle.

    Returns
    -------
    dict
        Keys are ``ANGLE_NAMES``, values are float degrees or ``None``.
    """
    angles: Dict[str, Optional[float]] = {name: None for name in ANGLE_NAMES}

    # ── Joint angles ─────────────────────────────────────────────────────
    # Elbows
    if _ok(scores, [_LSH, _LEL, _LWR], min_conf):
        angles["elbow_left_deg"] = angle_at_vertex(coords[_LSH], coords[_LEL], coords[_LWR])
    if _ok(scores, [_RSH, _REL, _RWR], min_conf):
        angles["elbow_right_deg"] = angle_at_vertex(coords[_RSH], coords[_REL], coords[_RWR])

    # Shoulders
    if _ok(scores, [_LEL, _LSH, _LHI], min_conf):
        angles["shoulder_left_deg"] = angle_at_vertex(coords[_LEL], coords[_LSH], coords[_LHI])
    if _ok(scores, [_REL, _RSH, _RHI], min_conf):
        angles["shoulder_right_deg"] = angle_at_vertex(coords[_REL], coords[_RSH], coords[_RHI])

    # Hips
    if _ok(scores, [_LSH, _LHI, _LKN], min_conf):
        angles["hip_left_deg"] = angle_at_vertex(coords[_LSH], coords[_LHI], coords[_LKN])
    if _ok(scores, [_RSH, _RHI, _RKN], min_conf):
        angles["hip_right_deg"] = angle_at_vertex(coords[_RSH], coords[_RHI], coords[_RKN])

    # Knees
    if _ok(scores, [_LHI, _LKN, _LAN], min_conf):
        angles["knee_left_deg"] = angle_at_vertex(coords[_LHI], coords[_LKN], coords[_LAN])
    if _ok(scores, [_RHI, _RKN, _RAN], min_conf):
        angles["knee_right_deg"] = angle_at_vertex(coords[_RHI], coords[_RKN], coords[_RAN])

    # Ankles — use a synthetic foot point projected along shin direction
    if _ok(scores, [_LKN, _LAN], min_conf):
        shin = coords[_LAN] - coords[_LKN]
        foot_proxy = coords[_LAN] + shin * 0.3  # project forward
        angles["ankle_left_deg"] = angle_at_vertex(coords[_LKN], coords[_LAN], foot_proxy)
    if _ok(scores, [_RKN, _RAN], min_conf):
        shin = coords[_RAN] - coords[_RKN]
        foot_proxy = coords[_RAN] + shin * 0.3
        angles["ankle_right_deg"] = angle_at_vertex(coords[_RKN], coords[_RAN], foot_proxy)

    # ── Orientation angles ───────────────────────────────────────────────
    # Midpoints
    sh_ok = _ok(scores, [_LSH, _RSH], min_conf)
    hi_ok = _ok(scores, [_LHI, _RHI], min_conf)

    if sh_ok:
        mid_sh = midpoint(coords[_LSH], coords[_RSH])
    if hi_ok:
        mid_hi = midpoint(coords[_LHI], coords[_RHI])

    # Trunk inclination: midpoint(shoulders) → midpoint(hips) vs vertical
    if sh_ok and hi_ok:
        angles["trunk_deg"] = _angle_to_vertical(mid_sh, mid_hi)

    # Neck: nose → midpoint(shoulders) vs vertical
    if _ok(scores, [_NOSE], min_conf) and sh_ok:
        angles["neck_deg"] = _angle_to_vertical(coords[_NOSE], mid_sh)

    # Pelvis tilt: midpoint(hips) to midpoint(shoulders) vs horizontal
    if sh_ok and hi_ok:
        angles["pelvis_deg"] = _angle_to_horizontal(mid_hi, mid_sh)

    # Shoulder line tilt
    if sh_ok:
        angles["shoulder_line_deg"] = _angle_to_horizontal(coords[_LSH], coords[_RSH])

    # Hip line tilt
    if hi_ok:
        angles["hip_line_deg"] = _angle_to_horizontal(coords[_LHI], coords[_RHI])

    # ── Elevation / abduction angles ─────────────────────────────────────
    # Arm raise: angle between torso axis and shoulder→elbow
    if sh_ok and hi_ok:
        torso_dir = mid_hi - mid_sh  # pointing downward along torso

        if _ok(scores, [_LSH, _LEL], min_conf):
            arm_dir = coords[_LEL] - coords[_LSH]
            angles["left_arm_raise_deg"] = _vec_angle(torso_dir, arm_dir)

        if _ok(scores, [_RSH, _REL], min_conf):
            arm_dir = coords[_REL] - coords[_RSH]
            angles["right_arm_raise_deg"] = _vec_angle(torso_dir, arm_dir)

        if _ok(scores, [_LHI, _LKN], min_conf):
            leg_dir = coords[_LKN] - coords[_LHI]
            angles["left_leg_abduction_deg"] = _vec_angle(torso_dir, leg_dir)

        if _ok(scores, [_RHI, _RKN], min_conf):
            leg_dir = coords[_RKN] - coords[_RHI]
            angles["right_leg_abduction_deg"] = _vec_angle(torso_dir, leg_dir)

    # Round all non-None values for CSV readability
    return {k: round(v, 2) if v is not None else None for k, v in angles.items()}


def _vec_angle(u: np.ndarray, v: np.ndarray) -> float:
    """Angle (degrees) between two 2-D vectors."""
    cos_val = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))
