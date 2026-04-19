#!/usr/bin/env python3
"""
training/scripts/clean_training_csv.py

Reutilizable script to clean and transform a consolidated training CSV
(potentially from multiple videos) into a reference training CSV with
normalized labels.

Usage:
    python training/scripts/clean_training_csv.py \
        --input_csv  training/data/train_squats.csv \
        --output_csv training/data/train_squats_cleaned.csv \
        --exercise_label squat \
        [--target_id 1] \
        [--phase_algo derivative|legacy]

Operations:
  1. (Optional) Filter rows to a single track_id (target_id).
  2. Drop columns: is_valid_pose, activity_label.
  3. Normalize exercise_label using new schema.
  4. Auto-compute phase_label and rep_id from biomechanical signals,
     GROUPED BY video_id to avoid temporal contamination between videos.
  5. Save the cleaned CSV to output_csv.

Phase detection (squat):
  Two algorithms available (--phase_algo):
    - derivative: enhanced detection using derivatives, peak detection, adaptive thresholds
    - legacy: original finite-state-machine with temporal smoothing and hysteresis
  Classifies each frame as idle / down / up (normalized to phase_1, phase_2, none).

Label normalization:
  - Exercises are normalized to canonical names (e.g., squat → squat)
  - Phases are semantic per exercise (e.g., up/down for squat, open/closed for jumping_jacks)
  - activity_label column is removed (not included in output)
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# ── Label normalization (shared) ───────────────────────────────────────────
from label_normalizer import (
    normalize_exercise_label,
    normalize_phase_label,
)

# ── Canonical schema ────────────────────────────────────────────────────────
try:
    from src.core.types import FrameRecord
    CANONICAL_HEADER = FrameRecord.csv_header()
except ImportError as e:
    print(f"⚠️  Warning: Could not import FrameRecord: {e}")
    CANONICAL_HEADER = None

# ── Constants ──────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "track_id",
    "frame_number",
    "knee_left_deg",
    "knee_right_deg",
    "hip_center_y_norm",
]

COLUMNS_TO_DROP = ["is_valid_pose", "activity_label", "activitylabel", "activity"]

# ── Column name normalization ──────────────────────────────────────────────
# Maps alternative column names to canonical names (snake_case)
COLUMN_NAME_ALIASES = {
    "exerciselabel": "exercise_label",
    "phaselabel": "phase_label",
    "repid": "rep_id",
    "trackid": "track_id",
    "framenumber": "frame_number",
    "sessionid": "session_id",
    "videoid": "video_id",
}

# Expected label columns in final CSV (activity_label removed)
LABEL_COLUMNS = ["exercise_label", "phase_label", "rep_id"]

# ── Squat-specific thresholds ──────────────────────────────────────────────
# Nota: En coordenadas de imagen, Y=0 es la parte superior, Y=1 es abajo.
# Al bajar (squat), Y aumenta -> Velocidad (Vy) es POSITIVA.
# Al subir, Y disminuye -> Velocidad (Vy) es NEGATIVA.

SMOOTHING_WINDOW = 7                  # Ventana para suavizado de señal primaria
SMOOTHING_WINDOW_DERIV = 3            # Ventana para suavizado de derivadas (menor → menos lag)
CONSISTENCY_FRAMES_BASE = 3           # Frames base para histéresis (se escala con FPS)
CONSISTENCY_MIN_MS = 100              # Ventana mínima de histéresis en milisegundos

KNEE_ANGLE_STANDING = 160.0           # > 160° consideramos que está de pie (relajado)
KNEE_ANGLE_BENT = 120.0               # < 120° consideramos flexión clara (squat profundo ~90°)

HIP_VY_MOTION_THRESH = 0.0005         # Magnitud mínima de velocidad vertical normalizada (más sensible)
KNEE_VY_LEGACY_THRESH = 0.3           # deg/frame mínimo para confirmar extensión en legacy


# ── Enhanced derivative‑based detection ─────────────────────────────────────

# Signal processing
PEAK_DETECTION_WINDOW = 5              # Window for local peak detection

# Thresholds (initial values, adapted later)
KNEE_VY_THRESH = 0.5                   # deg/frame knee velocity threshold for motion
KNEE_ACCEL_THRESH = 0.3                # deg/frame² knee acceleration threshold
HIP_VY_THRESH = 0.0005                 # normalized units/frame hip velocity threshold (same as HIP_VY_MOTION_THRESH)
HIP_ACCEL_THRESH = 0.0001              # normalized units/frame² hip acceleration threshold

# Idle detection
KNEE_ANGLE_RELAXED = 150.0             # Considered standing (relaxed)
KNEE_VY_IDLE_THRESH = 0.1              # deg/frame max knee velocity for idle
HIP_VY_IDLE_THRESH = 0.0002            # normalized units/frame max hip velocity for idle
MIN_IDLE_FRAMES = 3                    # Minimum consecutive frames to confirm idle

# Adaptive parameters
ADAPTIVE_WINDOW = 30                   # Frames for computing signal statistics
VELOCITY_RATIO = 0.3                   # Ratio of peak velocity for motion threshold
IDLE_RATIO = 0.1                       # Ratio of peak velocity for idle threshold


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean a consolidated training CSV: filter subject, "
                    "set labels, auto-compute phase_label per video.",
    )
    p.add_argument(
        "--input_csv", required=True,
        help="Path to the input consolidated CSV (may contain multiple videos).",
    )
    p.add_argument(
        "--output_csv", required=True,
        help="Path for the output cleaned CSV.",
    )
    p.add_argument(
        "--target_id", type=int, default=None,
        help="(Optional) track_id to keep; all other subjects are removed. "
             "If omitted, all track_ids are kept (one per video is expected).",
    )
    p.add_argument(
        "--exercise_label", required=True,
        help="Value for the exercise_label column (e.g. 'squat').",
    )
    p.add_argument(
        "--phase_algo", choices=["derivative", "legacy"], default="derivative",
        help="Phase detection algorithm: derivative (new) or legacy (original)",
    )
    return p.parse_args()


# ── Validation ─────────────────────────────────────────────────────────────

def validate_columns(df: pd.DataFrame) -> None:
    """Verify that all required columns exist in the DataFrame."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"❌ ERROR: Columnas requeridas ausentes: {missing}")
        print(f"   Columnas disponibles: {list(df.columns)}")
        sys.exit(1)


def validate_target_id(df: pd.DataFrame, target_id: int) -> None:
    """Ensure that the target_id exists within the CSV."""
    unique_ids = df["track_id"].unique()
    if target_id not in unique_ids:
        print(f"❌ ERROR: track_id={target_id} no existe en el CSV.")
        print(f"   IDs disponibles: {sorted(unique_ids)}")
        sys.exit(1)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns using aliases to canonical snake_case names.
    Returns a copy with normalized column names.
    """
    df = df.copy()
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in COLUMN_NAME_ALIASES:
            rename_map[col] = COLUMN_NAME_ALIASES[col_lower]
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"🔠 Normalized column names: {rename_map}")
    return df


def ensure_activity_label(df: pd.DataFrame, fallback_exercise: str) -> pd.DataFrame:
    """
    Ensure exercise_label column exists with correct values.
    Priority:
      1. If exercise_label already exists, keep it (will be normalized later).
      2. If activity_label exists, infer exercise_label from it (normalized).
      3. Otherwise, use fallback_exercise as exercise_label (normalized).

    Note: activity_label column is dropped later via COLUMNS_TO_DROP.
    """
    df = df.copy()

    # Rename activity_label aliases for easier inference
    if "activity_label" not in df.columns:
        for alt in ["activitylabel", "activity"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "activity_label"})
                print(f"🔠 Renamed column '{alt}' → 'activity_label'")
                break

    # 1. Normalize column names (only exercise_label aliases)
    if "exercise_label" not in df.columns:
        # Try alternative names for exercise_label
        for alt in ["exerciselabel", "exercise"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "exercise_label"})
                print(f"🔠 Renamed column '{alt}' → 'exercise_label'")
                break

    # 2. If exercise_label already exists, keep it (will be normalized later)
    if "exercise_label" in df.columns:
        original_counts = df["exercise_label"].value_counts()
        print(f"🏷️  exercise_label already present: {dict(original_counts)}")
        return df

    # 3. exercise_label missing, try to infer from activity_label if present
    if "activity_label" in df.columns:
        # Infer exercise_label from activity_label values
        # If activity_label is "none" or similar, set exercise_label to "none"
        # Otherwise, normalize the activity_label value to canonical form
        def infer_exercise_from_activity(act):
            if isinstance(act, str):
                act_lower = act.strip().lower()
                if act_lower == "none" or act_lower == "no_exercise" or act_lower == "idle":
                    return "none"
                # For any other value, treat as exercise name and normalize
                return normalize_exercise_label(act)
            # Default for non-string
            return "none"

        df["exercise_label"] = df["activity_label"].apply(infer_exercise_from_activity)
        print(f"🏷️  Inferred exercise_label from activity_label column")
        return df

    # 4. No exercise_label or activity_label, use fallback
    normalized_fallback = normalize_exercise_label(fallback_exercise)
    df["exercise_label"] = normalized_fallback
    print(f"🏷️  Using fallback exercise label '{fallback_exercise}' → '{normalized_fallback}'")

    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns to match canonical schema, keeping extra columns at the end.
    """
    if CANONICAL_HEADER is None:
        print("⚠️  Cannot reorder columns: canonical header not available")
        return df

    # Identify columns present in both canonical header and dataframe
    canonical_cols = [col for col in CANONICAL_HEADER if col in df.columns]
    extra_cols = [col for col in df.columns if col not in CANONICAL_HEADER]

    # Reorder: canonical columns first, then extra columns
    new_order = canonical_cols + extra_cols
    if list(df.columns) != new_order:
        df = df[new_order]
        print(f"📋 Reordered columns: {len(canonical_cols)} canonical + {len(extra_cols)} extra")

    return df


# ── Filtering & column operations ──────────────────────────────────────────

def filter_single_subject(df: pd.DataFrame, target_id: int) -> pd.DataFrame:
    """Keep only rows matching the target track_id."""
    filtered = df[df["track_id"] == target_id].copy()
    filtered = filtered.sort_values("frame_number").reset_index(drop=True)
    print(f"✅ Filtrado a track_id={target_id}: {len(filtered)} filas "
          f"(de {len(df)} originales)")
    return filtered


def drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that shouldn't be in the training output."""
    dropped = []
    for col in COLUMNS_TO_DROP:
        if col in df.columns:
            df = df.drop(columns=[col])
            dropped.append(col)
    if dropped:
        print(f"🗑️  Columnas eliminadas: {dropped}")
    return df




# ── Phase detection (squat) ────────────────────────────────────────────────

def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    """Apply a simple moving average with edge padding for noise reduction."""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(values)]


def compute_squat_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes smoothed signals and velocity independently from the tracker.
    This guarantees robust, zero-centered velocity even if tracking was noisy.

    Fixes applied (audit 2026-04):
      P1: Interpolation instead of fillna(180) to avoid false spikes.
      P5: Use min(L, R) knee angle when both valid; single-side fallback.
      P7: Reduced derivative smoothing window to cut detection lag.
      FPS normalization: Scale velocities to 30fps-equivalent.
    """
    # ── P1: Interpolate missing values instead of hard fill ────────────
    knee_l_series = pd.to_numeric(df["knee_left_deg"], errors="coerce")
    knee_r_series = pd.to_numeric(df["knee_right_deg"], errors="coerce")
    hip_y_series = pd.to_numeric(df["hip_center_y_norm"], errors="coerce")

    # Track original validity for L/R knee fusion (P5)
    knee_l_valid = knee_l_series.notna().values
    knee_r_valid = knee_r_series.notna().values

    # Interpolate short gaps (≤5 frames), then forward/backward fill edges
    knee_l = knee_l_series.interpolate(limit=5).ffill().bfill().fillna(180.0).values
    knee_r = knee_r_series.interpolate(limit=5).ffill().bfill().fillna(180.0).values
    hip_y = hip_y_series.interpolate(limit=5).ffill().bfill().fillna(0.5).values

    # ── P5: Intelligent L/R knee fusion ───────────────────────────────
    # When both sides are valid: use the more flexed (lower angle) for
    # conservative squat detection. Single-side fallback when one is missing.
    avg_knee_raw = np.where(
        knee_l_valid & knee_r_valid,
        np.minimum(knee_l, knee_r),
        np.where(knee_l_valid, knee_l,
                 np.where(knee_r_valid, knee_r, 180.0))
    )

    # ── FPS normalization for velocities ──────────────────────────────
    fps_factor = 1.0
    if "fps" in df.columns:
        fps_val = pd.to_numeric(df["fps"], errors="coerce").median()
        if fps_val and fps_val > 0:
            fps_factor = 30.0 / fps_val  # Normalize to 30fps equivalent
            if abs(fps_factor - 1.0) > 0.05:
                print(f"📐 FPS normalization: {fps_val:.1f}fps → factor={fps_factor:.3f}")

    # Suavizado para resistir fallos de un frame
    avg_knee_s = _smooth_series(avg_knee_raw, SMOOTHING_WINDOW)
    hip_y_s = _smooth_series(hip_y, SMOOTHING_WINDOW)

    # ── P7: Reduced derivative smoothing (SMOOTHING_WINDOW_DERIV) ─────
    # Calcular derivada temporal robusta (velocity), scaled by fps_factor
    hip_vy_raw = np.zeros_like(hip_y_s)
    if len(hip_y_s) > 1:
        hip_vy_raw[1:] = (hip_y_s[1:] - hip_y_s[:-1]) * fps_factor

    hip_vy_s = _smooth_series(hip_vy_raw, SMOOTHING_WINDOW_DERIV)

    # Calcular velocidad del ángulo de rodilla (derivada)
    knee_vy_raw = np.zeros_like(avg_knee_s)
    if len(avg_knee_s) > 1:
        knee_vy_raw[1:] = (avg_knee_s[1:] - avg_knee_s[:-1]) * fps_factor

    knee_vy_s = _smooth_series(knee_vy_raw, SMOOTHING_WINDOW_DERIV)

    # Compute acceleration (second derivative)
    hip_ay_raw = np.zeros_like(hip_vy_s)
    if len(hip_vy_s) > 1:
        hip_ay_raw[1:] = (hip_vy_s[1:] - hip_vy_s[:-1]) * fps_factor
    hip_ay_s = _smooth_series(hip_ay_raw, SMOOTHING_WINDOW_DERIV)

    knee_ay_raw = np.zeros_like(knee_vy_s)
    if len(knee_vy_s) > 1:
        knee_ay_raw[1:] = (knee_vy_s[1:] - knee_vy_s[:-1]) * fps_factor
    knee_ay_s = _smooth_series(knee_ay_raw, SMOOTHING_WINDOW_DERIV)

    # Inyectar variables de depuración (útil para validar el etiquetado)
    df["debug_knee_smooth"] = avg_knee_s
    df["debug_hip_y_smooth"] = hip_y_s
    df["debug_hip_vy_smooth"] = hip_vy_s
    df["debug_knee_vy_smooth"] = knee_vy_s
    df["debug_hip_ay_smooth"] = hip_ay_s
    df["debug_knee_ay_smooth"] = knee_ay_s

    return df


def _compute_consistency_frames(df: pd.DataFrame) -> int:
    """P6: Compute FPS-aware consistency frames for hysteresis."""
    if "fps" in df.columns:
        fps_val = pd.to_numeric(df["fps"], errors="coerce").median()
        if fps_val and fps_val > 0:
            frames = max(CONSISTENCY_FRAMES_BASE, int(fps_val * CONSISTENCY_MIN_MS / 1000))
            return frames
    return CONSISTENCY_FRAMES_BASE


def detect_phase_with_hysteresis(df: pd.DataFrame, exercise_label: str, debug: bool = True) -> Tuple[List[str], List[int]]:
    """
    State machine con histéresis temporal.
    Requiere que una condición se mantenga por consistency_frames para transicionar,
    evitando etiquetas saltarinas ("flickering").

    Raw phase states are "idle", "down", "up", then normalized to the canonical set
    for the exercise (e.g., squat uses "up", "down", "none").

    Audit fixes (2026-04):
      P2: Min depth validation — requires knee < KNEE_ANGLE_BENT before confirming rep.
      P3: kv threshold — uses KNEE_VY_LEGACY_THRESH instead of kv > 0.
      P6: FPS-aware consistency frames.

    Args:
        df: DataFrame with computed biomechanical signals
        exercise_label: canonical exercise name (for phase normalization)
    """
    n = len(df)
    if n == 0:
        return [], []

    avg_knee = df["debug_knee_smooth"].values
    hip_vy = df["debug_hip_vy_smooth"].values
    knee_vy = df["debug_knee_vy_smooth"].values

    # P6: FPS-aware consistency frames
    consistency_frames = _compute_consistency_frames(df)
    if debug:
        print(f"🔧 Legacy: consistency_frames={consistency_frames}")

    phases: List[str] = []
    rep_ids: List[int] = []

    # Estado inicial: asumimos 'idle' (el sujeto suele empezar de pie)
    current_state = "idle"
    current_rep_id = 0

    # Variables para histéresis
    candidate_state = current_state
    consistency_count = 0

    # P2: Track minimum squat depth
    reached_min_depth = False

    for i in range(n):
        k = avg_knee[i]
        vy = hip_vy[i]

        # 1) Determinar el estado propuesto basado puramente en este frame
        prop_state = current_state

        kv = knee_vy[i] if i < len(knee_vy) else 0.0

        if current_state == "idle":
            # Para salir de idle: rodilla se dobla y velocidad es claramente -> DOWN
            if k < KNEE_ANGLE_STANDING and vy > HIP_VY_MOTION_THRESH:
                prop_state = "down"
                if debug:
                    print(f"  frame {i}: idle → down (k={k:.1f}, vy={vy:.6f})")

        elif current_state == "down":
            # P2: Track if minimum squat depth was reached
            if k < KNEE_ANGLE_BENT:
                reached_min_depth = True

            # P3: Para pasar a up: dirección se invierte Y rodilla se extiende con umbral real
            if vy < -HIP_VY_MOTION_THRESH and kv > KNEE_VY_LEGACY_THRESH:
                if reached_min_depth:
                    prop_state = "up"
                    if debug:
                        print(f"  frame {i}: down → up (k={k:.1f}, vy={vy:.6f}, kv={kv:.6f})")
                else:
                    # Not a real squat — revert to idle
                    prop_state = "idle"
                    if debug:
                        print(f"  frame {i}: down → idle (no min depth, k={k:.1f})")
            elif debug and vy < -HIP_VY_MOTION_THRESH:
                print(f"  frame {i}: down → up REJECTED (kv={kv:.4f} < {KNEE_VY_LEGACY_THRESH})")

        elif current_state == "up":
            # Para volver a idle: subida se detiene y rodilla está casi recta
            if k > KNEE_ANGLE_STANDING and abs(vy) < HIP_VY_MOTION_THRESH:
                prop_state = "idle"
                if debug:
                    print(f"  frame {i}: up → idle (k={k:.1f}, vy={vy:.6f})")
            # Fallback de seguridad: si empieza a bajar abruptamente antes de estirarse del todo
            elif k < KNEE_ANGLE_BENT and vy > HIP_VY_MOTION_THRESH:
                prop_state = "down"
                if debug:
                    print(f"  frame {i}: up → down (k={k:.1f}, vy={vy:.6f})")

        # 2) Manejo temporal de la histéresis
        if prop_state != current_state:
            if prop_state == candidate_state:
                consistency_count += 1
                if consistency_count >= consistency_frames:
                    # Transición confirmada
                    # Incrementar rep si pasamos de algo distinto a down, a down
                    if prop_state == "down" and current_state in ["idle", "up"]:
                        current_rep_id += 1
                        reached_min_depth = False  # P2: Reset for new rep
                    elif prop_state == "idle":
                        reached_min_depth = False  # P2: Reset on idle

                    current_state = prop_state
                    candidate_state = current_state
                    consistency_count = 0
            else:
                candidate_state = prop_state
                consistency_count = 1
        else:
            candidate_state = current_state
            consistency_count = 0

        # En retrocesos, el frame actual se graba con el current_state confirmado
        phases.append(current_state)
        rep_ids.append(current_rep_id)

    # Normalizar fases según el ejercicio
    normalized_phases = [
        normalize_phase_label(exercise_label, phase) for phase in phases
    ]

    return normalized_phases, rep_ids


def detect_phase_derivative_based(df: pd.DataFrame, exercise_label: str, debug: bool = True) -> Tuple[List[str], List[int]]:
    """
    Enhanced phase detection using derivatives, peak detection, and adaptive thresholds.

    States: idle → descent → ascent → idle
    Raw states: "idle", "down", "up" (normalized later).

    Audit fixes (2026-04):
      P2: Min depth validation — requires knee < KNEE_ANGLE_BENT to confirm rep.
      P4: Adaptive thresholds recalculated per sliding window (not once globally).
      P6: FPS-aware consistency frames.

    Args:
        df: DataFrame with computed biomechanical signals (including debug columns)
        exercise_label: canonical exercise name (for phase normalization)
        debug: print transition details

    Returns:
        phases list, rep_ids list
    """
    n = len(df)
    if n == 0:
        return [], []

    avg_knee = df["debug_knee_smooth"].values
    hip_y = df["debug_hip_y_smooth"].values
    hip_vy = df["debug_hip_vy_smooth"].values
    knee_vy = df["debug_knee_vy_smooth"].values
    hip_ay = df["debug_hip_ay_smooth"].values
    knee_ay = df["debug_knee_ay_smooth"].values

    # P4: Sliding-window adaptive threshold function
    def compute_adaptive_thresholds_at(signal, idx, window=ADAPTIVE_WINDOW,
                                       motion_ratio=VELOCITY_RATIO, idle_ratio=IDLE_RATIO):
        """Compute thresholds based on a sliding window ending at idx."""
        start = max(0, idx - window)
        window_slice = signal[start:idx + 1]
        if len(window_slice) == 0:
            peak = 1.0
        else:
            peak = np.max(np.abs(window_slice))
        motion_thresh = max(HIP_VY_THRESH, peak * motion_ratio)
        idle_thresh = max(HIP_VY_IDLE_THRESH, peak * idle_ratio)
        return motion_thresh, idle_thresh

    # P6: FPS-aware consistency frames
    consistency_frames = _compute_consistency_frames(df)

    # Initial thresholds (will be recalculated per window)
    hip_vy_motion_thresh, hip_vy_idle_thresh = compute_adaptive_thresholds_at(hip_vy, 0)
    knee_vy_motion_thresh = KNEE_VY_THRESH
    knee_vy_idle_thresh = KNEE_VY_IDLE_THRESH

    if debug:
        print(f"🔧 Derivative: consistency_frames={consistency_frames}")
        print(f"🔧 Initial thresholds: hip_vy_motion={hip_vy_motion_thresh:.6f}, hip_vy_idle={hip_vy_idle_thresh:.6f}")

    phases: List[str] = []
    rep_ids: List[int] = []

    current_state = "idle"
    current_rep_id = 0

    # Hysteresis variables
    candidate_state = current_state
    consistency_count = 0

    # For peak detection: track previous hip_vy sign
    prev_hip_vy_sign = 0  # 0 unknown, 1 positive, -1 negative
    peak_detected = False

    # For idle detection: count consecutive low-motion frames
    idle_counter = 0

    # P2: Track minimum squat depth
    reached_min_depth = False

    for i in range(n):
        k = avg_knee[i]
        vy = hip_vy[i]
        kv = knee_vy[i]
        ay = hip_ay[i]
        ka = knee_ay[i]

        # P4: Recalculate adaptive thresholds every ADAPTIVE_WINDOW frames
        if i % ADAPTIVE_WINDOW == 0:
            hip_vy_motion_thresh, hip_vy_idle_thresh = compute_adaptive_thresholds_at(hip_vy, i)
            knee_window = knee_vy[max(0, i - ADAPTIVE_WINDOW):i + 1]
            if len(knee_window) > 0:
                knee_peak = np.max(np.abs(knee_window))
                knee_vy_motion_thresh = max(KNEE_VY_THRESH, knee_peak * VELOCITY_RATIO)
                knee_vy_idle_thresh = max(KNEE_VY_IDLE_THRESH, knee_peak * IDLE_RATIO)

        # Determine proposed state based on current frame
        prop_state = current_state

        # Peak detection: hip Y maximum (vy changes from positive to negative)
        if prev_hip_vy_sign == 1 and vy < -hip_vy_motion_thresh:
            peak_detected = True
            if debug:
                print(f"  frame {i}: PEAK detected (hip Y max)")
        prev_hip_vy_sign = 1 if vy > hip_vy_motion_thresh else (-1 if vy < -hip_vy_motion_thresh else prev_hip_vy_sign)

        # State transitions
        if current_state == "idle":
            # Start descent: knee flexing (negative knee velocity) AND hip lowering (positive hip velocity)
            # OR knee angle already bent below relaxed threshold with positive hip acceleration
            if (kv < -knee_vy_motion_thresh and vy > hip_vy_motion_thresh) or \
               (k < KNEE_ANGLE_RELAXED and ay > HIP_ACCEL_THRESH):
                prop_state = "down"
                if debug:
                    print(f"  frame {i}: idle → down (k={k:.1f}, kv={kv:.3f}, vy={vy:.6f})")

        elif current_state == "down":
            # P2: Track if minimum squat depth was reached
            if k < KNEE_ANGLE_BENT:
                reached_min_depth = True

            # Transition to ascent when peak detected AND knee starts extending
            if peak_detected and kv > knee_vy_motion_thresh:
                if reached_min_depth:
                    prop_state = "up"
                    peak_detected = False
                    if debug:
                        print(f"  frame {i}: down → up (peak + knee extending)")
                else:
                    # P2: Not a real squat — revert to idle
                    prop_state = "idle"
                    peak_detected = False
                    if debug:
                        print(f"  frame {i}: down → idle (no min depth, k={k:.1f})")
            # Safety: if hip starts rising before peak but knee extends
            elif vy < -hip_vy_motion_thresh and kv > knee_vy_motion_thresh:
                if reached_min_depth:
                    prop_state = "up"
                    if debug:
                        print(f"  frame {i}: down → up (early ascent)")
                else:
                    prop_state = "idle"
                    if debug:
                        print(f"  frame {i}: down → idle (early ascent, no min depth, k={k:.1f})")

        elif current_state == "up":
            # Check for idle: low velocities and knee near relaxed angle
            if abs(vy) < hip_vy_idle_thresh and abs(kv) < knee_vy_idle_thresh and k > KNEE_ANGLE_RELAXED:
                idle_counter += 1
                if idle_counter >= MIN_IDLE_FRAMES:
                    prop_state = "idle"
                    if debug:
                        print(f"  frame {i}: up → idle (low motion for {idle_counter} frames)")
            else:
                idle_counter = 0

            # Direct transition to next descent if descent starts before full idle
            if kv < -knee_vy_motion_thresh and vy > hip_vy_motion_thresh and k < KNEE_ANGLE_RELAXED:
                prop_state = "down"
                idle_counter = 0
                if debug:
                    print(f"  frame {i}: up → down (next rep without full idle)")

        # Hysteresis handling
        if prop_state != current_state:
            if prop_state == candidate_state:
                consistency_count += 1
                if consistency_count >= consistency_frames:
                    # Transition confirmed
                    # Increment rep when moving to down from idle or up
                    if prop_state == "down" and current_state in ["idle", "up"]:
                        current_rep_id += 1
                        reached_min_depth = False  # P2: Reset for new rep
                    elif prop_state == "idle":
                        reached_min_depth = False  # P2: Reset on idle

                    current_state = prop_state
                    candidate_state = current_state
                    consistency_count = 0
                    idle_counter = 0  # reset idle counter on state change
            else:
                candidate_state = prop_state
                consistency_count = 1
        else:
            candidate_state = current_state
            consistency_count = 0

        phases.append(current_state)
        rep_ids.append(current_rep_id)

    # Normalize phases according to exercise
    normalized_phases = [
        normalize_phase_label(exercise_label, phase) for phase in phases
    ]

    return normalized_phases, rep_ids


def assign_phase_and_rep_labels(df: pd.DataFrame, exercise_label: str, phase_algo: str = "derivative") -> pd.DataFrame:
    """
    Compute biomechanical features and assign phase_label and rep_id columns.

    Args:
        df: DataFrame with biomechanical columns
        exercise_label: canonical exercise name (for phase validation)
        phase_algo: "derivative" (new) or "legacy" (original)

    Returns:
        DataFrame with computed phase_label and rep_id
    """
    df = compute_squat_features(df)

    if phase_algo == "derivative":
        phases, rep_ids = detect_phase_derivative_based(df, exercise_label, debug=True)
    elif phase_algo == "legacy":
        phases, rep_ids = detect_phase_with_hysteresis(df, exercise_label, debug=True)
    else:
        raise ValueError(f"Unknown phase_algo: {phase_algo}")

    df["phase_label"] = phases
    df["rep_id"] = rep_ids

    # Summary
    counts = Counter(phases)
    print(f"🔄 phase_label auto-calculado y normalizado (algoritmo: {phase_algo}):")
    for phase, count in counts.most_common():
        print(f"   {phase}: {count} frames")
    print(f"🔄 Total de repeticiones encontradas: {max(rep_ids) if rep_ids else 0}")

    return df


def adjust_exercise_label_based_on_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust exercise_label based on phase_label.

    Simple rule: if phase_label == "none", then exercise_label must also be "none".
    This ensures that when the person is standing idle (not flexing knees),
    the exercise label reflects "none" instead of the exercise name.

    This applies to:
      - Before starting the exercise
      - Between repetitions when standing upright
      - After finishing the exercise
    """
    df = df.copy()

    # Mask for rows where phase_label is "none" but exercise_label is not "none"
    mask_none_phase = df["phase_label"] == "none"
    mask_not_none_exercise = df["exercise_label"].fillna("").ne("none")
    mask_to_adjust = mask_none_phase & mask_not_none_exercise

    if mask_to_adjust.any():
        count = mask_to_adjust.sum()
        df.loc[mask_to_adjust, "exercise_label"] = "none"
        print(f"🔄 Adjusted exercise_label to 'none' for {count} frames with phase_label='none'")

    return df


def validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate label consistency and generate a report.

    Rules:
    1. If exercise_label == "none":
        - phase_label must be "none"
    2. If exercise_label != "none":
        - phase_label must be in VALID_PHASES ("phase_1", "phase_2", or "none")
    3. exercise_label is canonical (already normalized)

    Returns the DataFrame with corrected inconsistencies (if any).
    """
    from label_normalizer import VALID_PHASES, normalize_exercise_label

    df = df.copy()
    inconsistencies = []

    # Ensure exercise_label is canonical (should already be normalized)
    df["exercise_label"] = df["exercise_label"].apply(lambda x: normalize_exercise_label(x) if isinstance(x, str) else "none")

    for idx, row in df.iterrows():
        ex = row["exercise_label"]
        phase = row["phase_label"]

        # Rule 1: exercise_label == "none"
        if ex == "none":
            if phase != "none":
                inconsistencies.append((idx, f"exercise_label='none' but phase_label='{phase}' (should be 'none')"))
                df.at[idx, "phase_label"] = "none"
        else:
            # Rule 2: exercise_label != "none"
            if phase not in VALID_PHASES:
                inconsistencies.append((idx, f"phase_label='{phase}' invalid, setting to 'none'"))
                df.at[idx, "phase_label"] = "none"
            elif phase not in ("phase_1", "phase_2", "none"):
                # Should be phase_1, phase_2, or none (already covered by VALID_PHASES)
                pass

    if inconsistencies:
        print("\n⚠️  LABEL INCONSISTENCIES DETECTED:")
        for idx, msg in inconsistencies[:20]:  # Limit output
            print(f"   Row {idx}: {msg}")
        if len(inconsistencies) > 20:
            print(f"   ... and {len(inconsistencies) - 20} more")
        print(f"   Total inconsistencies: {len(inconsistencies)}")
        # Automatically corrected
    else:
        print("✅ All label consistency checks passed.")

    return df


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Validate input file ────────────────────────────────────────────
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"❌ ERROR: Archivo no encontrado: {input_path}")
        sys.exit(1)

    # ── Read CSV ───────────────────────────────────────────────────────
    print(f"📂 Leyendo: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   → {len(df)} filas, {len(df.columns)} columnas")

    # ── Validate schema ────────────────────────────────────────────────
    validate_columns(df)

    # ── Optional: filter to a single track_id ──────────────────────────
    if args.target_id is not None:
        validate_target_id(df, args.target_id)
        df = filter_single_subject(df, args.target_id)
    else:
        print(f"👤 target_id no especificado — manteniendo todos los track_ids.")
        print(f"   IDs presentes: {sorted(df['track_id'].unique())}")

    # ── Normalize columns ──────────────────────────────────────────────
    df = normalize_column_names(df)
    df = ensure_activity_label(df, args.exercise_label)
    df = drop_unwanted_columns(df)

    # Determine canonical exercise label
    unique_exercises = df["exercise_label"].unique()
    if len(unique_exercises) == 1:
        normalized_exercise = unique_exercises[0]
    else:
        normalized_exercise = normalize_exercise_label(args.exercise_label)
        print(f"⚠️  Multiple exercise labels found: {unique_exercises}, using fallback '{normalized_exercise}'")

    # ── Phase detection grouped by video_id ────────────────────────────
    # CRITICAL: signals from different videos must NOT be mixed.
    # Velocities, derivatives and smoothing are all temporal —
    # the boundary between two videos is not a continuous sequence.
    if "video_id" in df.columns and df["video_id"].nunique() > 1:
        video_ids = df["video_id"].unique()
        print(f"🎥 Procesando fases por video ({len(video_ids)} vídeos): {list(video_ids)}")
        processed_chunks = []
        for vid in video_ids:
            chunk = df[df["video_id"] == vid].copy().reset_index(drop=True)
            print(f"   ▶ {vid}: {len(chunk)} frames")
            chunk = assign_phase_and_rep_labels(chunk, normalized_exercise, phase_algo=args.phase_algo)
            chunk = adjust_exercise_label_based_on_phase(chunk)
            chunk = validate_labels(chunk)
            processed_chunks.append(chunk)
        df = pd.concat(processed_chunks, ignore_index=True)
    else:
        # Single video or no video_id column: original behaviour
        df = assign_phase_and_rep_labels(df, normalized_exercise, phase_algo=args.phase_algo)
        df = adjust_exercise_label_based_on_phase(df)
        df = validate_labels(df)

    # ── Reorder and final validation ──────────────────────────────────
    df = reorder_columns(df)

    assert "is_valid_pose" not in df.columns, "is_valid_pose should be removed"
    assert "rep_id" in df.columns, "rep_id should be present"
    assert "exercise_label" in df.columns, "exercise_label should be present"
    assert "phase_label" in df.columns, "phase_label should be present"
    assert "activity_label" not in df.columns, "activity_label should be removed"

    from label_normalizer import VALID_PHASES
    actual_phases = set(df["phase_label"].unique())
    assert actual_phases.issubset(VALID_PHASES), \
        f"Unexpected phases: {actual_phases}. Expected subset of: {VALID_PHASES}"

    # ── Save ───────────────────────────────────────────────────────────
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    n_videos = df["video_id"].nunique() if "video_id" in df.columns else 1
    print(f"\n{'='*60}")
    print(f"✅ CSV limpio guardado: {output_path}")
    print(f"   → {len(df)} filas, {len(df.columns)} columnas")
    print(f"   → vídeos procesados: {n_videos}")
    print(f"   → exercise_label: {df['exercise_label'].iloc[0]} (input: '{args.exercise_label}')")
    print(f"   → phase_label distribution: {dict(df['phase_label'].value_counts())}")
    print(f"   → total reps: {df['rep_id'].max()}")
    print(f"   (Las columnas 'debug_*' se mantienen para análisis extra).")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
