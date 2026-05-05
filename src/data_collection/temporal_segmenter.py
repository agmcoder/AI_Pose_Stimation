"""
src/data_collection/temporal_segmenter.py

Offline post-processing module for temporal segmentation of session CSVs.

Provides a complete pipeline to re-label an existing frames.csv with
proper temporal segmentation based on:
  1. Sliding-window confidence aggregation
  2. Frame-level smoothing (median / exponential)
  3. Hysteresis-based segment detection (enter/exit thresholds)
  4. State machine for robust start/end detection
  5. Minimum segment length enforcement
  6. Adjacent segment merging
  7. Rep counting within validated segments via phase transitions

Usage (standalone):
    python -m src.data_collection.temporal_segmenter \\
        --input data/sessions/2026-05-04_19-07-40/frames.csv \\
        --output data/sessions/2026-05-04_19-07-40/frames_segmented.csv

Usage (library):
    from src.data_collection.temporal_segmenter import segment_session_csv
    df = segment_session_csv("path/to/frames.csv", config={...})
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SegmentationConfig:
    """All tunable parameters for temporal segmentation."""

    # ── Smoothing ─────────────────────────────────────────────────────────
    smoothing_method: str = "median"      # "median", "ema", "mean"
    smoothing_window: int = 7             # window size for median/mean filter
    ema_alpha: float = 0.3                # alpha for exponential smoothing

    # ── Hysteresis thresholds ─────────────────────────────────────────────
    enter_threshold: float = 0.5          # confidence to enter exercise
    exit_threshold: float = 0.3           # confidence to exit exercise

    # ── Temporal constraints ──────────────────────────────────────────────
    confirm_frames: int = 12              # consecutive frames to confirm start
    cooldown_frames: int = 8              # consecutive frames to confirm end
    min_segment_frames: int = 8           # minimum frames for a valid segment
    merge_gap_frames: int = 5             # merge segments closer than this

    # ── Knee angle thresholds for rep counting ─────────────────────────────
    knee_down_max: float = 120.0        # knee angle ≤ this → DOWN
    knee_down_min: float = 80.0         # valid depth range lower bound
    knee_up_threshold: float = 155.0    # knee angle ≥ this → UP

    # ── Column names (adaptable to different CSV schemas) ─────────────────
    confidence_col: str = "raw_confidence"
    exercise_col: str = "raw_exercise_label"
    phase_col: str = "raw_phase_label"
    frame_col: str = "frame_number"
    timestamp_col: str = "timestamp"
    track_col: str = "track_id"
    session_col: str = "session_id"
    knee_left_col: str = "knee_left_deg"
    knee_right_col: str = "knee_right_deg"


# ══════════════════════════════════════════════════════════════════════════════
# State Machine
# ══════════════════════════════════════════════════════════════════════════════

class _State(Enum):
    IDLE = auto()
    CANDIDATE_SQUAT = auto()
    IN_SQUAT = auto()
    EXITING_SQUAT = auto()


@dataclass
class _Segment:
    """A detected exercise segment."""
    start_idx: int
    end_idx: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    exercise: str
    mean_confidence: float
    rep_count: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# Core Functions
# ══════════════════════════════════════════════════════════════════════════════

def smooth_confidence(
    values: np.ndarray,
    method: str = "median",
    window: int = 7,
    ema_alpha: float = 0.3,
) -> np.ndarray:
    """
    Smooth frame-level confidence scores.

    Parameters
    ----------
    values : np.ndarray
        Raw per-frame confidence values.
    method : str
        Smoothing method: "median", "mean", or "ema".
    window : int
        Window size for median/mean filter (must be odd for median).
    ema_alpha : float
        Alpha coefficient for exponential moving average.

    Returns
    -------
    np.ndarray
        Smoothed confidence values, same length as input.
    """
    n = len(values)
    if n == 0:
        return values.copy()

    if method == "median":
        # Ensure odd window
        w = window if window % 2 == 1 else window + 1
        half = w // 2
        smoothed = np.zeros(n)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            smoothed[i] = np.median(values[lo:hi])
        return smoothed

    elif method == "mean":
        half = window // 2
        smoothed = np.zeros(n)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            smoothed[i] = np.mean(values[lo:hi])
        return smoothed

    elif method == "ema":
        smoothed = np.zeros(n)
        smoothed[0] = values[0]
        for i in range(1, n):
            smoothed[i] = ema_alpha * values[i] + (1 - ema_alpha) * smoothed[i - 1]
        return smoothed

    else:
        logger.warning("Unknown smoothing method '{}', returning raw values", method)
        return values.copy()


def detect_segments(
    smoothed_conf: np.ndarray,
    frames: np.ndarray,
    timestamps: np.ndarray,
    phases: np.ndarray,
    exercises: np.ndarray,
    cfg: SegmentationConfig,
    knee_angles: np.ndarray | None = None,
) -> list[_Segment]:
    """
    Run hysteresis state machine to detect exercise segments.

    Parameters
    ----------
    smoothed_conf : np.ndarray
        Smoothed confidence values per frame.
    frames : np.ndarray
        Frame numbers.
    timestamps : np.ndarray
        Timestamps per frame.
    phases : np.ndarray
        Phase labels per frame ("up", "down", "none").
    exercises : np.ndarray
        Exercise labels per frame.
    cfg : SegmentationConfig
        Configuration parameters.

    Returns
    -------
    list[_Segment]
        Detected exercise segments after filtering and merging.
    """
    n = len(smoothed_conf)
    raw_segments: list[_Segment] = []

    state = _State.IDLE
    candidate_start = 0
    cooldown_counter = 0
    candidate_exercise = ""

    for i in range(n):
        conf = smoothed_conf[i]
        ex = exercises[i] if exercises[i] not in ("none", "", "pending") else ""

        if state == _State.IDLE:
            if conf >= cfg.enter_threshold:
                state = _State.CANDIDATE_SQUAT
                candidate_start = i
                candidate_exercise = ex if ex else "squat"

        elif state == _State.CANDIDATE_SQUAT:
            if conf >= cfg.enter_threshold:
                consecutive = i - candidate_start + 1
                if consecutive >= cfg.confirm_frames:
                    state = _State.IN_SQUAT
            else:
                # Reset — not enough consecutive high-confidence frames
                state = _State.IDLE
                # Check if this frame itself starts a new candidate
                if conf >= cfg.enter_threshold:
                    state = _State.CANDIDATE_SQUAT
                    candidate_start = i
                    candidate_exercise = ex if ex else "squat"

        elif state == _State.IN_SQUAT:
            if conf < cfg.exit_threshold:
                state = _State.EXITING_SQUAT
                cooldown_counter = 1

        elif state == _State.EXITING_SQUAT:
            if conf >= cfg.exit_threshold:
                state = _State.IN_SQUAT
                cooldown_counter = 0
            else:
                cooldown_counter += 1
                if cooldown_counter >= cfg.cooldown_frames:
                    # Close the segment
                    seg_end = i - cooldown_counter
                    if seg_end < candidate_start:
                        seg_end = candidate_start
                    raw_segments.append(_Segment(
                        start_idx=candidate_start,
                        end_idx=seg_end,
                        start_frame=int(frames[candidate_start]),
                        end_frame=int(frames[seg_end]),
                        start_time=float(timestamps[candidate_start]),
                        end_time=float(timestamps[seg_end]),
                        exercise=candidate_exercise,
                        mean_confidence=float(
                            np.mean(smoothed_conf[candidate_start:seg_end + 1])
                        ),
                    ))
                    state = _State.IDLE
                    cooldown_counter = 0

    # Handle end-of-stream: if still in a segment, close it
    if state in (_State.IN_SQUAT, _State.EXITING_SQUAT):
        seg_end = n - 1
        if state == _State.EXITING_SQUAT:
            seg_end = max(candidate_start, n - 1 - cooldown_counter)
        raw_segments.append(_Segment(
            start_idx=candidate_start,
            end_idx=seg_end,
            start_frame=int(frames[candidate_start]),
            end_frame=int(frames[seg_end]),
            start_time=float(timestamps[candidate_start]),
            end_time=float(timestamps[seg_end]),
            exercise=candidate_exercise,
            mean_confidence=float(
                np.mean(smoothed_conf[candidate_start:seg_end + 1])
            ),
        ))

    # ── Filter by minimum segment length ─────────────────────────────────
    filtered = [
        s for s in raw_segments
        if (s.end_idx - s.start_idx + 1) >= cfg.min_segment_frames
    ]
    if len(filtered) < len(raw_segments):
        logger.info(
            "Filtered {} short segments (< {} frames)",
            len(raw_segments) - len(filtered), cfg.min_segment_frames,
        )

    # ── Merge adjacent segments with small gaps ──────────────────────────
    merged = _merge_segments(filtered, frames, timestamps, smoothed_conf, cfg)

    # ── Count reps within each segment ───────────────────────────────────
    for seg in merged:
        if knee_angles is not None:
            seg.rep_count, _ = _count_reps_by_knee_angle(
                knee_angles[seg.start_idx:seg.end_idx + 1],
                cfg.knee_down_max, cfg.knee_down_min, cfg.knee_up_threshold,
            )
        else:
            seg.rep_count = _count_reps_by_phase_labels(
                phases[seg.start_idx:seg.end_idx + 1]
            )

    return merged


def _merge_segments(
    segments: list[_Segment],
    frames: np.ndarray,
    timestamps: np.ndarray,
    smoothed_conf: np.ndarray,
    cfg: SegmentationConfig,
) -> list[_Segment]:
    """Merge segments that are separated by fewer than merge_gap_frames."""
    if len(segments) <= 1:
        return segments

    merged: list[_Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg.start_idx - prev.end_idx - 1
        if gap <= cfg.merge_gap_frames:
            # Merge: extend previous segment
            prev.end_idx = seg.end_idx
            prev.end_frame = seg.end_frame
            prev.end_time = seg.end_time
            prev.mean_confidence = float(
                np.mean(smoothed_conf[prev.start_idx:prev.end_idx + 1])
            )
            logger.debug(
                "Merged segments (gap={} frames): frames {}-{}",
                gap, prev.start_frame, prev.end_frame,
            )
        else:
            merged.append(seg)

    return merged


def _count_reps_by_knee_angle(
    knee_angles: np.ndarray,
    down_max: float = 120.0,
    down_min: float = 80.0,
    up_threshold: float = 155.0,
) -> tuple[int, list[str]]:
    """
    Count reps using knee angle oscillation and compute per-frame phase.

    Uses the same UP/DOWN state machine as SquatLstmDetector._update_phase():
    - UP → DOWN when knee_angle ≤ down_max
    - DOWN → UP when knee_angle ≥ up_threshold (rep counted if valid depth)

    Returns (rep_count, per_frame_phases).
    """
    n = len(knee_angles)
    phases = ["none"] * n
    phase = "up"
    valid_down = False
    rep_count = 0

    for i in range(n):
        ka = knee_angles[i]
        if np.isnan(ka):
            phases[i] = phase  # keep current phase
            continue

        if phase == "up":
            if ka <= down_max:
                phase = "down"
                if down_min <= ka <= down_max:
                    valid_down = True
        elif phase == "down":
            if down_min <= ka <= down_max:
                valid_down = True
            if ka >= up_threshold:
                phase = "up"
                if valid_down:
                    rep_count += 1
                valid_down = False

        phases[i] = phase

    return rep_count, phases


def _count_reps_by_phase_labels(phases: np.ndarray) -> int:
    """Fallback: count down→up transitions in phase labels."""
    count = 0
    prev = None
    for phase in phases:
        if prev == "down" and phase == "up":
            count += 1
        prev = phase
    return count


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def segment_session_csv(
    input_path: str | Path,
    output_path: str | Path | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Process a session CSV and apply temporal segmentation labels.

    Parameters
    ----------
    input_path : str | Path
        Path to the input frames.csv.
    output_path : str | Path | None
        Path for the output CSV. If None, returns DataFrame only.
    config : dict | None
        Override config values. Keys match SegmentationConfig fields.

    Returns
    -------
    pd.DataFrame
        The labeled DataFrame with segmentation columns.
    """
    cfg = SegmentationConfig()
    if config:
        for key, val in config.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    logger.info("📂 Loading session CSV: {}", input_path)
    df = pd.read_csv(input_path)
    logger.info("📈 Total rows: {}", len(df))

    # ── Ensure required columns exist ────────────────────────────────────
    required_cols = [cfg.frame_col, cfg.track_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Add output columns with defaults ─────────────────────────────────
    output_cols = {
        "segment_id": 0,
        "segment_start_frame": -1,
        "segment_end_frame": -1,
        "segment_start_time": -1.0,
        "segment_end_time": -1.0,
        "prediction_confidence": 0.0,
    }
    for col, default in output_cols.items():
        if col not in df.columns:
            df[col] = default

    # ── Handle confidence column ─────────────────────────────────────────
    if cfg.confidence_col not in df.columns:
        logger.warning(
            "Confidence column '{}' not found. Using 0.0 for all frames.",
            cfg.confidence_col,
        )
        df[cfg.confidence_col] = 0.0
    else:
        df[cfg.confidence_col] = pd.to_numeric(
            df[cfg.confidence_col], errors="coerce"
        ).fillna(0.0)

    # ── Handle optional columns gracefully ───────────────────────────────
    if cfg.timestamp_col not in df.columns:
        df[cfg.timestamp_col] = 0.0
    if cfg.phase_col not in df.columns:
        df[cfg.phase_col] = "none"
    if cfg.exercise_col not in df.columns:
        df[cfg.exercise_col] = "none"

    # ── Group by session + track ─────────────────────────────────────────
    group_cols = [cfg.track_col]
    if cfg.session_col in df.columns:
        group_cols.insert(0, cfg.session_col)

    global_segment_id = 0

    for group_key, group_df in df.groupby(group_cols, sort=False):
        group_df = group_df.sort_values(cfg.frame_col)
        indices = group_df.index.values

        if len(indices) == 0:
            continue

        # Extract arrays
        raw_conf = group_df[cfg.confidence_col].values.astype(np.float64)
        frame_nums = group_df[cfg.frame_col].values
        timestamps = group_df[cfg.timestamp_col].values.astype(np.float64)
        phases = group_df[cfg.phase_col].fillna("none").values
        exercises = group_df[cfg.exercise_col].fillna("none").values

        # Extract knee angle columns (for angle-based rep counting)
        has_knee_angles = (
            cfg.knee_left_col in group_df.columns
            and cfg.knee_right_col in group_df.columns
        )
        knee_angles = None
        if has_knee_angles:
            kl = pd.to_numeric(
                group_df[cfg.knee_left_col], errors="coerce"
            ).values.astype(np.float64)
            kr = pd.to_numeric(
                group_df[cfg.knee_right_col], errors="coerce"
            ).values.astype(np.float64)
            knee_angles = np.nanmean(np.column_stack([kl, kr]), axis=1)

        # ── Step 1: Smooth confidence ────────────────────────────────
        smoothed = smooth_confidence(
            raw_conf,
            method=cfg.smoothing_method,
            window=cfg.smoothing_window,
            ema_alpha=cfg.ema_alpha,
        )

        # ── Step 2: Detect segments ──────────────────────────────────
        segments = detect_segments(
            smoothed, frame_nums, timestamps, phases, exercises, cfg,
            knee_angles=knee_angles,
        )

        # ── Step 3: Apply labels ─────────────────────────────────────────
        # Default: no exercise for all frames in this group
        df.loc[indices, "activity_label"] = "no_exercise"
        df.loc[indices, "exercise_label"] = "none"
        df.loc[indices, "phase_label"] = "none"
        df.loc[indices, "label_source"] = "smoothed"
        df.loc[indices, "decision_status"] = "confirmed"
        df.loc[indices, "segment_id"] = 0
        df.loc[indices, "segment_start_frame"] = -1
        df.loc[indices, "segment_end_frame"] = -1
        df.loc[indices, "segment_start_time"] = -1.0
        df.loc[indices, "segment_end_time"] = -1.0
        df.loc[indices, "prediction_confidence"] = 0.0
        df.loc[indices, "rep_id"] = 0
        df.loc[indices, "event_id"] = 0

        for seg in segments:
            global_segment_id += 1
            seg_indices = indices[seg.start_idx:seg.end_idx + 1]

            df.loc[seg_indices, "activity_label"] = "exercise"
            df.loc[seg_indices, "exercise_label"] = seg.exercise
            df.loc[seg_indices, "label_source"] = "smoothed"
            df.loc[seg_indices, "decision_status"] = "confirmed"
            df.loc[seg_indices, "segment_id"] = global_segment_id
            df.loc[seg_indices, "event_id"] = global_segment_id
            df.loc[seg_indices, "segment_start_frame"] = seg.start_frame
            df.loc[seg_indices, "segment_end_frame"] = seg.end_frame
            df.loc[seg_indices, "segment_start_time"] = seg.start_time
            df.loc[seg_indices, "segment_end_time"] = seg.end_time
            df.loc[seg_indices, "prediction_confidence"] = round(
                seg.mean_confidence, 4
            )

            # Assign per-frame phase labels and rep_id within segment
            # Prefer knee-angle-based phase detection over raw_phase_label
            if knee_angles is not None:
                seg_knee = knee_angles[seg.start_idx:seg.end_idx + 1]
                _, seg_phases = _count_reps_by_knee_angle(
                    seg_knee, cfg.knee_down_max,
                    cfg.knee_down_min, cfg.knee_up_threshold,
                )
                rep_id = 0
                prev_phase = None
                for local_i, phase in enumerate(seg_phases):
                    global_i = seg_indices[local_i]
                    df.loc[global_i, "phase_label"] = phase
                    if prev_phase == "down" and phase == "up":
                        rep_id += 1
                    df.loc[global_i, "rep_id"] = rep_id
                    prev_phase = phase
            else:
                # Fallback: use raw_phase_label transitions
                seg_phases = phases[seg.start_idx:seg.end_idx + 1]
                rep_id = 0
                prev_phase = None
                for local_i, phase in enumerate(seg_phases):
                    global_i = seg_indices[local_i]
                    df.loc[global_i, "phase_label"] = phase
                    if prev_phase == "down" and phase == "up":
                        rep_id += 1
                    df.loc[global_i, "rep_id"] = rep_id
                    prev_phase = phase

        track_info = group_key if isinstance(group_key, tuple) else (group_key,)
        logger.info(
            "Track {}: {} segments detected, {} total reps",
            track_info,
            len(segments),
            sum(s.rep_count for s in segments),
        )

    # ── Write output ─────────────────────────────────────────────────────
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("📄 Segmented CSV written: {}", output_path)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Post-process a session CSV with temporal segmentation."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the input frames.csv",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path for output CSV (default: input with _segmented suffix)",
    )

    # Smoothing
    parser.add_argument("--smoothing-method", default="median",
                        choices=["median", "mean", "ema"])
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--ema-alpha", type=float, default=0.3)

    # Thresholds
    parser.add_argument("--enter-threshold", type=float, default=0.5)
    parser.add_argument("--exit-threshold", type=float, default=0.3)

    # Temporal constraints
    parser.add_argument("--confirm-frames", type=int, default=12)
    parser.add_argument("--cooldown-frames", type=int, default=8)
    parser.add_argument("--min-segment-frames", type=int, default=8)
    parser.add_argument("--merge-gap-frames", type=int, default=5)

    return parser.parse_args()


def main():
    args = _parse_args()

    output_path = args.output
    if output_path is None:
        p = Path(args.input)
        output_path = p.parent / f"{p.stem}_segmented{p.suffix}"

    config = {
        "smoothing_method": args.smoothing_method,
        "smoothing_window": args.smoothing_window,
        "ema_alpha": args.ema_alpha,
        "enter_threshold": args.enter_threshold,
        "exit_threshold": args.exit_threshold,
        "confirm_frames": args.confirm_frames,
        "cooldown_frames": args.cooldown_frames,
        "min_segment_frames": args.min_segment_frames,
        "merge_gap_frames": args.merge_gap_frames,
    }

    df = segment_session_csv(args.input, output_path, config=config)

    # Print summary
    exercise_frames = (df["activity_label"] == "exercise").sum()
    total_frames = len(df)
    n_segments = df["segment_id"].nunique() - (1 if 0 in df["segment_id"].values else 0)

    print(f"\n{'='*60}")
    print(f"📊 Segmentation Summary")
    print(f"{'='*60}")
    print(f"  Total frames:    {total_frames}")
    print(f"  Exercise frames: {exercise_frames} ({100*exercise_frames/total_frames:.1f}%)")
    print(f"  Idle frames:     {total_frames - exercise_frames}")
    print(f"  Segments:        {n_segments}")
    if n_segments > 0:
        seg_df = df[df["segment_id"] > 0]
        for sid in sorted(seg_df["segment_id"].unique()):
            seg = seg_df[seg_df["segment_id"] == sid]
            reps = seg["rep_id"].max()
            conf = seg["prediction_confidence"].iloc[0]
            sf = seg["segment_start_frame"].iloc[0]
            ef = seg["segment_end_frame"].iloc[0]
            print(f"  Segment {sid}: frames {sf}-{ef} | "
                  f"{len(seg)} frames | {reps} reps | "
                  f"confidence={conf:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
