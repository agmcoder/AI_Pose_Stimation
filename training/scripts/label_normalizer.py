"""
training/scripts/label_normalizer.py

Centralizes all label normalization logic for the training pipeline.

Contracts:
  - activity_label: "none" or canonical snake_case exercise name
  - exercise_label: canonical snake_case name or "none"
  - phase_label:    "phase_1" | "phase_2" | "none" (universal for all exercises)

Used by:
  - extract_training_from_crop.py
  - clean_training_csv.py
  - record_builder.py (runtime pipeline)
"""
from __future__ import annotations

from typing import Dict

from loguru import logger

# ── Exercise name mapping ─────────────────────────────────────────────────
# Alias / legacy name  →  canonical snake_case name
# Add new aliases here when new exercises or naming variants appear.

EXERCISE_ALIAS: Dict[str, str] = {
    # squat
    "squat":          "squat",
    "squats":         "squat",
    # jumping jacks
    "jumping_jacks":  "jumping_jacks",
    "jumpingjacks":   "jumping_jacks",
    "frontjump":      "jumping_jacks",
    # pushup
    "pushup":         "pushup",
    "pushups":        "pushup",
    "push_up":        "pushup",
    "push_ups":       "pushup",
    # pullup
    "pullup":         "pullup",
    "pullups":        "pullup",
    "pull_up":        "pullup",
    "pull_ups":       "pullup",
    # no-exercise markers
    "idle":           "none",
    "none":           "none",
    "":               "none",
}


# ── Universal phase schema ────────────────────────────────────────────────
# All exercises share the same two phases:
#   phase_1 — first part of the movement  (descent / opening / lowering)
#   phase_2 — second part of the movement (ascent  / closing  / raising)
#   none    — rest, transition, or unknown
#
# The mapping from raw/legacy names to universal phases is below.

VALID_PHASES: set[str] = {"phase_1", "phase_2", "none"}

_PHASE_ALIAS: Dict[str, str] = {
    # Descent / first half
    "down":    "phase_1",
    "open":    "phase_1",
    # Ascent / second half
    "up":      "phase_2",
    "close":   "phase_2",
    "closed":  "phase_2",
    # Already universal
    "phase_1": "phase_1",
    "phase_2": "phase_2",
    # No-phase markers
    "idle":    "none",
    "unknown": "none",
    "none":    "none",
    "":        "none",
}


# ── Public API ────────────────────────────────────────────────────────────

def normalize_exercise_label(raw_exercise: str) -> str:
    """
    Resolve a raw exercise name to its canonical form.

    Returns "none" for empty, unknown, or idle-like inputs.
    """
    if not raw_exercise:
        return "none"
    key = raw_exercise.strip().lower()
    canonical = EXERCISE_ALIAS.get(key)
    if canonical is None:
        logger.warning(
            "Unknown exercise '{}' — defaulting to 'none'", raw_exercise,
        )
        return "none"
    return canonical


def derive_activity_label(exercise_label: str) -> str:
    """
    Derive activity from the *already-normalized* exercise label.

    "none" → "none", anything else → the canonical exercise name.
    """
    return "none" if exercise_label == "none" else exercise_label


def normalize_phase_label(
    exercise_label: str,
    raw_phase: str,
) -> str:
    """
    Normalize a raw phase string to the universal schema.

    Returns "phase_1", "phase_2", or "none".

    Rules:
      1. If exercise is "none" → "none" (no exercise = no phase).
      2. Resolve raw phase through the alias table → phase_1 / phase_2 / none.
      3. If result is not in VALID_PHASES → "none" with debug log.
    """
    if exercise_label == "none":
        return "none"

    if not raw_phase:
        return "none"

    resolved = _PHASE_ALIAS.get(raw_phase.strip().lower(), "none")

    if resolved in VALID_PHASES:
        return resolved

    logger.debug(
        "Phase '{}' could not be mapped (resolved='{}') — defaulting to 'none'",
        raw_phase, resolved,
    )
    return "none"
