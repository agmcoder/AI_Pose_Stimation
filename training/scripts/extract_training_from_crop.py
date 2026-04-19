#!/usr/bin/env python3
"""
training/scripts/extract_training_from_crop.py

Headless script for extracting training data from a folder of cropped videos.

Runs YOLO pose inference on every video found in the input folder, computes all
derived features (angles, velocity, normalized keypoints, quality metrics), and
exports a SINGLE consolidated CSV whose label columns follow the normalized schema:

  activity_label : "exercise" | "no_exercise"
  exercise_label : canonical snake_case name (e.g. "squat") or "none"
  phase_label    : semantic phase for the exercise (e.g. "up", "down") or "none"

Phase detection is NOT performed here — the script writes phase_label="none"
for every frame.  Use clean_training_csv.py afterwards to auto-compute phases
from biomechanical signals.

Supported video extensions: .mp4 .mov .avi .mkv .webm
Total: 92-column schema (FrameRecord.csv_header).

Usage:
    python training/scripts/extract_training_from_crop.py \\
        --video_dir /path/to/videos/ \\
        --output_csv training/data/train_squats.csv \\
        --eval_csv training/data/squat_reference.csv \\
        --model models/weights/yolo26x-pose.pt \\
        --device cuda:0 \\
        --exercise_label squat
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Setup project root on sys.path ──────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.types import FrameRecord
from src.utils.body_angles import compute_body_angles
from src.utils.keypoint_normalizer import normalize_keypoints
from src.utils.pose_quality import compute_pose_quality
from src.utils.velocity_tracker import VelocityTracker

from loguru import logger

from label_normalizer import (
    normalize_exercise_label,
    normalize_phase_label,
)


# ── Supported video extensions ───────────────────────────────────────────────
VIDEO_EXTENSIONS: set[str] = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# ── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract training CSV from a folder of crop videos using YOLO pose.",
    )
    p.add_argument(
        "--video_dir", required=True,
        help="Path to the folder containing crop video files (.mp4 .mov .avi .mkv .webm).",
    )
    p.add_argument("--output_csv", required=True, help="Path for the output consolidated CSV.")
    p.add_argument(
        "--eval_csv", required=True,
        help="Path to an existing evaluation CSV (only its header is read).",
    )
    p.add_argument(
        "--model", default="models/weights/yolo26x-pose.pt",
        help="YOLO pose model weights (default: models/weights/yolo26x-pose.pt).",
    )
    p.add_argument(
        "--device", default="auto",
        help="Inference device: auto | cpu | cuda | cuda:0 | mps (default: auto).",
    )
    p.add_argument(
        "--target_track_id", type=int, default=None,
        help="If set, use only this track ID for ALL videos. Otherwise auto-select per video.",
    )
    p.add_argument(
        "--exercise_label", default="none",
        help=(
            "Exercise name to assign (default: none). "
            "Recognized: squat, jumping_jacks, pushup, pullup, etc. "
            "Aliases like 'squats' or 'frontjump' are auto-resolved."
        ),
    )
    p.add_argument(
        "--person_id", default=None,
        help=(
            "Optional label identifying the subject (e.g. 'alex', 'mario'). "
            "Added as a 'person_id' column so CV can be grouped by person "
            "instead of by video, enabling Leave-One-Person-Out evaluation."
        ),
    )
    return p.parse_args()


# ── YOLO result parsing ────────────────────────────────────────────────────

def _parse_persons(result) -> List[Dict]:
    """
    Parse a single YOLO result into a list of person dicts.

    Each dict has:
        track_id: int
        bbox: (x1, y1, x2, y2)
        coords: np.ndarray (17, 2)
        scores: np.ndarray (17,)
    """
    persons = []
    if result.boxes is None or result.boxes.id is None:
        return persons

    for i, (box, tid) in enumerate(
        zip(result.boxes.xyxy, result.boxes.id.int())
    ):
        kps = result.keypoints[i]
        persons.append({
            "track_id": int(tid),
            "bbox": tuple(box.cpu().numpy().astype(float)),
            "coords": kps.xy[0].cpu().numpy().astype(np.float32),
            "scores": kps.conf[0].cpu().numpy().astype(np.float32),
        })
    return persons


# ── Subject selection heuristic ─────────────────────────────────────────────

class _SubjectSelector:
    """
    Selects the principal subject from a crop video when no target_track_id
    is provided.

    Accumulates statistics per track and scores them using:
        score = detection_count × mean_bbox_area × mean_centrality

    Where centrality = 1 - distance_from_frame_center (normalized).
    """

    def __init__(self, frame_w: int, frame_h: int) -> None:
        self._frame_cx = frame_w / 2.0
        self._frame_cy = frame_h / 2.0
        self._frame_diag = max(np.hypot(frame_w, frame_h), 1e-6)
        self._counts: Counter = Counter()
        self._areas: defaultdict = defaultdict(list)
        self._centralities: defaultdict = defaultdict(list)

    def observe(self, track_id: int, bbox: Tuple[float, ...]) -> None:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = np.hypot(cx - self._frame_cx, cy - self._frame_cy)
        centrality = 1.0 - (dist / self._frame_diag)

        self._counts[track_id] += 1
        self._areas[track_id].append(area)
        self._centralities[track_id].append(centrality)

    def best_track_id(self) -> Optional[int]:
        if not self._counts:
            return None

        best_id, best_score = None, -1.0
        for tid in self._counts:
            score = (
                self._counts[tid]
                * float(np.mean(self._areas[tid]))
                * float(np.mean(self._centralities[tid]))
            )
            if score > best_score:
                best_score, best_id = score, tid
        return best_id


# ── Frame processing ────────────────────────────────────────────────────────

def _build_record(
    person: Dict,
    frame_number: int,
    timestamp: float,
    fps: float,
    exercise_label: str,
    phase_label: str,
    velocity_tracker: VelocityTracker,
    activity_label: str = "no_exercise",
    session_id: str = "",
    video_id: str = "",
    person_id: str = "",
) -> FrameRecord:
    """Build a FrameRecord from a parsed person dict, computing all features."""
    coords = person["coords"]
    scores = person["scores"]
    bbox = person["bbox"]
    track_id = person["track_id"]

    quality = compute_pose_quality(scores)
    kp_norm = normalize_keypoints(coords, bbox)

    body_angles = {}
    if quality.is_valid_pose:
        body_angles = compute_body_angles(coords, scores)

    velocity = velocity_tracker.update(track_id, coords, scores, bbox)

    return FrameRecord(
        timestamp=timestamp,
        frame_number=frame_number,
        track_id=track_id,
        fps=fps,
        session_id=session_id,
        video_id=video_id,
        bbox=bbox,
        activity_label=activity_label,
        exercise_label=exercise_label,
        phase_label=phase_label,
        rep_id=0,
        mean_kpt_conf=quality.mean_kpt_conf,
        visible_kpt_count=quality.visible_kpt_count,
        is_valid_pose=quality.is_valid_pose,
        body_angles=body_angles,
        velocity=velocity,
        keypoints_xy=coords.copy(),
        keypoints_conf=scores.copy(),
        keypoints_xy_norm=kp_norm,
    )


# ── Video discovery ─────────────────────────────────────────────────────────

def _list_videos(video_dir: Path) -> List[Tuple[Path, str]]:
    """
    Return sorted list of (video_path, person_id) tuples.

    Discovery strategy (in order):
      1. If video_dir contains subdirectories with videos inside → treats each
         subdirectory as a person group. person_id = subfolder name.
      2. If all videos are directly in video_dir (flat layout) → person_id = ""
         for all (or use --person_id CLI override).

    Example (subfolder mode):
        dataset/
            alex/Alex_01.mov  → person_id="alex"
            mario/Mario_01.mov → person_id="mario"
    """
    def is_video(p: Path) -> bool:
        return p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS

    # Check for subfolder layout
    subdirs = sorted(d for d in video_dir.iterdir() if d.is_dir())
    subdir_videos: List[Tuple[Path, str]] = []
    for subdir in subdirs:
        vids = sorted(p for p in subdir.iterdir() if is_video(p))
        for v in vids:
            subdir_videos.append((v, subdir.name))

    # Videos directly in root
    root_videos = sorted(p for p in video_dir.iterdir() if is_video(p))

    if subdir_videos and not root_videos:
        # Pure subfolder layout → use subfolder names as person_id
        return subdir_videos
    elif subdir_videos and root_videos:
        # Mixed: warn and use subfolder videos only
        logger.warning(
            "Se encontraron vídeos tanto en la raíz ({}) como en subcarpetas ({}). "
            "Se usarán SOLO los de las subcarpetas. Mueve los vídeos de la raíz a "
            "una subcarpeta con el nombre de la persona.",
            len(root_videos), len(subdir_videos),
        )
        return subdir_videos
    else:
        # Flat layout → no person_id (or overridden by --person_id CLI)
        return [(v, "") for v in root_videos]


# ── Per-video processing ─────────────────────────────────────────────────────

def _process_one_video(
    video_path: Path,
    model,
    device: str,
    exercise_label: str,
    phase_label: str,
    canonical_header: List[str],
    target_track_id: Optional[int],
    person_id: str = "",
) -> Tuple[List[list], int]:
    """
    Run the full extraction pipeline on a single video file.

    Returns:
        (rows, skipped)  where rows are raw CSV rows (canonical_header length).
        Returns ([], 0) on non-fatal errors (corrupt video, no subject found).
    """
    # ── Video metadata ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("No se pudo abrir el vídeo, omitiendo: {}", video_path.name)
        cap.release()
        return [], 0

    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_raw if (fps_raw and fps_raw > 0) else 30.0
    if not fps_raw or fps_raw <= 0:
        logger.warning("FPS no reportado en {}, usando 30.0", video_path.name)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps < 10 or fps > 120:
        logger.warning("FPS sospechoso {:.1f} en {} — timestamps inexactos", fps, video_path.name)

    duration = total_frames / fps if total_frames > 0 else 0.0
    logger.info(
        "  Vídeo: {} | {}x{} | {:.1f} FPS | ~{} frames | {:.1f}s",
        video_path.name, frame_w, frame_h, fps, total_frames, duration,
    )

    # ── Pass 1: discover principal subject (if not fixed) ────────────────
    target_tid = target_track_id

    if target_tid is None:
        logger.info("  Pass 1 — seleccionando sujeto principal...")
        selector = _SubjectSelector(frame_w, frame_h)

        results = model.track(
            source=str(video_path),
            stream=True, show=False, save=False, verbose=False,
            device=device, persist=True, tracker="bytetrack.yaml",
        )
        for result in results:
            for person in _parse_persons(result):
                selector.observe(person["track_id"], person["bbox"])

        target_tid = selector.best_track_id()
        if target_tid is None:
            logger.warning("  No se detectó ningún sujeto en {}, omitiendo.", video_path.name)
            return [], 0
        logger.info("  Sujeto seleccionado: track_id={}", target_tid)

        # E3: Reset tracker between passes
        if hasattr(model, "predictor") and model.predictor is not None:
            model.predictor = None
            logger.debug("  Predictor reseteado entre pasadas")

    # ── Pass 2: extract features ─────────────────────────────────────────
    # Each video gets its own VelocityTracker (independent temporal state)
    velocity_tracker = VelocityTracker()
    video_id = video_path.stem
    session_id = f"train_{video_path.stem}_{int(time.time())}"

    results = model.track(
        source=str(video_path),
        stream=True, show=False, save=False, verbose=False,
        device=device, persist=True, tracker="bytetrack.yaml",
    )

    rows: List[list] = []
    frame_idx = 0
    skipped = 0
    for result in results:
        timestamp = frame_idx / fps
        persons = _parse_persons(result)

        target_persons = [p for p in persons if p["track_id"] == target_tid]
        if not target_persons:
            skipped += 1
            frame_idx += 1
            continue

        record = _build_record(
            person=target_persons[0],
            frame_number=frame_idx,
            timestamp=timestamp,
            fps=fps,
            exercise_label=exercise_label,
            phase_label=phase_label,
            velocity_tracker=velocity_tracker,
            session_id=session_id,
            video_id=video_id,
            person_id=person_id,
        )
        rows.append(record.to_csv_row())
        frame_idx += 1

    logger.info(
        "  ✓ {} — {} filas, {} skipped (track_id={})",
        video_path.name, len(rows), skipped, target_tid,
    )

    # E3: Reset tracker before next video
    if hasattr(model, "predictor") and model.predictor is not None:
        model.predictor = None

    return rows, skipped


# ── Main pipeline ──────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── Validate video folder ────────────────────────────────────────────
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        logger.error("Carpeta no encontrada: {}", video_dir)
        sys.exit(1)
    if not video_dir.is_dir():
        logger.error("La ruta no es una carpeta: {}", video_dir)
        sys.exit(1)

    videos = _list_videos(video_dir)
    if not videos:
        logger.error(
            "No se encontraron vídeos ({}) en: {}",
            ", ".join(VIDEO_EXTENSIONS), video_dir,
        )
        sys.exit(1)

    # Report what was found
    persons_found = sorted(set(pid for _, pid in videos if pid))
    if persons_found:
        logger.info(
            "Encontrados {} vídeos de {} personas en subcarpetas: {}",
            len(videos), len(persons_found), persons_found,
        )
    else:
        logger.info("Encontrados {} vídeos en {} (modo plano)", len(videos), video_dir)
    for v, pid in videos:
        logger.info("  • {} [person_id='{}']", v.name, pid or args.person_id or "")

    # ── Validate other inputs ────────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("Modelo no encontrado: {}", model_path)
        sys.exit(1)

    eval_csv_path = Path(args.eval_csv)
    if not eval_csv_path.exists():
        logger.error("CSV de evaluación no encontrado: {}", eval_csv_path)
        sys.exit(1)

    # ── Read evaluation CSV header (contract) ───────────────────────────
    with open(eval_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        eval_header = next(reader)

    canonical_header = FrameRecord.csv_header()
    if eval_header != canonical_header:
        if len(eval_header) != len(canonical_header):
            logger.warning(
                "El CSV de evaluación tiene {} columnas, esperadas {}. "
                "Se usará el esquema canónico.",
                len(eval_header), len(canonical_header),
            )
        else:
            diff_cols = [
                (i, e, c) for i, (e, c)
                in enumerate(zip(eval_header, canonical_header)) if e != c
            ]
            if diff_cols:
                logger.info(
                    "El CSV de evaluación difiere en {} columnas (mismo conteo: {}). "
                    "Primera diferencia en posición {}: '{}' vs '{}'. "
                    "Se usará el esquema canónico.",
                    len(diff_cols), len(eval_header),
                    diff_cols[0][0], diff_cols[0][1], diff_cols[0][2],
                )
    logger.info("Esquema CSV: {} columnas", len(canonical_header))

    # ── Normalize labels from CLI ───────────────────────────────────────
    exercise_label = normalize_exercise_label(args.exercise_label)
    phase_label = normalize_phase_label(exercise_label, "none")
    logger.info(
        "Labels — raw_exercise='{}' → exercise_label='{}', phase_label='{}'",
        args.exercise_label, exercise_label, phase_label,
    )

    # ── Load YOLO model (E1+E2: robust device selection) ─────────────────
    from ultralytics import YOLO
    import torch

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    elif device.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA no disponible, cayendo a CPU")
            device = "cpu"
        elif ":" in device:
            gpu_idx = int(device.split(":")[1])
            if gpu_idx >= torch.cuda.device_count():
                logger.warning(
                    "GPU {} no existe (hay {}), usando cuda:0",
                    gpu_idx, torch.cuda.device_count(),
                )
                device = "cuda:0"

    logger.info("Cargando modelo {} en {}", model_path, device)
    model = YOLO(str(model_path))
    model.to(device)

    # ── Process each video ───────────────────────────────────────────────
    all_rows: List[list] = []
    total_skipped = 0
    failed_videos: List[str] = []

    for idx, (video_path, discovered_person_id) in enumerate(videos, 1):
        # CLI --person_id overrides auto-discovered subfolder name
        effective_person_id = args.person_id or discovered_person_id or ""
        logger.info(
            "[{}/{}] Procesando: {} [person_id='{}']",
            idx, len(videos), video_path.name, effective_person_id,
        )
        try:
            rows, skipped = _process_one_video(
                video_path=video_path,
                model=model,
                device=device,
                exercise_label=exercise_label,
                phase_label=phase_label,
                canonical_header=canonical_header,
                target_track_id=args.target_track_id,
                person_id=effective_person_id,
            )
            all_rows.extend(rows)
            total_skipped += skipped
            if not rows:
                failed_videos.append(video_path.name)
        except Exception as exc:
            logger.error("Error inesperado en {}: {}", video_path.name, exc)
            failed_videos.append(video_path.name)

    if not all_rows:
        logger.error("No se generaron registros de ningún vídeo. Abortando.")
        sys.exit(1)

    if failed_videos:
        logger.warning(
            "{} vídeo(s) no produjeron datos: {}",
            len(failed_videos), ", ".join(failed_videos),
        )

    # ── Remove activity_label column ────────────────────────────────────
    try:
        output_header = [col for col in canonical_header if col != "activity_label"]
        # Inject person_id column after video_id (if provided)
        if args.person_id and "person_id" not in output_header:
            vid_idx = output_header.index("video_id") if "video_id" in output_header else -1
            if vid_idx >= 0:
                output_header.insert(vid_idx + 1, "person_id")
            else:
                output_header.append("person_id")
        output_rows = []
        for row in all_rows:
            if len(row) == len(canonical_header):
                filtered = [
                    val for i, val in enumerate(row)
                    if canonical_header[i] != "activity_label"
                ]
                if args.person_id and "person_id" not in canonical_header:
                    vid_col_pos = output_header.index("person_id")
                    filtered.insert(vid_col_pos, args.person_id)
                output_rows.append(filtered)
            else:
                logger.warning("Row length mismatch: expected {}, got {}", len(canonical_header), len(row))
    except ValueError:
        logger.warning("activity_label column not found in canonical header")
        output_header = canonical_header
        output_rows = all_rows

    # ── Write consolidated CSV ───────────────────────────────────────────
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(output_header)
        writer.writerows(output_rows)

    logger.info(
        "CSV consolidado guardado: {}\n"
        "   vídeos procesados={}/{}, filas={}, columnas={}, total_skipped={}\n"
        "   exercise_label={}, phase_label={}",
        output_path, len(videos) - len(failed_videos), len(videos),
        len(output_rows), len(output_header), total_skipped,
        exercise_label, phase_label,
    )


if __name__ == "__main__":
    main()

