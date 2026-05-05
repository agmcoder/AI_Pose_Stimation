"""
src/pipeline/pipeline_thread.py

PipelineThread — background QThread that owns the entire domain pipeline.

Responsibilities (SRP):
  - Read frames from VideoCapture
  - Run the full domain pipeline: detect → merge → process → persist → collect → render
  - Emit lightweight ViewModels to the Qt main thread via signals
  - Never touch Qt widgets directly

Design:
  - The main thread only calls setPixmap / setText (microseconds per frame)
  - The pipeline thread runs as fast as the GPU allows, fully decoupled from UI refresh
  - Qt queued connections deliver signals across threads safely
"""
from __future__ import annotations

import time
from collections import deque

import cv2
from loguru import logger
from PySide6.QtCore import QThread, Signal

from src.data_collection.bus import DataCollectionBus
from src.data_collection.deferred_label_buffer import DeferredLabelBuffer
from src.data_collection.record_builder import build_frame_records
from src.utils.velocity_tracker import VelocityTracker
from src.models.yolo_pose import YoloPoseDetector
from src.pipeline.frame_processor import FrameProcessor
from src.pipeline.person_registry import PersonRegistry
from src.presentation.presenter import DashboardPresenter
from src.visualization.renderer import Renderer


class _FpsTracker:
    """Moving-average FPS over the last N frames."""

    def __init__(self, window_size: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window_size)

    def tick(self, t: float) -> float:
        self._times.append(t)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return 0.0 if elapsed == 0 else (len(self._times) - 1) / elapsed


class PipelineThread(QThread):
    """
    Runs the domain pipeline on a dedicated OS thread.

    Signals emitted (queued → delivered to main thread's event loop):
      frame_ready(object)  — VideoFrameVM with the annotated frame
      stats_ready(object)  — DashboardVM with session stats
      stream_ended()       — cap.read() returned False; caller should quit
    """

    frame_ready  = Signal(object)   # VideoFrameVM
    stats_ready  = Signal(object)   # DashboardVM
    stream_ended = Signal()

    def __init__(
        self,
        cap: cv2.VideoCapture,
        detector: YoloPoseDetector,
        registry: PersonRegistry,
        processor: FrameProcessor,
        data_bus: DataCollectionBus,
        renderer: Renderer,
        presenter: DashboardPresenter,
        session_id: str = "",
        video_id: str = "",
        deferred_cfg: dict | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._cap       = cap
        self._detector  = detector
        self._registry  = registry
        self._processor = processor
        self._data_bus  = data_bus
        self._renderer  = renderer
        self._presenter = presenter
        self._fps       = _FpsTracker(window_size=30)
        self._frame_no  = 0
        self._running   = True
        self._session_id = session_id
        self._video_id   = video_id
        self._velocity   = VelocityTracker()

        # Deferred label buffer — sits between record_builder and data_bus
        dl_cfg = deferred_cfg or {}
        self._label_buffer = DeferredLabelBuffer(
            confirm_window=dl_cfg.get("confirm_window", 12),
            cooldown_window=dl_cfg.get("cooldown_window", 8),
            max_buffer_size=dl_cfg.get("max_buffer_size", 300),
            enter_threshold=dl_cfg.get("enter_threshold", 0.5),
            exit_threshold=dl_cfg.get("exit_threshold", 0.3),
            min_segment_frames=dl_cfg.get("min_segment_frames", 8),
            merge_gap_frames=dl_cfg.get("merge_gap_frames", 5),
        )

    # ── QThread entry point ───────────────────────────────────────────────────

    def run(self) -> None:
        logger.info("▶️  Pipeline thread started")
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                logger.warning("⚠️  Stream ended or invalid frame")
                self.stream_ended.emit()
                break

            # ── Domain pipeline ──────────────────────────────────────────────
            persons = self._detector.detect(frame)
            persons = self._registry.merge(persons)
            persons = self._processor.process(persons)
            self._registry.persist(persons)

            # ── Data collection ──────────────────────────────────────────────
            ts          = time.time()
            current_fps = self._fps.tick(ts)

            if self._frame_no > 0 and self._frame_no % 30 == 0:
                logger.debug("FPS: {:.1f} | Tracked: {}", current_fps, len(persons))

            for record in build_frame_records(
                persons, self._frame_no, ts, current_fps,
                session_id=self._session_id,
                video_id=self._video_id,
                velocity_tracker=self._velocity,
            ):
                for finalized in self._label_buffer.push(record):
                    self._data_bus.on_frame(finalized)
            self._frame_no += 1

            # ── Render + ViewModels ──────────────────────────────────────────
            annotated = self._renderer.render(frame, persons)
            self.frame_ready.emit(self._presenter.build_video_vm(annotated))
            self.stats_ready.emit(self._presenter.build_dashboard_vm(persons, current_fps))

        # ── End-of-stream: flush remaining buffered frames ────────────────
        for finalized in self._label_buffer.flush_all():
            self._data_bus.on_frame(finalized)

        logger.info("⏹️  Pipeline thread stopped")

    # ── Graceful shutdown ─────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the run loop to exit after the current frame."""
        self._running = False
