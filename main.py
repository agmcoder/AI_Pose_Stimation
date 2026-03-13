"""
main.py — AI Fitness Tracker (PySide6 edition)

Orchestration layer: wires domain pipeline with the new presentation layer.
No domain logic lives here, only wiring and the frame-tick callback.

Architecture:
  QApplication  — Qt event loop owner
  QTimer        — drives the frame-tick at maximum throughput
  DashboardPresenter — translates domain → ViewModels each tick
  ApplicationWindow  — two-panel Qt window (stats left, video right)

Clean shutdown:
  qt_app.aboutToQuit signal guarantees data_bus.close() + cap.release()
  even if the user closes with the window X button or Ctrl+Q.
"""
import sys
import time
from collections import deque

import cv2
import yaml
from loguru import logger
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from src.data_collection.bus import DataCollectionBus
from src.data_collection.factory import create_collectors
from src.data_collection.record_builder import build_frame_records
from src.models.yolo_pose import YoloPoseDetector
from src.pipeline.device_selector import select_device
from src.pipeline.frame_processor import FrameProcessor
from src.pipeline.person_registry import PersonRegistry
from src.presentation.presenter import DashboardPresenter
from src.presentation.widgets.main_window import ApplicationWindow
from src.tracking.exercise_counter import ExerciseCounter
from src.visualization.renderer import Renderer


# ── Config helpers ────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class FpsTracker:
    """Calculates moving average FPS over the last N frames."""

    def __init__(self, window_size: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window_size)

    def tick(self, current_time: float) -> float:
        self._times.append(current_time)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        if elapsed == 0:
            return 0.0
        return (len(self._times) - 1) / elapsed


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    # ── Configuration ─────────────────────────────────────────────────────────
    app_cfg       = load_config("config/app.yml")["app"]
    pipe_cfg      = load_config("config/app.yml")["pipeline"]
    model_cfg     = load_config("config/model.yml")["model"]
    ex_cfg        = load_config("config/exercises.yml")["exercises"]
    win_cfg       = app_cfg["window"]
    dashboard_cfg = app_cfg["dashboard"]

    # ── Domain pipeline (unchanged) ───────────────────────────────────────────
    device    = select_device(model_cfg["device"])
    counter   = ExerciseCounter()
    detector  = YoloPoseDetector(model_cfg, device)
    processor = FrameProcessor(ex_cfg, pipe_cfg["num_workers"], counter)
    registry  = PersonRegistry()
    renderer  = Renderer()              # draws skeletons / overlays on frames

    # ── Data collection (unchanged) ───────────────────────────────────────────
    collectors = create_collectors(app_cfg.get("data_collection", {}))
    data_bus   = DataCollectionBus(collectors)

    # ── Presentation layer ────────────────────────────────────────────────────
    qt_app    = QApplication(sys.argv)
    presenter = DashboardPresenter(counter)
    window    = ApplicationWindow(win_cfg, dashboard_cfg)
    window.show()

    # ── Exercise checklist wiring ──────────────────────────────────────────────
    checklist_vm = presenter.build_checklist_vm(
        processor.available_exercises(),
        processor.active_exercise_names(),
    )
    window.exercise_checklist.update_checklist(checklist_vm)
    window.exercise_checklist.exercises_changed.connect(processor.set_active_exercises)

    # ── Video capture ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(app_cfg["source"])
    if not cap.isOpened():
        logger.error("❌ No se pudo abrir la fuente de vídeo: {}", app_cfg["source"])
        sys.exit(1)

    logger.info("🎥 Iniciando stream — cierra la ventana para salir")
    frame_number = 0
    fps_tracker = FpsTracker(window_size=30)

    # ── Frame-tick callback ───────────────────────────────────────────────────

    def process_frame() -> None:
        nonlocal frame_number

        ok, frame = cap.read()
        if not ok:
            logger.warning("⚠️ Stream terminado o frame inválido — cerrando")
            qt_app.quit()
            return

        # Domain pipeline (unchanged logic)
        persons = detector.detect(frame)
        persons = registry.merge(persons)
        persons = processor.process(persons)
        registry.persist(persons)

        # Data collection & FPS
        timestamp = time.time()
        current_fps = fps_tracker.tick(timestamp)

        if frame_number > 0 and frame_number % 30 == 0:
            logger.debug("FPS: {:.1f} | Tracked: {}", current_fps, len(persons))

        for record in build_frame_records(persons, frame_number, timestamp, current_fps):
            data_bus.on_frame(record)
        frame_number += 1

        # Render skeleton overlays (OpenCV, on the frame only)
        annotated = renderer.render(frame, persons)

        # Build ViewModels and push to widgets
        window.video_panel.update_frame(presenter.build_video_vm(annotated))
        window.stats_panel.update_stats(presenter.build_dashboard_vm(persons, current_fps))

    # ── QTimer drives the loop ────────────────────────────────────────────────
    timer = QTimer()
    timer.timeout.connect(process_frame)
    timer.start(1)   # interval=1ms → as fast as cap.read() allows

    # ── Clean shutdown ────────────────────────────────────────────────────────
    def on_quit() -> None:
        timer.stop()
        data_bus.close()
        cap.release()
        logger.info("✅ Recursos liberados correctamente")

    qt_app.aboutToQuit.connect(on_quit)

    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
