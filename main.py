"""
main.py — AI Fitness Tracker (PySide6 edition)

Orchestration layer: wires domain pipeline with the presentation layer.
No domain logic lives here, only wiring and clean shutdown.

Architecture:
  QApplication     — Qt event loop owner
  PipelineThread   — background thread: capture → detect → process → render
                     emits frame_ready / stats_ready signals to the main thread
  ApplicationWindow — two-panel Qt window (stats left, video right)
                     only receives ViewModels and updates widget text/pixmap

Clean shutdown:
  qt_app.aboutToQuit guarantees pipeline.stop() + data_bus.close() + cap.release()
  even if the user closes with the window X button or Ctrl+Q.
"""
import sys

import cv2
import yaml
from loguru import logger
from PySide6.QtWidgets import QApplication

from src.data_collection.bus import DataCollectionBus
from src.data_collection.factory import create_collectors
from src.models.yolo_pose import YoloPoseDetector
from src.pipeline.device_selector import select_device
from src.pipeline.frame_processor import FrameProcessor
from src.pipeline.person_registry import PersonRegistry
from src.pipeline.pipeline_thread import PipelineThread
from src.presentation.presenter import DashboardPresenter
from src.presentation.widgets.main_window import ApplicationWindow
from src.tracking.exercise_counter import ExerciseCounter
from src.visualization.renderer import Renderer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    # ── Configuration ─────────────────────────────────────────────────────────
    app_cfg       = load_config("config/app.yml")["app"]
    pipe_cfg      = load_config("config/app.yml")["pipeline"]
    model_cfg     = load_config("config/model.yml")["model"]
    ex_cfg        = load_config("config/exercises.yml")["exercises"]
    win_cfg       = app_cfg["window"]
    dashboard_cfg = app_cfg["dashboard"]

    # ── Domain pipeline ───────────────────────────────────────────────────────
    device    = select_device(model_cfg["device"])
    counter   = ExerciseCounter()
    detector  = YoloPoseDetector(model_cfg, device)
    processor = FrameProcessor(ex_cfg, pipe_cfg["num_workers"], counter)
    registry  = PersonRegistry()
    renderer  = Renderer()

    # ── Data collection ───────────────────────────────────────────────────────
    collectors, session_id = create_collectors(app_cfg.get("data_collection", {}))
    data_bus   = DataCollectionBus(collectors)

    # ── Qt application + window ───────────────────────────────────────────────
    qt_app    = QApplication(sys.argv)
    presenter = DashboardPresenter(counter)
    window    = ApplicationWindow(win_cfg, dashboard_cfg)
    window.show()

    # ── Exercise checklist wiring ─────────────────────────────────────────────
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

    # ── Pipeline thread ───────────────────────────────────────────────────────
    pipeline = PipelineThread(
        cap=cap,
        detector=detector,
        registry=registry,
        processor=processor,
        data_bus=data_bus,
        renderer=renderer,
        presenter=presenter,
        session_id=session_id,
        video_id=str(app_cfg["source"]),
    )

    # Qt queued connections: signals delivered on the main thread event loop
    pipeline.frame_ready.connect(window.video_panel.update_frame)
    pipeline.stats_ready.connect(window.stats_panel.update_stats)
    pipeline.stream_ended.connect(qt_app.quit)

    logger.info("🎥 Iniciando stream — cierra la ventana para salir")
    pipeline.start()

    # ── Clean shutdown ────────────────────────────────────────────────────────
    def on_quit() -> None:
        pipeline.stop()
        pipeline.wait()     # block until run() returns
        data_bus.close()
        cap.release()
        logger.info("✅ Recursos liberados correctamente")

    qt_app.aboutToQuit.connect(on_quit)

    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
