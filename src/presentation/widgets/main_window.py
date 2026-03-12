"""
src/presentation/widgets/main_window.py

ApplicationWindow — the top-level Qt window that composes the two panels.

Layout:
  ┌──────────────────── QMainWindow ─────────────────────────────┐
  │  StatsPanel (fixed width)  │  VideoPanel (expanding)         │
  │  • tracking counts         │  • annotated camera stream      │
  │  • rep totals              │  • scales with aspect ratio     │
  │  • per-ID breakdown        │                                 │
  │  • feedback messages       │                                 │
  └──────────────────────────────────────────────────────────────┘

Responsibilities (SRP):
  - Compose the two child widgets inside a QSplitter
  - Configure window geometry, title, minimum size from config
  - Load and apply the QSS stylesheet
  - Expose video_panel and stats_panel as properties so the
    orchestration layer can call update_* on them (Facade)

IApplicationWindow contract gives an abstraction point so future
frontends (e.g. web dashboard, remote viewer) can replace this
class without touching the main loop.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QSplitter, QStatusBar, QWidget

from .stats_panel import StatsPanel
from .video_panel import VideoPanel

_QSS_PATH = Path(__file__).resolve().parent.parent / "styles" / "dark_theme.qss"


class ApplicationWindow(QMainWindow):
    """
    Main application window with a two-panel split layout.

    config keys expected (matches config/app.yml structure):
      window.name          → window title
      window.initial_width → initial window width in px
      window.initial_height → initial window height in px
      window.min_width     → minimum window width in px
      window.min_height    → minimum window height in px
      dashboard.panel_width    → fixed left panel width in px
      dashboard.panel_title    → header text of the stats panel
      dashboard.ids_per_line_limit → char limit for ID breakdown lines
    """

    def __init__(
        self,
        window_cfg: dict,
        dashboard_cfg: dict,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._window_cfg = window_cfg
        self._dashboard_cfg = dashboard_cfg

        self._setup_window()
        self._load_stylesheet()
        self._setup_layout()

    # ── Properties (Facade: expose children without exposing internals) ────────

    @property
    def video_panel(self) -> VideoPanel:
        return self._video_panel

    @property
    def stats_panel(self) -> StatsPanel:
        return self._stats_panel

    # ── IApplicationWindow ────────────────────────────────────────────────────

    def show(self) -> None:           # type: ignore[override]
        super().show()

    def is_open(self) -> bool:
        return self.isVisible()

    def close(self) -> None:          # type: ignore[override]
        super().close()

    # ── Setup helpers ─────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        cfg = self._window_cfg
        self.setWindowTitle(cfg.get("name", "AI Fitness Tracker"))
        self.setMinimumSize(
            cfg.get("min_width", 860),
            cfg.get("min_height", 400),
        )
        self.resize(
            cfg.get("initial_width", 1280),
            cfg.get("initial_height", 720),
        )

        # Status bar for future use (fps counter, session info…)
        self.setStatusBar(QStatusBar(self))

    def _load_stylesheet(self) -> None:
        if _QSS_PATH.exists():
            self.setStyleSheet(_QSS_PATH.read_text(encoding="utf-8"))

    def _setup_layout(self) -> None:
        # Create the two panels
        self._stats_panel = StatsPanel(self._dashboard_cfg, parent=self)
        self._video_panel = VideoPanel(parent=self)

        # QSplitter: horizontal, left panel not collapsible
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setHandleWidth(0)          # invisible handle — fixed panel
        splitter.setChildrenCollapsible(False)

        splitter.addWidget(self._stats_panel)
        splitter.addWidget(self._video_panel)

        # Pin panel width and prevent interactive resizing of the left widget
        panel_w = self._dashboard_cfg.get("panel_width", 320)
        splitter.setSizes([panel_w, self.width() - panel_w])
        splitter.setStretchFactor(0, 0)   # stats panel: no stretch
        splitter.setStretchFactor(1, 1)   # video panel: all stretch

        self.setCentralWidget(splitter)
