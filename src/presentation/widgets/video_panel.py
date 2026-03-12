"""
src/presentation/widgets/video_panel.py

VideoPanel — right-side widget that displays the annotated video stream.

Responsibilities (SRP):
  - Receive VideoFrameVM from the Presenter
  - Convert the numpy frame to QPixmap via FrameToQImageAdapter
  - Scale the pixmap to fill available space (keeping aspect ratio)
  - That's it — no domain logic, no stats, no OpenCV layout

IFrameDisplay contract ensures this widget is substitutable
(LSP) and the Presenter only calls update_frame (ISP).
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..adapters import FrameToQImageAdapter
from ..viewmodels import VideoFrameVM


class VideoPanel(QWidget):
    """
    Right panel: scalable video display.

    The QLabel fills the entire panel and scales the QPixmap
    to fit without distortion (KeepAspectRatio + SmoothTransformation).
    Background is pure black so letterboxing is invisible.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("VideoPanel")
        self._setup_ui()

    # ── IFrameDisplay ─────────────────────────────────────────────────────────

    def update_frame(self, vm: VideoFrameVM) -> None:
        """Called each frame. Converts numpy → QPixmap and updates the label."""
        pixmap = FrameToQImageAdapter.convert(vm.frame)
        if pixmap.isNull():
            return
        scaled = pixmap.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        self._label = QLabel(self)
        self._label.setObjectName("VideoLabel")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._label)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
