"""
src/presentation/widgets/stats_panel.py

StatsPanel — fixed-width left panel displaying session statistics.

Responsibilities (SRP):
  - Receive DashboardVM from the Presenter
  - Render tracking counts, per-exercise totals, per-ID breakdown, feedback
  - Fixed width (config-driven); grows vertically with QScrollArea
  - No domain logic, no OpenCV, no direct Counter access

Design:
  - Uses a QScrollArea so it never overflows if many persons are tracked
  - Content is rebuilt each frame: a QWidget with a QVBoxLayout
    containing QLabel instances styled via QSS object names
  - The panel width is pinned with setFixedWidth; Qt's size policies
    prevent it from stretching regardless of window resize
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..viewmodels import DashboardVM

_APP_NAME = "AI Fitness Tracker"


class StatsPanel(QWidget):
    """
    Left panel: fixed width, scrollable stats.

    Call update_stats(DashboardVM) each frame to refresh the display.
    """

    def __init__(self, config: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatsPanel")

        panel_width: int = config.get("panel_width", 320)
        self._panel_title: str = config.get("panel_title", "CLASS STATS")
        self._ids_per_line: int = config.get("ids_per_line_limit", 34)

        self.setFixedWidth(panel_width)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self._setup_ui()

    # ── IStatsDisplay ─────────────────────────────────────────────────────────

    def update_stats(self, vm: DashboardVM) -> None:
        """Replace the scrollable content with a fresh render of *vm*."""
        self._rebuild_content(vm)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Title ────────────────────────────────────────────────────────────
        title = QLabel(self._panel_title)
        title.setObjectName("PanelTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        root_layout.addWidget(title)

        root_layout.addWidget(self._make_divider())

        # ── Scrollable content ────────────────────────────────────────────────
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        # Placeholder content until first frame arrives
        self._content_widget = QWidget()
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        self._content_layout.addStretch()

        self._scroll.setWidget(self._content_widget)
        root_layout.addWidget(self._scroll, stretch=1)

        # ── Footer ───────────────────────────────────────────────────────────
        root_layout.addWidget(self._make_divider())
        footer = QLabel(_APP_NAME)
        footer.setObjectName("FooterLabel")
        root_layout.addWidget(footer)

    # ── Content builder (called every frame) ──────────────────────────────────

    def _rebuild_content(self, vm: DashboardVM) -> None:
        """Discard old content widget, build a new one from *vm*."""
        new_content = QWidget()
        layout = QVBoxLayout(new_content)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(0)

        # Tracked count
        tracked_lbl = QLabel(f"Tracked persons: {vm.num_tracked}")
        tracked_lbl.setObjectName("TrackedLabel")
        layout.addWidget(tracked_lbl)
        layout.addWidget(self._make_divider())

        if not vm.exercise_totals:
            no_data = QLabel("No exercise data yet")
            no_data.setObjectName("NoDataLabel")
            layout.addWidget(no_data)
        else:
            for ex_name, total in vm.exercise_totals.items():
                # Exercise header: name + total
                ex_hdr = QLabel(f"{ex_name.upper()}:  {total} reps")
                ex_hdr.setObjectName("ExerciseTitle")
                layout.addWidget(ex_hdr)

                # Per-ID breakdown
                by_id = vm.exercise_by_id.get(ex_name, {})
                if by_id:
                    parts = [f"id{tid}:{reps}" for tid, reps in sorted(by_id.items())]
                    breakdown_lines = self._wrap_parts(parts, self._ids_per_line)
                    for line in breakdown_lines:
                        lbl = QLabel(f"  ({line})")
                        lbl.setObjectName("ExerciseBreakdown")
                        layout.addWidget(lbl)

                layout.addWidget(self._make_divider())

            # Per-person feedback (real-time coaching)
            for person_vm in vm.persons:
                for ex_vm in person_vm.exercises.values():
                    if ex_vm.feedback:
                        fb_lbl = QLabel(
                            f"  ID {person_vm.track_id} › {ex_vm.feedback}"
                        )
                        fb_lbl.setObjectName("FeedbackLabel")
                        fb_lbl.setWordWrap(True)
                        layout.addWidget(fb_lbl)

        layout.addStretch()

        # Swap widget in the scroll area
        old = self._scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        self._scroll.setWidget(new_content)
        self._content_widget = new_content

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_divider() -> QFrame:
        line = QFrame()
        line.setObjectName("SectionDivider")
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Plain)
        return line

    @staticmethod
    def _wrap_parts(parts: list[str], max_chars: int) -> list[str]:
        """Wrap a list of short strings into lines of at most *max_chars* chars."""
        lines, current = [], ""
        for part in parts:
            candidate = (current + ", " + part) if current else part
            if len(candidate) > max_chars and current:
                lines.append(current)
                current = part
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines
