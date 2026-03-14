"""
src/presentation/widgets/stats_panel.py

StatsPanel — fixed-width left panel displaying session statistics.

Responsibilities (SRP):
  - Receive DashboardVM from the Presenter
  - Render tracking counts, per-exercise totals, per-ID breakdown, feedback
  - Fixed width (config-driven); grows vertically with QScrollArea
  - No domain logic, no OpenCV, no direct Counter access

Performance:
  - Structural fingerprint guards a full widget rebuild (rare: only when
    the set of tracked persons or active exercises changes).
  - On most frames the structure is identical, so only QLabel.setText()
    is called — zero Qt object allocation/destruction on the hot path.
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

        # ── Label caches (populated on rebuild, reused on update) ─────────────
        self._cached_structure: tuple = ()
        self._header_lbl: QLabel | None = None
        self._ex_total_lbls:     dict[str, QLabel] = {}   # ex_name → "SQUAT: N reps"
        self._ex_breakdown_lbls: dict[str, QLabel] = {}   # ex_name → breakdown text
        self._feedback_lbls:     dict[int, QLabel] = {}   # track_id → feedback text

        self._setup_ui()

    # ── Public API ────────────────────────────────────────────────────────────

    def update_stats(self, vm: DashboardVM) -> None:
        """Replace or refresh content based on structural change detection."""
        structure = self._fingerprint(vm)
        if structure != self._cached_structure:
            self._rebuild_content(vm)
            self._cached_structure = structure
        else:
            self._update_content(vm)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        title = QLabel(self._panel_title)
        title.setObjectName("PanelTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        root_layout.addWidget(title)

        root_layout.addWidget(self._make_divider())

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._content_widget = QWidget()
        placeholder = QVBoxLayout(self._content_widget)
        placeholder.addStretch()
        self._scroll.setWidget(self._content_widget)
        root_layout.addWidget(self._scroll, stretch=1)

        root_layout.addWidget(self._make_divider())
        footer = QLabel(_APP_NAME)
        footer.setObjectName("FooterLabel")
        root_layout.addWidget(footer)

    # ── Hot path: update text only (zero Qt allocations) ─────────────────────

    def _update_content(self, vm: DashboardVM) -> None:
        if self._header_lbl is not None:
            self._header_lbl.setText(
                f"Tracked persons: {vm.num_tracked}    |    FPS: {vm.current_fps:.1f}"
            )

        for ex_name, lbl in self._ex_total_lbls.items():
            total = vm.exercise_totals.get(ex_name, 0)
            lbl.setText(f"{ex_name.upper()}:  {total} reps")

        for ex_name, lbl in self._ex_breakdown_lbls.items():
            by_id = vm.exercise_by_id.get(ex_name, {})
            if by_id:
                parts = [f"id{tid}:{reps}" for tid, reps in sorted(by_id.items())]
                lines = self._wrap_parts(parts, self._ids_per_line)
                lbl.setText("\n".join(f"  ({ln})" for ln in lines))
            else:
                lbl.setText("")

        for person_vm in vm.persons:
            lbl = self._feedback_lbls.get(person_vm.track_id)
            if lbl is not None:
                fb = " | ".join(
                    s.feedback for s in person_vm.exercises.values() if s.feedback
                )
                lbl.setText(f"  ID {person_vm.track_id} › {fb}" if fb else "")

    # ── Cold path: full rebuild on structural change ───────────────────────────

    def _rebuild_content(self, vm: DashboardVM) -> None:
        """Discard old content widget, build a fresh one, cache label refs."""
        self._ex_total_lbls.clear()
        self._ex_breakdown_lbls.clear()
        self._feedback_lbls.clear()
        self._header_lbl = None

        new_content = QWidget()
        layout = QVBoxLayout(new_content)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        self._header_lbl = QLabel(
            f"Tracked persons: {vm.num_tracked}    |    FPS: {vm.current_fps:.1f}"
        )
        self._header_lbl.setObjectName("TrackedLabel")
        layout.addWidget(self._header_lbl)
        layout.addWidget(self._make_divider())

        # ── Per-exercise ───────────────────────────────────────────────────────
        if not vm.exercise_totals:
            no_data = QLabel("No exercise data yet")
            no_data.setObjectName("NoDataLabel")
            layout.addWidget(no_data)
        else:
            for ex_name, total in vm.exercise_totals.items():
                title_lbl = QLabel(f"{ex_name.upper()}:  {total} reps")
                title_lbl.setObjectName("ExerciseTitle")
                layout.addWidget(title_lbl)
                self._ex_total_lbls[ex_name] = title_lbl

                by_id = vm.exercise_by_id.get(ex_name, {})
                parts = [f"id{tid}:{reps}" for tid, reps in sorted(by_id.items())]
                lines = self._wrap_parts(parts, self._ids_per_line)
                bd_lbl = QLabel("\n".join(f"  ({ln})" for ln in lines))
                bd_lbl.setObjectName("ExerciseBreakdown")
                bd_lbl.setWordWrap(True)
                layout.addWidget(bd_lbl)
                self._ex_breakdown_lbls[ex_name] = bd_lbl

                layout.addWidget(self._make_divider())

            # ── Per-person feedback ──────────────────────────────────────────
            for person_vm in vm.persons:
                fb = " | ".join(
                    s.feedback for s in person_vm.exercises.values() if s.feedback
                )
                fb_lbl = QLabel(f"  ID {person_vm.track_id} › {fb}" if fb else "")
                fb_lbl.setObjectName("FeedbackLabel")
                fb_lbl.setWordWrap(True)
                layout.addWidget(fb_lbl)
                self._feedback_lbls[person_vm.track_id] = fb_lbl

        layout.addStretch()

        old = self._scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        self._scroll.setWidget(new_content)
        self._content_widget = new_content

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fingerprint(vm: DashboardVM) -> tuple:
        """Cheap structural hash: exercise names + tracked person IDs."""
        return (
            tuple(sorted(vm.exercise_totals.keys())),
            tuple(sorted(p.track_id for p in vm.persons)),
        )

    @staticmethod
    def _make_divider() -> QFrame:
        line = QFrame()
        line.setObjectName("SectionDivider")
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Plain)
        return line

    @staticmethod
    def _wrap_parts(parts: list[str], max_chars: int) -> list[str]:
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
