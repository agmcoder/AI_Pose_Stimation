"""
src/presentation/widgets/exercise_checklist.py

ExerciseChecklist — dynamic checkbox list for toggling exercises.

Responsibilities (SRP):
  - Receive ExerciseChecklistVM from the Presenter
  - Render checkboxes for each available exercise
  - Emit exercises_changed signal when the user toggles a checkbox
  - No domain logic, no pipeline access

Design:
  - Only rebuilds checkboxes when the set of available exercises changes
    (not every frame) to avoid checkbox flickering
  - The signal carries a set[str] of currently checked exercise names
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..viewmodels import ExerciseChecklistVM


class ExerciseChecklist(QWidget):
    """
    Checkbox list of available exercises.

    Emits exercises_changed(set) when the user toggles any checkbox.
    Call update_checklist(ExerciseChecklistVM) to sync state.
    """

    exercises_changed = Signal(set)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ExerciseChecklist")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 4, 8, 4)
        self._layout.setSpacing(4)

        title = QLabel("EXERCISES")
        title.setObjectName("ChecklistTitle")
        self._layout.addWidget(title)

        self._checkboxes: dict[str, QCheckBox] = {}
        self._known_names: tuple[str, ...] = ()

    def update_checklist(self, vm: ExerciseChecklistVM) -> None:
        """Sync checkboxes with the ViewModel. Only rebuilds if items changed."""
        incoming_names = tuple(item.name for item in vm.items)

        if incoming_names != self._known_names:
            self._rebuild_checkboxes(vm)
            return

        for item in vm.items:
            cb = self._checkboxes.get(item.name)
            if cb is not None and cb.isChecked() != item.is_active:
                cb.blockSignals(True)
                cb.setChecked(item.is_active)
                cb.blockSignals(False)

    def _rebuild_checkboxes(self, vm: ExerciseChecklistVM) -> None:
        """Full rebuild — only when the available exercise set changes."""
        for cb in self._checkboxes.values():
            self._layout.removeWidget(cb)
            cb.deleteLater()
        self._checkboxes.clear()

        for item in vm.items:
            cb = QCheckBox(item.display_name)
            cb.setObjectName("ExerciseCheckbox")
            cb.setChecked(item.is_active)
            cb.toggled.connect(self._on_toggled)
            self._layout.addWidget(cb)
            self._checkboxes[item.name] = cb

        self._known_names = tuple(item.name for item in vm.items)

    def _on_toggled(self, _checked: bool) -> None:
        """Collect all checked names and emit."""
        active = {name for name, cb in self._checkboxes.items() if cb.isChecked()}
        self.exercises_changed.emit(active)
