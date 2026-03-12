"""
src/presentation/interfaces.py

Segregated presentation interfaces (ISP) using typing.Protocol.

Using Protocol instead of ABC avoids the Qt metaclass conflict:
  - Qt widgets use an internal Shiboken metaclass
  - ABC uses ABCMeta
  - Multiple inheritance of both metaclasses raises TypeError

Protocol uses structural subtyping (duck typing), imposing no metaclass
on the concrete widget class. runtime_checkable enables isinstance() checks.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .viewmodels import VideoFrameVM, DashboardVM


@runtime_checkable
class IFrameDisplay(Protocol):
    """Single Responsibility: display one annotated video frame."""

    def update_frame(self, vm: VideoFrameVM) -> None: ...


@runtime_checkable
class IStatsDisplay(Protocol):
    """Single Responsibility: display aggregated session statistics."""

    def update_stats(self, vm: DashboardVM) -> None: ...


@runtime_checkable
class IApplicationWindow(Protocol):
    """Single Responsibility: manage the lifecycle of the main window."""

    def show(self) -> None: ...

    def is_open(self) -> bool: ...

    def close(self) -> None: ...
