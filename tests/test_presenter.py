"""
tests/test_presenter.py

Unit tests for DashboardPresenter.

Tests verify that the Presenter correctly transforms domain objects
into ViewModels without any Qt or OpenCV dependency in this layer.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.types import ExerciseState, Keypoints, Person
from src.presentation.presenter import DashboardPresenter
from src.presentation.viewmodels import DashboardVM, VideoFrameVM
from src.tracking.exercise_counter import ExerciseCounter


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_person(track_id: int, reps: int = 0) -> Person:
    kps = Keypoints(
        coords=np.zeros((17, 2), dtype=np.float32),
        scores=np.ones(17, dtype=np.float32),
    )
    state = ExerciseState(
        name="squat",
        rep_count=reps,
        phase="up",
        angle=160.0,
        angles={"knee": 160.0, "hip": 170.0},
        feedback="",
        valid_down=False,
    )
    return Person(track_id=track_id, bbox=(0.0, 0.0, 100.0, 200.0),
                  keypoints=kps, exercises={"squat": state})


def _make_counter_with_reps() -> ExerciseCounter:
    counter = ExerciseCounter()
    counter.record(1, "squat")
    counter.record(1, "squat")
    counter.record(2, "squat")
    return counter


# ── Tests: build_video_vm ─────────────────────────────────────────────────────


class TestBuildVideoVM:
    def test_returns_video_frame_vm(self):
        counter = ExerciseCounter()
        presenter = DashboardPresenter(counter)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        vm = presenter.build_video_vm(frame)

        assert isinstance(vm, VideoFrameVM)

    def test_dimensions_match_frame(self):
        counter = ExerciseCounter()
        presenter = DashboardPresenter(counter)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        vm = presenter.build_video_vm(frame)

        assert vm.width == 640
        assert vm.height == 480

    def test_frame_reference_preserved(self):
        counter = ExerciseCounter()
        presenter = DashboardPresenter(counter)
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        vm = presenter.build_video_vm(frame)

        assert vm.frame is frame


# ── Tests: build_dashboard_vm ─────────────────────────────────────────────────


class TestBuildDashboardVM:
    def test_returns_dashboard_vm(self):
        counter = ExerciseCounter()
        presenter = DashboardPresenter(counter)

        vm = presenter.build_dashboard_vm([])

        assert isinstance(vm, DashboardVM)

    def test_num_tracked_matches_persons_list(self):
        counter = ExerciseCounter()
        presenter = DashboardPresenter(counter)
        persons = [_make_person(1), _make_person(2)]

        vm = presenter.build_dashboard_vm(persons)

        assert vm.num_tracked == 2

    def test_empty_persons_num_tracked_zero(self):
        counter = ExerciseCounter()
        presenter = DashboardPresenter(counter)

        vm = presenter.build_dashboard_vm([])

        assert vm.num_tracked == 0

    def test_exercise_totals_match_counter(self):
        counter = _make_counter_with_reps()
        presenter = DashboardPresenter(counter)

        vm = presenter.build_dashboard_vm([_make_person(1), _make_person(2)])

        assert vm.exercise_totals["squat"] == 3  # 2 + 1

    def test_exercise_by_id_matches_counter(self):
        counter = _make_counter_with_reps()
        presenter = DashboardPresenter(counter)

        vm = presenter.build_dashboard_vm([])

        assert vm.exercise_by_id["squat"][1] == 2
        assert vm.exercise_by_id["squat"][2] == 1

    def test_persons_viewmodels_built_correctly(self):
        counter = ExerciseCounter()
        presenter = DashboardPresenter(counter)
        persons = [_make_person(42)]

        vm = presenter.build_dashboard_vm(persons)

        assert len(vm.persons) == 1
        person_vm = vm.persons[0]
        assert person_vm.track_id == 42
        assert "squat" in person_vm.exercises

    def test_exercise_vm_fields(self):
        counter = ExerciseCounter()
        presenter = DashboardPresenter(counter)
        persons = [_make_person(1)]

        vm = presenter.build_dashboard_vm(persons)

        ex_vm = vm.persons[0].exercises["squat"]
        assert ex_vm.name == "squat"
        assert ex_vm.phase == "up"
        assert ex_vm.feedback == ""
        assert "knee" in ex_vm.angles
        assert "hip" in ex_vm.angles

    def test_no_domain_types_in_viewmodel(self):
        """ViewModels must not expose Person or ExerciseState."""
        from src.core.types import Person, ExerciseState
        counter = ExerciseCounter()
        presenter = DashboardPresenter(counter)
        persons = [_make_person(1)]

        vm = presenter.build_dashboard_vm(persons)

        for person_vm in vm.persons:
            assert not isinstance(person_vm, Person)
            for ex_vm in person_vm.exercises.values():
                assert not isinstance(ex_vm, ExerciseState)
