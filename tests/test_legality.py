from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from robotrace_shortcut_lab.course import load_course
from robotrace_shortcut_lab.legality import (
    WhiteLineContactEvaluator,
    load_vehicle_footprint,
    resample_swept_poses,
    solve_contact_progress,
)
from robotrace_shortcut_lab.legal_planner import run_legal_global_mode
from robotrace_shortcut_lab.model import (
    BoardBoundary,
    Course,
    PlannerConfig,
    VehicleFootprint,
)


def _course_from_xy(x_mm: list[float], y_mm: list[float]) -> Course:
    x = np.asarray(x_mm, dtype=np.float32)
    y = np.asarray(y_mm, dtype=np.float32)
    distance = np.concatenate(
        (np.zeros(1), np.cumsum(np.hypot(np.diff(x), np.diff(y))))
    ).astype(np.float32)
    return Course("legality_test", distance, x, y)


def _confirmed_rectangle(half_length: float = 20.0, half_width: float = 20.0):
    vertices = np.array(
        [
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width],
        ],
        dtype=np.float32,
    )
    return VehicleFootprint(
        (vertices,),
        "試験用: 原点は中心、+X前方、+Y左方",
        half_length,
        half_length,
        half_width,
        half_width,
        0.0,
        "合成テスト外形",
        True,
    )


class SweptFootprintLegalityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PlannerConfig()
        self.footprint = _confirmed_rectangle()

    def test_obvious_black_floor_jump_is_illegal(self) -> None:
        course = _course_from_xy(
            [0.0, 100.0, 500.0, 500.0, 100.0, 0.0],
            [0.0, 0.0, 0.0, 200.0, 200.0, 200.0],
        )
        evaluator = WhiteLineContactEvaluator(course, self.footprint, self.config)
        result = evaluator.evaluate(
            np.array([50.0, 50.0], dtype=np.float32),
            np.array([0.0, 200.0], dtype=np.float32),
            np.array([np.pi / 2.0, np.pi / 2.0], dtype=np.float32),
        )
        self.assertFalse(result.legal)
        self.assertGreaterEqual(result.detachment_count, 1)
        self.assertIn("白線完全離脱", result.violation)

    def test_nearby_line_transfer_with_simultaneous_contact_is_legal(self) -> None:
        course = _course_from_xy(
            [0.0, 200.0, 500.0, 500.0, 200.0, 0.0],
            [0.0, 0.0, 0.0, 30.0, 30.0, 30.0],
        )
        evaluator = WhiteLineContactEvaluator(course, self.footprint, self.config)
        result = evaluator.evaluate(
            np.array([50.0, 150.0], dtype=np.float32),
            np.array([0.0, 30.0], dtype=np.float32),
            np.array([0.0, np.arctan2(30.0, 100.0)], dtype=np.float32),
            start_segment=0,
            end_segment=4,
        )
        self.assertTrue(result.legal, result.violation)
        self.assertEqual(result.detachment_count, 0)
        self.assertGreaterEqual(int(np.max(result.simultaneous_line_count)), 2)
        self.assertGreaterEqual(result.line_switch_count, 1)
        self.assertGreaterEqual(
            int(np.max(result.future_contact_count + result.past_contact_count)), 1
        )
        self.assertTrue(
            all(
                int(selected) in segments
                for selected, segments in zip(
                    result.source_progress_index,
                    result.contact_segments,
                    strict=True,
                )
            )
        )

    def test_intermediate_detachment_is_detected_between_legal_endpoints(self) -> None:
        course = _course_from_xy(
            [0.0, 100.0, 500.0, 500.0, 100.0, 0.0],
            [0.0, 0.0, 0.0, 200.0, 200.0, 200.0],
        )
        evaluator = WhiteLineContactEvaluator(course, self.footprint, self.config)
        result = evaluator.evaluate(
            np.array([50.0, 50.0], dtype=np.float32),
            np.array([0.0, 200.0], dtype=np.float32),
            np.array([np.pi / 2.0, np.pi / 2.0], dtype=np.float32),
        )
        self.assertTrue(result.contact_segments[0])
        self.assertTrue(result.contact_segments[-1])
        self.assertTrue(any(not segments for segments in result.contact_segments[1:-1]))
        self.assertLessEqual(float(np.max(np.diff(result.pose_distance_mm))), 2.001)

    def test_crossing_dp_selects_monotonic_contact_sequence(self) -> None:
        progress = solve_contact_progress(
            ((0,), (0, 10), (10,), (10, 11)),
            self.config,
            start_segment=0,
            end_segment=11,
        )
        self.assertIsNotNone(progress)
        assert progress is not None
        np.testing.assert_array_equal(progress, np.array([0, 10, 10, 11]))
        self.assertTrue(np.all(np.diff(progress) >= 0.0))

    def test_contact_dp_rejects_return_to_past_line(self) -> None:
        progress = solve_contact_progress(((10,), (9,)), self.config)
        self.assertIsNone(progress)

    def test_contact_dp_cannot_assign_uncontacted_interpolated_segment(self) -> None:
        progress = solve_contact_progress(
            ((0,), (0,), (10,), (10,)),
            self.config,
            start_segment=0,
            end_segment=10,
        )
        self.assertIsNone(progress)

    def test_pose_sampling_obeys_distance_and_yaw_limits(self) -> None:
        distance, _, _, yaw = resample_swept_poses(
            np.array([0.0, 10.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([0.0, np.deg2rad(10.0)], dtype=np.float32),
            self.config,
        )
        self.assertLessEqual(float(np.max(np.diff(distance))), 2.001)
        self.assertLessEqual(float(np.max(np.rad2deg(np.diff(yaw)))), 1.001)

    def test_confirmed_board_rejects_vehicle_polygon_outside(self) -> None:
        course = _course_from_xy([0.0, 100.0], [0.0, 0.0])
        boundary = BoardBoundary(((0.0, 100.0, -30.0, 30.0),), "合成板", True)
        evaluator = WhiteLineContactEvaluator(
            course, self.footprint, self.config, boundary
        )
        result = evaluator.evaluate(
            np.array([50.0, 95.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
        )
        self.assertFalse(result.legal)
        self.assertIn("板外", result.violation)


class VehicleFootprintGateTests(unittest.TestCase):
    def test_unconfirmed_or_missing_footprint_is_not_confirmed(self) -> None:
        self.assertFalse(load_vehicle_footprint().confirmed)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertFalse(
                load_vehicle_footprint(Path(temp_dir) / "missing.json").confirmed
            )

    def test_unconfirmed_footprint_forces_4471_fallback(self) -> None:
        course = load_course("data/courses/normalized/2025alljapan.tsv")
        local, result, _ = run_legal_global_mode(
            course,
            "embedded-lite",
            PlannerConfig(),
        )
        self.assertFalse(result.legal)
        self.assertTrue(result.fallback_used)
        self.assertAlmostEqual(
            result.adopted.metrics.predicted_time_s,
            local.best.metrics.predicted_time_s,
            places=9,
        )
        self.assertAlmostEqual(result.adopted.metrics.predicted_time_s, 4.4709220381, places=6)


if __name__ == "__main__":
    unittest.main()
