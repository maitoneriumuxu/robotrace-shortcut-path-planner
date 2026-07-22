from __future__ import annotations

from dataclasses import replace
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
        full_footprint_components_mm=(vertices,),
        contact_witness_components_mm=(vertices,),
        origin_definition="試験用: 原点は中心、+X前方、+Y左方",
        board_clearance_radius_mm=125.0,
        safety_margin_mm=0.0,
        source="合成テスト外形",
        full_footprint_source="合成テスト",
        contact_witness_source="合成テスト",
        design_confirmed=True,
        as_built_confirmed=True,
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

    def test_nearby_line_transfer_is_illegal_when_intermediate_lines_are_skipped(self) -> None:
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
        self.assertFalse(result.legal)
        self.assertEqual(result.detachment_count, 0)
        self.assertGreaterEqual(int(np.max(result.simultaneous_line_count)), 2)
        self.assertFalse(result.all_line_segments_covered)
        self.assertGreater(result.unvisited_segment_count, 0)
        self.assertIn("未通過LINE", result.violation)

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

    def test_crossing_dp_rejects_simultaneous_future_line_jump(self) -> None:
        progress = solve_contact_progress(
            ((0,), (0, 10), (10,), (10, 11)),
            self.config,
            start_segment=0,
            end_segment=11,
        )
        self.assertIsNone(progress)

    def test_all_line_radial_prefilter_rejects_unreachable_middle_segment(self) -> None:
        course = _course_from_xy(
            [0.0, 300.0, 300.0, 0.0],
            [0.0, 0.0, 300.0, 300.0],
        )
        evaluator = WhiteLineContactEvaluator(course, self.footprint, self.config)
        unreachable = evaluator.count_radially_unreachable_segments(
            np.array([50.0, 50.0], dtype=np.float32),
            np.array([0.0, 300.0], dtype=np.float32),
            0,
            2,
        )
        self.assertGreater(unreachable, 0)

    def test_contact_dp_accepts_every_line_segment_in_order(self) -> None:
        progress = solve_contact_progress(
            ((0,), (0, 1), (1, 2), (2, 3), (3,)),
            self.config,
            start_segment=0,
            end_segment=3,
        )
        self.assertIsNotNone(progress)
        assert progress is not None
        self.assertEqual(float(progress[0]), 0.0)
        self.assertEqual(float(progress[-1]), 3.0)
        self.assertTrue(np.all((np.diff(progress) >= 0.0) & (np.diff(progress) <= 1.0)))

    def test_straight_path_reports_every_required_line_segment_covered(self) -> None:
        course = _course_from_xy(
            [0.0, 25.0, 50.0, 75.0, 100.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        )
        evaluator = WhiteLineContactEvaluator(course, self.footprint, self.config)
        result = evaluator.evaluate(
            course.x_mm,
            course.y_mm,
            np.zeros(course.point_count, dtype=np.float32),
            start_segment=0,
            end_segment=course.point_count - 2,
        )
        self.assertTrue(result.legal, result.violation)
        self.assertTrue(result.all_line_segments_covered)
        self.assertEqual(result.unvisited_segment_count, 0)

    def test_contact_dp_can_advance_over_dense_segments_only_when_all_are_contacted(self) -> None:
        progress = solve_contact_progress(
            ((0,), (0, 1, 2, 3, 4), (4,)),
            self.config,
            start_segment=0,
            end_segment=4,
        )
        self.assertIsNotNone(progress)
        assert progress is not None
        self.assertEqual(float(progress[0]), 0.0)
        self.assertEqual(float(progress[-1]), 4.0)
        self.assertGreater(float(np.max(np.diff(progress))), 1.0)

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

    def test_contact_does_not_use_125mm_board_circle(self) -> None:
        course = _course_from_xy([0.0, 300.0], [0.0, 0.0])
        evaluator = WhiteLineContactEvaluator(course, self.footprint, self.config)
        result = evaluator.evaluate(
            np.array([50.0, 250.0], dtype=np.float32),
            np.array([80.0, 80.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
        )
        self.assertFalse(result.legal)
        self.assertGreater(result.detachment_count, 0)


class VehicleFootprintGateTests(unittest.TestCase):
    def test_transverse_bar_is_loaded_with_separate_confirmation_states(self) -> None:
        footprint = load_vehicle_footprint()
        self.assertTrue(footprint.design_confirmed)
        self.assertFalse(footprint.as_built_confirmed)
        self.assertFalse(footprint.full_footprint_components_mm)
        self.assertEqual(len(footprint.contact_witness_components_mm), 1)
        vertices = footprint.contact_witness_components_mm[0]
        self.assertEqual(float(np.min(vertices[:, 1])), -100.0)
        self.assertEqual(float(np.max(vertices[:, 1])), 100.0)
        self.assertLessEqual(float(np.max(np.hypot(vertices[:, 0], vertices[:, 1]))), 125.0)
        self.assertEqual(footprint.board_clearance_radius_mm, 125.0)

    def test_transverse_bar_is_centered_at_origin_and_ten_mm_thick(self) -> None:
        vertices = load_vehicle_footprint().contact_witness_components_mm[0]
        self.assertEqual(float(np.min(vertices[:, 0])), -5.0)
        self.assertEqual(float(np.max(vertices[:, 0])), 5.0)
        np.testing.assert_allclose(np.mean(vertices, axis=0), np.zeros(2), atol=0.0)

    def test_board_clearance_radius_does_not_change_white_line_contact(self) -> None:
        course = _course_from_xy([0.0, 100.0], [0.0, 0.0])
        footprint = load_vehicle_footprint()
        path_x = np.array([0.0, 100.0], dtype=np.float32)
        path_y = np.zeros(2, dtype=np.float32)
        yaw = np.zeros(2, dtype=np.float32)
        normal = WhiteLineContactEvaluator(course, footprint, PlannerConfig()).evaluate(
            path_x, path_y, yaw, start_segment=0, end_segment=0
        )
        huge_circle = WhiteLineContactEvaluator(
            course,
            replace(footprint, board_clearance_radius_mm=10_000.0),
            PlannerConfig(),
        ).evaluate(path_x, path_y, yaw, start_segment=0, end_segment=0)
        self.assertEqual(normal.legal, huge_circle.legal)
        self.assertEqual(normal.contact_segments, huge_circle.contact_segments)

    def test_missing_footprint_is_not_design_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertFalse(
                load_vehicle_footprint(Path(temp_dir) / "missing.json").design_confirmed
            )

    def test_200mm_bar_detects_simultaneous_contact_but_not_skipped_line_legality(self) -> None:
        course = _course_from_xy(
            [0.0, 300.0, 300.0, 0.0],
            [0.0, 0.0, 180.0, 180.0],
        )
        evaluator = WhiteLineContactEvaluator(
            course, load_vehicle_footprint(), PlannerConfig()
        )
        result = evaluator.evaluate(
            np.array([100.0, 100.0], dtype=np.float32),
            np.array([0.0, 180.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
            start_segment=0,
            end_segment=2,
        )
        self.assertFalse(result.legal)
        self.assertGreater(int(np.max(result.simultaneous_line_count)), 1)
        self.assertGreater(result.unvisited_segment_count, 0)

    def test_unconfirmed_contact_witness_forces_4471_fallback(self) -> None:
        course = load_course("data/courses/normalized/2025alljapan.tsv")
        unconfirmed = replace(
            load_vehicle_footprint(),
            contact_witness_components_mm=(),
            design_confirmed=False,
        )
        local, result, _ = run_legal_global_mode(
            course,
            "embedded-lite",
            PlannerConfig(),
            footprint=unconfirmed,
        )
        self.assertFalse(result.legal)
        self.assertTrue(result.fallback_used)
        self.assertAlmostEqual(
            result.adopted.metrics.predicted_time_s,
            local.best.metrics.predicted_time_s,
            places=9,
        )
        self.assertAlmostEqual(result.adopted.metrics.predicted_time_s, 4.4709220381, places=6)

    def test_design_confirmed_reference_search_can_run(self) -> None:
        x = np.arange(0.0, 500.0 + 10.0, 10.0, dtype=np.float32)
        course = _course_from_xy(x.tolist(), np.zeros_like(x).tolist())
        compact = replace(
            PlannerConfig(),
            reference_anchor_limit=16,
            reference_edge_limit=32,
            legal_reference_edge_check_limit=16,
            legal_reference_top_k=2,
        )
        _, result, _ = run_legal_global_mode(
            course,
            "reference",
            compact,
            footprint=load_vehicle_footprint(),
        )
        self.assertTrue(result.legal)
        self.assertIn("設計上合法", result.legality_status)

    def test_as_built_false_is_never_reported_as_confirmed(self) -> None:
        x = np.arange(0.0, 500.0 + 10.0, 10.0, dtype=np.float32)
        course = _course_from_xy(x.tolist(), np.zeros_like(x).tolist())
        compact = replace(
            PlannerConfig(),
            embedded_anchor_limit=12,
            embedded_edge_limit=16,
            legal_embedded_edge_check_limit=8,
            embedded_top_k=2,
        )
        _, result, _ = run_legal_global_mode(
            course,
            "embedded-lite",
            compact,
            footprint=load_vehicle_footprint(),
        )
        self.assertNotIn("実車確認済み", result.legality_status)


if __name__ == "__main__":
    unittest.main()
