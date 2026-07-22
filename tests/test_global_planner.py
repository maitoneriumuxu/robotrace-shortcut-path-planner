from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robotrace_shortcut_lab.course import load_board_boundary, load_course
from robotrace_shortcut_lab.global_planner import (
    _make_edges,
    analyze_line_interactions,
    analyze_line_interactions_detailed,
    board_path_is_inside,
    edge_connection_is_valid,
    extract_anchor_indices,
    generate_edge_shapes,
    run_global_comparison,
    run_global_mode,
)
from robotrace_shortcut_lab.legal_planner import run_legal_global_mode
from robotrace_shortcut_lab.model import BoardBoundary, COURSE_FILE, Course, PlannerConfig
from robotrace_shortcut_lab.portable import plan_speed, run_comparison
from robotrace_shortcut_lab.report import write_global_result_png


def _course_from_xy(course_id: str, x_mm: np.ndarray, y_mm: np.ndarray) -> Course:
    distance = np.concatenate(
        (
            np.zeros(1),
            np.cumsum(np.hypot(np.diff(x_mm), np.diff(y_mm))),
        )
    ).astype(np.float32)
    return Course(
        course_id,
        distance,
        x_mm.astype(np.float32),
        y_mm.astype(np.float32),
    )


class GlobalPlanner2025Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = PlannerConfig()
        cls.course = load_course(COURSE_FILE)
        cls.global_comparison = run_global_comparison(cls.course, cls.config)

    def test_fixed_speed_model_is_unchanged(self) -> None:
        config = self.config
        self.assertEqual(config.gfcp_reference_speed_mps, 3.6)
        self.assertEqual(config.min_speed_mps, 3.6)
        self.assertEqual(config.max_speed_mps, 13.0)
        self.assertEqual(config.gfcp_exponent, 0.33)
        self.assertEqual(config.min_acceleration_mps2, 20.0)
        self.assertEqual(config.max_acceleration_mps2, 55.0)
        self.assertEqual(config.deceleration_mps2, 55.0)
        self.assertEqual(config.break_kp, 1.0)
        self.assertEqual(config.max_acceleration_omega_deg_s, 300.0)
        self.assertEqual(config.min_acceleration_omega_deg_s, 1500.0)
        self.assertEqual(config.max_aalp_deg_s_per_ms, 100.0)
        self.assertEqual(config.search_run_speed_mps, 3.6)
        self.assertEqual(config.speed_scan_iterations, 4)
        self.assertGreater(config.legal_crossing_risk_penalty_s, 0.0)
        self.assertGreater(config.legal_shallow_crossing_risk_penalty_s, 0.0)

    def test_current_4471_second_baseline_is_reproduced(self) -> None:
        self.assertAlmostEqual(
            self.global_comparison.current_baseline.metrics.predicted_time_s,
            4.4709220381,
            places=6,
        )

    def test_2980_path_is_theoretical_and_never_competition_valid(self) -> None:
        theoretical = self.global_comparison.maximum_vehicle_lower_bound
        self.assertLess(theoretical.adopted.metrics.predicted_time_s, 4.0)
        self.assertFalse(theoretical.legal)
        self.assertIn("競技無効", theoretical.legality_status)
        self.assertNotEqual(
            theoretical.adopted.path.selected_edges,
            self.global_comparison.final.path.selected_edges,
        )

    def test_unconfirmed_ln5_reference_falls_back_to_4471(self) -> None:
        reference = self.global_comparison.reference
        self.assertFalse(reference.legal)
        self.assertTrue(reference.fallback_used)
        self.assertAlmostEqual(
            reference.adopted.metrics.predicted_time_s,
            self.global_comparison.current_baseline.metrics.predicted_time_s,
            places=9,
        )

    def test_embedded_lite_finishes_and_falls_back_when_slower(self) -> None:
        embedded = self.global_comparison.embedded_lite
        self.assertTrue(embedded.adopted.metrics.valid)
        self.assertFalse(embedded.legal)
        self.assertTrue(embedded.fallback_used)
        self.assertLessEqual(
            embedded.adopted.metrics.predicted_time_s,
            self.global_comparison.current_baseline.metrics.predicted_time_s + 1.0e-9,
        )

    def test_anchor_and_edge_limits_are_kept(self) -> None:
        reference = self.global_comparison.maximum_vehicle_lower_bound
        embedded = self.global_comparison.embedded_lite
        self.assertLessEqual(reference.stats.anchor_count, 256)
        self.assertLessEqual(embedded.stats.anchor_count, 96)
        self.assertLessEqual(reference.stats.candidate_edge_count, self.config.reference_edge_limit)
        self.assertLessEqual(embedded.stats.candidate_edge_count, self.config.embedded_edge_limit)
        self.assertLessEqual(reference.stats.valid_edge_count, self.config.reference_edge_limit)
        self.assertLessEqual(embedded.stats.valid_edge_count, self.config.embedded_edge_limit)

    def test_selected_edges_are_strictly_forward(self) -> None:
        for result in (
            self.global_comparison.maximum_vehicle_lower_bound,
            self.global_comparison.reference,
            self.global_comparison.embedded_lite,
        ):
            for _, start, finish, _ in result.adopted.path.selected_edges:
                self.assertLess(start, finish)

    def test_source_progress_is_monotonic_and_does_not_jump_back(self) -> None:
        for item in (
            self.global_comparison.current_baseline,
            self.global_comparison.reference.adopted,
            self.global_comparison.embedded_lite.adopted,
            self.global_comparison.final,
        ):
            self.assertTrue(np.all(np.diff(item.path.source_progress_index) >= -1.0e-6))
            self.assertEqual(float(item.path.source_progress_index[0]), 0.0)
            self.assertAlmostEqual(
                float(item.path.source_progress_index[-1]),
                self.course.point_count - 1,
                delta=1.0e-3,
            )

    def test_global_path_is_resampled_around_ten_mm(self) -> None:
        segment = np.hypot(
            np.diff(self.global_comparison.reference.adopted.path.x_mm),
            np.diff(self.global_comparison.reference.adopted.path.y_mm),
        )
        self.assertAlmostEqual(float(np.median(segment)), 10.0, delta=0.2)
        self.assertLessEqual(float(np.max(segment)), self.config.max_segment_mm)

    def test_top_k_path_is_fully_replanned_with_attack_model(self) -> None:
        reference = self.global_comparison.maximum_vehicle_lower_bound
        self.assertEqual(reference.stats.top_k_count, self.config.reference_top_k)
        replanned = plan_speed(
            reference.adopted.path.x_mm,
            reference.adopted.path.y_mm,
            self.config,
        )
        np.testing.assert_allclose(
            reference.adopted.speed_mps,
            replanned.speed_mps,
            atol=0.0,
        )

    def test_reference_selection_is_deterministic(self) -> None:
        _, second, _ = run_legal_global_mode(
            self.course,
            "reference",
            self.config,
            local_comparison=self.global_comparison.local,
        )
        self.assertEqual(
            self.global_comparison.reference.adopted.path.selected_edges,
            second.adopted.path.selected_edges,
        )
        np.testing.assert_allclose(
            self.global_comparison.reference.adopted.path.x_mm,
            second.adopted.path.x_mm,
            atol=0.0,
        )

    def test_board_boundary_rejects_robot_envelope_outside(self) -> None:
        boundary = load_board_boundary(self.course)
        self.assertIsNotNone(boundary)
        simple = BoardBoundary(((0.0, 1000.0, 0.0, 1000.0),), "test", True)
        self.assertFalse(
            board_path_is_inside(
                np.array([950.0], dtype=np.float32),
                np.array([500.0], dtype=np.float32),
                simple,
                100.0,
            )
        )

    def test_result_png_is_generated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = write_global_result_png(
                Path(temp_dir) / "result.png",
                self.global_comparison,
                self.config,
            )
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 100_000)


class GlobalPlannerUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PlannerConfig()

    def test_arbitrary_competition_tsv_can_be_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "sample.tsv"
            source.write_text(
                "# format\trobotrace-shortcut-course-v1\n"
                "# course_id\tsample\n"
                "distance_mm\tline_x_mm\tline_y_mm\n"
                "0\t0\t0\n10\t10\t0\n20\t20\t0\n",
                encoding="utf-8",
            )
            course = load_course(source)
            self.assertEqual(course.course_id, "sample")
            self.assertEqual(course.point_count, 3)

    def test_straight_course_does_not_create_unnecessary_shortcut(self) -> None:
        x = np.arange(0.0, 1200.0 + 10.0, 10.0)
        course = _course_from_xy("straight", x, np.zeros_like(x))
        compact = replace(
            self.config,
            embedded_anchor_limit=32,
            embedded_edge_limit=128,
            embedded_top_k=4,
        )
        _, result, _ = run_global_mode(course, "embedded-lite", compact)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.adopted.metrics.shortcut_edge_count, 0)

    def test_multiple_s_curve_generates_long_straight_candidate(self) -> None:
        x = np.arange(0.0, 3000.0 + 10.0, 10.0)
        y = 180.0 * np.sin(2.0 * np.pi * x / 600.0)
        course = _course_from_xy("multi_s", x, y)
        compact = replace(
            self.config,
            embedded_anchor_limit=48,
            embedded_edge_limit=256,
        )
        local = run_comparison(course, compact)
        anchors = extract_anchor_indices(
            course,
            local.best.path.x_mm,
            local.best.path.y_mm,
            compact,
            "embedded-lite",
        )
        edges, _, _ = _make_edges(
            course, local, anchors, compact, "embedded-lite", None
        )
        self.assertTrue(
            any(
                edge.kind == "直線"
                and course.distance_mm[edge.end_index]
                - course.distance_mm[edge.start_index]
                > 1000.0
                for edge in edges
            )
        )
        self.assertTrue(all(edge.start_index < edge.end_index for edge in edges))

    def test_all_required_edge_shape_families_are_generated(self) -> None:
        shapes = generate_edge_shapes(
            np.array([0.0, 0.0]),
            np.array([1000.0, 200.0]),
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0]),
            0.0,
            0.0,
        )
        names = {name for name, _ in shapes}
        self.assertTrue(
            {
                "直線",
                "入口遷移＋直線＋出口遷移",
                "3次Hermite",
                "5次Hermite",
                "G2近似",
                "軽量biarc",
            }.issubset(names)
        )

    def test_abnormal_entry_exit_connection_is_rejected(self) -> None:
        points = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
        self.assertFalse(
            edge_connection_is_valid(
                points,
                np.array([-1.0, 0.0]),
                np.array([1.0, 0.0]),
                55.0,
            )
        )

    def test_line_crossing_angle_and_parallel_distance_are_reported(self) -> None:
        source_x = np.array([-100.0, 100.0], dtype=np.float32)
        source_y = np.array([0.0, 0.0], dtype=np.float32)
        crossings, shallow, _, indices = analyze_line_interactions(
            source_x,
            source_y,
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([-100.0, 100.0], dtype=np.float32),
            np.array([100.0, 100.0], dtype=np.float32),
            self.config,
        )
        self.assertEqual(crossings, 1)
        self.assertEqual(shallow, 0)
        self.assertEqual(indices, (0,))
        details = analyze_line_interactions_detailed(
            source_x,
            source_y,
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([-100.0, 100.0], dtype=np.float32),
            np.array([100.0, 100.0], dtype=np.float32),
            self.config,
        )
        self.assertAlmostEqual(details.crossing_angles_deg[0], 90.0)
        self.assertEqual(details.past_crossing_count, 1)
        self.assertEqual(details.future_crossing_count, 0)
        _, _, parallel_mm, _ = analyze_line_interactions(
            source_x,
            source_y,
            np.array([-100.0, 100.0], dtype=np.float32),
            np.array([10.0, 10.0], dtype=np.float32),
            np.array([100.0, 100.0], dtype=np.float32),
            self.config,
        )
        self.assertGreater(parallel_mm, 190.0)

    def test_all_31_course_files_load_and_extract_anchors(self) -> None:
        paths = sorted(Path("data/courses/normalized").glob("*.tsv"))
        self.assertEqual(len(paths), 31)
        for path in paths:
            course = load_course(path)
            anchors = extract_anchor_indices(
                course,
                course.x_mm,
                course.y_mm,
                self.config,
                "embedded-lite",
            )
            self.assertGreaterEqual(anchors.size, 2)
            self.assertLessEqual(anchors.size, self.config.embedded_anchor_limit)
            self.assertEqual(int(anchors[0]), 0)
            self.assertEqual(int(anchors[-1]), course.point_count - 1)


if __name__ == "__main__":
    unittest.main()
