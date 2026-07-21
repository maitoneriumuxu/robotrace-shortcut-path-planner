from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from robotrace_shortcut_lab.course import load_course
from robotrace_shortcut_lab.geometry import frenet_normals, path_length_m
from robotrace_shortcut_lab.model import COURSE_FILE, GeneratedPath, PlannerConfig
from robotrace_shortcut_lab.portable import evaluate_path, run_comparison
from robotrace_shortcut_lab.report import write_result_png


class ShortcutPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = PlannerConfig()
        cls.course = load_course(COURSE_FILE)
        cls.comparison = run_comparison(cls.course, cls.config)

    def test_2025_course_is_loaded(self) -> None:
        self.assertEqual(self.course.course_id, "2025alljapan")
        self.assertEqual(self.course.point_count, 3504)

    def test_paths_keep_count_and_finite_values(self) -> None:
        for evaluated in (
            self.comparison.original,
            self.comparison.elastic,
            self.comparison.best,
        ):
            with self.subTest(path=evaluated.path.label):
                self.assertEqual(evaluated.path.x_mm.size, self.course.point_count)
                self.assertEqual(evaluated.path.y_mm.size, self.course.point_count)
                self.assertTrue(np.isfinite(evaluated.path.x_mm).all())
                self.assertTrue(np.isfinite(evaluated.path.y_mm).all())

    def test_start_and_finish_return_to_source(self) -> None:
        for evaluated in (self.comparison.elastic, self.comparison.best):
            with self.subTest(path=evaluated.path.label):
                np.testing.assert_allclose(
                    evaluated.path.x_mm[[0, -1]], self.course.x_mm[[0, -1]], atol=1.0e-5
                )
                np.testing.assert_allclose(
                    evaluated.path.y_mm[[0, -1]], self.course.y_mm[[0, -1]], atol=1.0e-5
                )

    def test_offset_limit_is_kept(self) -> None:
        for evaluated in (self.comparison.elastic, self.comparison.best):
            self.assertLessEqual(
                evaluated.metrics.max_offset_mm, self.config.offset_limit_mm + 0.01
            )

    def test_time_path_stays_on_source_frenet_normals_without_jump(self) -> None:
        best = self.comparison.best.path
        normal_x, normal_y = frenet_normals(self.course.x_mm, self.course.y_mm)
        np.testing.assert_allclose(
            best.x_mm,
            self.course.x_mm + normal_x * best.offset_mm,
            atol=1.0e-4,
        )
        np.testing.assert_allclose(
            best.y_mm,
            self.course.y_mm + normal_y * best.offset_mm,
            atol=1.0e-4,
        )
        segment_mm = np.hypot(np.diff(best.x_mm), np.diff(best.y_mm))
        self.assertLess(float(np.max(segment_mm)), 20.0)

    def test_minimum_radius_is_judged(self) -> None:
        self.assertTrue(self.comparison.best.metrics.valid)
        self.assertGreaterEqual(
            self.comparison.best.metrics.min_radius_mm,
            self.config.min_radius_mm - 0.01,
        )
        angle = np.linspace(0.0, 2.0 * np.pi, self.course.point_count, dtype=np.float32)
        x = 50.0 * np.cos(angle)
        y = 50.0 * np.sin(angle)
        bad = GeneratedPath(
            "半径違反テスト",
            x,
            y,
            np.zeros(self.course.point_count, dtype=np.float32),
            0.0,
        )
        evaluated = evaluate_path(
            self.course,
            bad,
            self.config,
            path_length_m(self.course.x_mm, self.course.y_mm),
        )
        self.assertFalse(evaluated.metrics.valid)
        self.assertIn("最小半径", evaluated.metrics.violation)

    def test_speed_limits_and_acceleration_scans(self) -> None:
        tolerance = 2.0e-4
        for evaluated in (
            self.comparison.original,
            self.comparison.elastic,
            self.comparison.best,
        ):
            speed = evaluated.speed_mps.astype(np.float64)
            segment = np.diff(evaluated.distance_m)
            self.assertGreaterEqual(float(np.min(speed)), self.config.min_speed_mps - tolerance)
            self.assertLessEqual(float(np.max(speed)), self.config.max_speed_mps + tolerance)
            acceleration_delta = speed[1:] ** 2 - speed[:-1] ** 2
            deceleration_delta = speed[:-1] ** 2 - speed[1:] ** 2
            self.assertTrue(
                np.all(
                    acceleration_delta
                    <= 2.0 * self.config.acceleration_mps2 * segment + tolerance
                )
            )
            self.assertTrue(
                np.all(
                    deceleration_delta
                    <= 2.0 * self.config.deceleration_mps2 * segment + tolerance
                )
            )

    def test_candidate_selection_is_deterministic(self) -> None:
        second = run_comparison(self.course, self.config)
        self.assertEqual(
            self.comparison.selected_candidate_id, second.selected_candidate_id
        )
        np.testing.assert_allclose(
            self.comparison.best.path.offset_mm, second.best.path.offset_mm, atol=0.0
        )

    def test_result_png_is_generated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = write_result_png(
                Path(temp_dir) / "result.png", self.comparison, self.config
            )
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
