from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from robotrace_shortcut_lab.course import load_course
from robotrace_shortcut_lab.geometry import (
    curvature_slew_per_m2,
    frenet_normals,
    path_order_is_forward,
    self_intersection_count,
    signed_curvature_per_m,
)
from robotrace_shortcut_lab.model import COURSE_FILE, PlannerConfig
from robotrace_shortcut_lab.portable import (
    LIMIT_AALP,
    acceleration_limit_from_omega,
    gate_runup_distance_m,
    gate_runup_start_speed_mps,
    plan_speed,
    run_comparison,
)
from robotrace_shortcut_lab.report import write_result_png


class ShortcutPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = PlannerConfig()
        cls.course = load_course(COURSE_FILE)
        cls.comparison = run_comparison(cls.course, cls.config)
        cls.paths = (
            cls.comparison.original,
            cls.comparison.elastic,
            cls.comparison.legacy_time,
            cls.comparison.best,
        )

    def test_2025_course_is_loaded(self) -> None:
        self.assertEqual(self.course.course_id, "2025alljapan")
        self.assertEqual(self.course.point_count, 3504)

    def test_paths_keep_count_float32_and_finite_values(self) -> None:
        for evaluated in self.paths:
            with self.subTest(path=evaluated.path.label):
                self.assertEqual(evaluated.path.x_mm.size, self.course.point_count)
                self.assertEqual(evaluated.path.y_mm.size, self.course.point_count)
                self.assertEqual(evaluated.path.x_mm.dtype, np.float32)
                self.assertEqual(evaluated.path.y_mm.dtype, np.float32)
                self.assertTrue(np.isfinite(evaluated.path.x_mm).all())
                self.assertTrue(np.isfinite(evaluated.path.y_mm).all())

    def test_start_and_finish_return_to_source(self) -> None:
        for evaluated in self.paths:
            with self.subTest(path=evaluated.path.label):
                np.testing.assert_allclose(
                    evaluated.path.x_mm[[0, -1]],
                    self.course.x_mm[[0, -1]],
                    atol=1.0e-5,
                )
                np.testing.assert_allclose(
                    evaluated.path.y_mm[[0, -1]],
                    self.course.y_mm[[0, -1]],
                    atol=1.0e-5,
                )

    def test_offset_and_point_spacing_constraints_are_kept(self) -> None:
        for evaluated in (
            self.comparison.legacy_time,
            self.comparison.best,
        ):
            with self.subTest(path=evaluated.path.label):
                offset = evaluated.path.offset_mm
                segment = np.hypot(
                    np.diff(evaluated.path.x_mm), np.diff(evaluated.path.y_mm)
                )
                self.assertLessEqual(
                    float(np.max(np.abs(offset))),
                    self.config.offset_limit_mm + 0.01,
                )
                self.assertLessEqual(
                    float(np.max(np.abs(np.diff(offset)))),
                    self.config.max_offset_step_mm,
                )
                self.assertLessEqual(
                    float(np.max(segment)), self.config.max_segment_mm
                )

    def test_frenet_paths_stay_on_same_source_index_normals(self) -> None:
        normal_x, normal_y = frenet_normals(self.course.x_mm, self.course.y_mm)
        for evaluated in (
            self.comparison.legacy_time,
            self.comparison.best,
        ):
            with self.subTest(path=evaluated.path.label):
                self.assertTrue(evaluated.path.frenet_locked)
                np.testing.assert_allclose(
                    evaluated.path.x_mm,
                    self.course.x_mm + normal_x * evaluated.path.offset_mm,
                    atol=1.0e-4,
                )
                np.testing.assert_allclose(
                    evaluated.path.y_mm,
                    self.course.y_mm + normal_y * evaluated.path.offset_mm,
                    atol=1.0e-4,
                )

    def test_path_order_and_self_intersection_do_not_worsen(self) -> None:
        source_crossings = self_intersection_count(
            self.course.x_mm, self.course.y_mm
        )
        for evaluated in self.paths:
            with self.subTest(path=evaluated.path.label):
                self.assertTrue(
                    path_order_is_forward(
                        self.course.x_mm,
                        self.course.y_mm,
                        evaluated.path.x_mm,
                        evaluated.path.y_mm,
                    )
                )
                self.assertLessEqual(
                    self_intersection_count(
                        evaluated.path.x_mm, evaluated.path.y_mm
                    ),
                    source_crossings,
                )

    def test_attack_speed_minimum_maximum_and_gate_boundaries(self) -> None:
        tolerance = 2.0e-4
        self.assertEqual(
            self.config.gfcp_reference_speed_mps,
            self.config.min_speed_mps,
        )
        for evaluated in self.paths:
            with self.subTest(path=evaluated.path.label):
                speed = evaluated.speed_mps.astype(np.float64)
                self.assertGreaterEqual(
                    float(np.min(speed)), self.config.min_speed_mps - tolerance
                )
                self.assertLessEqual(
                    float(np.max(speed)), self.config.max_speed_mps + tolerance
                )
                start_limit = gate_runup_start_speed_mps(
                    evaluated.path.x_mm, evaluated.path.y_mm, self.config
                )
                self.assertLessEqual(float(speed[0]), start_limit + tolerance)
                self.assertGreater(float(speed[0]), self.config.min_speed_mps)
                self.assertGreater(float(speed[-1]), self.config.min_speed_mps)

    def test_one_meter_gate_runup_starts_from_rest_and_finish_is_free(self) -> None:
        x = np.linspace(0.0, 1000.0, 101, dtype=np.float32)
        y = np.zeros_like(x)
        self.assertAlmostEqual(gate_runup_distance_m(x, y), 1.0, places=7)
        expected = np.sqrt(2.0 * self.config.max_acceleration_mps2)
        self.assertAlmostEqual(
            gate_runup_start_speed_mps(x, y, self.config), expected, places=6
        )
        plan = plan_speed(x, y, self.config)
        self.assertAlmostEqual(float(plan.speed_mps[0]), expected, delta=2.0e-4)
        self.assertAlmostEqual(
            float(plan.speed_mps[-1]), self.config.max_speed_mps, delta=2.0e-4
        )

    def test_forward_scan_obeys_omega_dependent_acceleration(self) -> None:
        tolerance = 3.0e-5
        for evaluated in self.paths:
            speed = evaluated.speed_mps.astype(np.float64)
            curvature = signed_curvature_per_m(
                evaluated.path.x_mm, evaluated.path.y_mm
            )
            omega_deg_s = np.rad2deg(speed * curvature)
            acceleration = acceleration_limit_from_omega(
                omega_deg_s, self.config
            )
            allowed = (
                2.0
                * np.minimum(acceleration[:-1], acceleration[1:])
                * np.diff(evaluated.distance_m)
            )
            actual = speed[1:] ** 2 - speed[:-1] ** 2
            self.assertTrue(np.all(actual <= allowed + tolerance))

    def test_backward_scan_obeys_55_mps2_deceleration(self) -> None:
        tolerance = 3.0e-5
        for evaluated in self.paths:
            speed = evaluated.speed_mps.astype(np.float64)
            actual = speed[:-1] ** 2 - speed[1:] ** 2
            allowed = (
                2.0
                * self.config.deceleration_mps2
                / self.config.break_kp
                * np.diff(evaluated.distance_m)
            )
            self.assertTrue(np.all(actual <= allowed + tolerance))

    def test_1500_deg_s_is_not_a_speed_hard_limit(self) -> None:
        curvature = signed_curvature_per_m(
            self.comparison.best.path.x_mm,
            self.comparison.best.path.y_mm,
        )
        omega = np.abs(
            np.rad2deg(self.comparison.best.speed_mps * curvature)
        )
        self.assertGreater(float(np.max(omega)), 1500.0)
        self.assertGreaterEqual(
            float(np.min(self.comparison.best.speed_mps)),
            self.config.min_speed_mps - 2.0e-4,
        )

    def test_aalp_uses_firmware_one_ms_numeric_unit(self) -> None:
        plan = plan_speed(
            self.comparison.best.path.x_mm,
            self.comparison.best.path.y_mm,
            self.config,
        )
        slew = np.abs(
            curvature_slew_per_m2(
                self.comparison.best.path.x_mm,
                self.comparison.best.path.y_mm,
            )
        )
        alpha_numeric = np.rad2deg(
            self.config.search_run_speed_mps**2 * slew
        ) * self.config.firmware_sample_s
        expected = np.clip(
            self.config.search_run_speed_mps
            * np.sqrt(
                np.divide(
                    self.config.max_aalp_deg_s_per_ms,
                    alpha_numeric,
                    out=np.full_like(alpha_numeric, np.inf),
                    where=alpha_numeric > 1.0e-9,
                )
            ),
            self.config.min_speed_mps,
            self.config.max_speed_mps,
        )
        np.testing.assert_allclose(plan.aalp_limit_mps, expected, rtol=1.0e-6)

    def test_legacy_300_rad_s2_setting_is_removed(self) -> None:
        self.assertFalse(hasattr(self.config, "max_angular_accel_rad_s2"))
        self.assertFalse(hasattr(self.config, "max_omega_deg_s"))

    def test_minimum_radius_is_diagnostic_not_hard_constraint(self) -> None:
        self.assertTrue(self.comparison.best.metrics.valid)
        self.assertLess(self.comparison.best.metrics.min_radius_mm, 100.0)

    def test_elastic_is_a_fallback_and_final_time_never_worsens(self) -> None:
        self.assertLessEqual(
            self.comparison.best.metrics.predicted_time_s,
            self.comparison.elastic.metrics.predicted_time_s + 1.0e-9,
        )
        self.assertLess(
            self.comparison.best.metrics.predicted_time_s,
            self.comparison.legacy_time.metrics.predicted_time_s,
        )

    def test_long_window_centers_are_detected_without_fixed_course_indices(self) -> None:
        centers = self.comparison.window_center_indices
        self.assertGreaterEqual(len(centers), 3)
        curvature = signed_curvature_per_m(self.course.x_mm, self.course.y_mm)
        for left, right in zip(centers[:-1], centers[1:], strict=True):
            self.assertLess(curvature[left] * curvature[right], 0.0)

    def test_candidate_selection_is_deterministic(self) -> None:
        second = run_comparison(self.course, self.config)
        self.assertEqual(
            self.comparison.selected_candidate_id, second.selected_candidate_id
        )
        self.assertEqual(
            self.comparison.window_center_indices, second.window_center_indices
        )
        np.testing.assert_allclose(
            self.comparison.best.path.offset_mm,
            second.best.path.offset_mm,
            atol=0.0,
        )

    def test_aalp_compatibility_result_is_reported(self) -> None:
        for evaluated in self.paths:
            self.assertGreater(evaluated.metrics.gfcp_only_time_s, 0.0)
            self.assertEqual(
                int(np.count_nonzero(evaluated.speed_limit_reason == LIMIT_AALP)),
                0,
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
