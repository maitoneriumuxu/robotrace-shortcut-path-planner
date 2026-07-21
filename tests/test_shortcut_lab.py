from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from robotrace_shortcut_lab.algorithm import generate_paths
from robotrace_shortcut_lab.course import load_course
from robotrace_shortcut_lab.evaluation import evaluate, write_result_png
from robotrace_shortcut_lab.models import Settings


COURSE = Path("data/courses/normalized/2025alljapan.tsv")


class ShortcutLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = Settings()
        cls.course = load_course(COURSE)
        cls.baseline, cls.sharp, cls.smooth = generate_paths(cls.course, cls.settings)
        cls.baseline_result = evaluate(cls.course, cls.baseline, cls.settings)
        cls.sharp_result = evaluate(cls.course, cls.sharp, cls.settings)
        cls.smooth_result = evaluate(cls.course, cls.smooth, cls.settings)

    def test_course_map_is_loaded(self) -> None:
        self.assertEqual(self.course.course_id, "2025alljapan")
        self.assertEqual(self.course.point_count, 3504)

    def test_curvature_continuous_join_suppresses_peak(self) -> None:
        sharp_peak = self.sharp_result.metrics.join_curvature_slew_per_m2
        smooth_peak = self.smooth_result.metrics.join_curvature_slew_per_m2
        self.assertGreater(sharp_peak, 100.0)
        self.assertLess(smooth_peak, sharp_peak * 0.1)

    def test_candidate_keeps_geometry_limits_and_straights(self) -> None:
        metrics = self.smooth_result.metrics
        self.assertTrue(metrics.valid)
        self.assertLessEqual(metrics.max_offset_mm, 100.001)
        self.assertGreaterEqual(metrics.min_radius_mm, 60.0)
        self.assertEqual(len(self.smooth.straight_cores), 5)

    def test_selected_2025_path_matches_reference_result(self) -> None:
        metrics = self.smooth_result.metrics
        self.assertAlmostEqual(metrics.shortening_percent, 6.211, places=3)
        self.assertAlmostEqual(metrics.max_offset_mm, 92.7, places=1)
        self.assertAlmostEqual(metrics.min_radius_mm, 98.8, places=1)
        self.assertAlmostEqual(metrics.max_curvature_slew_per_m2, 132.2, places=1)
        self.assertAlmostEqual(
            self.sharp_result.metrics.join_curvature_slew_per_m2, 166.6, places=1
        )
        self.assertAlmostEqual(metrics.join_curvature_slew_per_m2, 14.4, places=1)

    def test_all_courses_keep_constraints_and_shorten(self) -> None:
        for course_file in Path("data/courses/normalized").glob("*.tsv"):
            with self.subTest(course=course_file.stem):
                course = load_course(course_file)
                _baseline, _sharp, smooth = generate_paths(course, self.settings)
                metrics = evaluate(course, smooth, self.settings).metrics
                self.assertTrue(metrics.valid)
                self.assertGreater(metrics.shortening_percent, 0.0)

    def test_rx651_candidate_uses_fixed_small_work_area(self) -> None:
        header = Path("embedded/shortcut_curvature_limiter.h").read_text(encoding="utf-8")
        source = Path("embedded/shortcut_curvature_limiter.c").read_text(encoding="utf-8")
        self.assertIn("#define SCL_WORK_BYTES            5160", header)
        self.assertNotIn("malloc(", source)
        self.assertNotIn("calloc(", source)
        self.assertNotIn("realloc(", source)

    def test_only_png_result_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.png"
            write_result_png(
                output,
                self.course,
                self.baseline,
                self.sharp,
                self.smooth,
                self.baseline_result,
                self.sharp_result,
                self.smooth_result,
                self.settings,
            )
            self.assertEqual([output], list(Path(temporary).iterdir()))
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
