from __future__ import annotations

import unittest

import numpy as np

from robotrace_shortcut_lab.cli import _parser
from robotrace_shortcut_lab.deep_planner import (
    DEFAULT_DEEP_STAGES,
    assemble_deep_path,
    generate_deep_edge_shapes,
)
from robotrace_shortcut_lab.global_planner import _ShortcutEdge
from robotrace_shortcut_lab.model import Course, PlannerConfig


def _straight_course() -> Course:
    x = np.arange(0.0, 101.0, 10.0, dtype=np.float32)
    y = np.zeros_like(x)
    return Course("deep_test", x.copy(), x, y)


def _edge(edge_id: int, start: int, end: int) -> _ShortcutEdge:
    x = np.linspace(start * 10.0, end * 10.0, end - start + 1).astype(np.float32)
    y = np.zeros_like(x)
    return _ShortcutEdge(
        edge_id,
        start,
        end,
        start,
        end,
        "試験直線",
        x,
        y,
        np.linspace(start, end, x.size).astype(np.float32),
        0.1,
        float((end - start) * 10.0),
        0,
        0,
        (),
        (),
        0,
        0,
        0.0,
        True,
        "",
    )


class DeepReferenceTests(unittest.TestCase):
    def test_cli_accepts_deep_reference(self) -> None:
        args = _parser().parse_args(["--mode", "deep-reference"])
        self.assertEqual(args.mode, "deep-reference")
        self.assertEqual(args.deep_budget_seconds, 1500.0)

    def test_staged_search_scales_are_fixed(self) -> None:
        self.assertEqual(
            [(item.anchor_limit, item.edge_limit, item.legal_edge_limit, item.top_k)
             for item in DEFAULT_DEEP_STAGES],
            [(256, 20_000, 1_200, 16), (384, 50_000, 5_000, 64),
             (512, 100_000, 20_000, 256)],
        )

    def test_deep_shapes_cover_requested_families_and_scales(self) -> None:
        p0 = np.asarray((0.0, 0.0))
        p1 = np.asarray((800.0, 100.0))
        t0 = np.asarray((1.0, 0.0))
        t1 = np.asarray((1.0, 0.1))
        t1 /= np.linalg.norm(t1)
        shapes = generate_deep_edge_shapes(p0, p1, t0, t1, 0.2, -0.2)
        names = {name for name, _ in shapes}
        for family in ("直線", "3次Hermite", "5次Hermite", "G2近似", "軽量biarc"):
            self.assertTrue(any(name.startswith(family) for name in names), family)
        self.assertTrue(any("入口遷移＋直線＋出口遷移" in name for name in names))
        for scale in ("s0.20", "s0.30", "s0.40", "s0.50", "s0.60", "s0.70", "s0.80"):
            self.assertTrue(any(scale in name for name in names), scale)
        self.assertTrue(any("非対称" in name for name in names))
        self.assertTrue(any("法線" in name for name in names))
        self.assertTrue(all(points.shape[1] == 2 for _, points in shapes))

    def test_assembled_path_is_resampled_and_forward_ordered(self) -> None:
        course = _straight_course()
        path = assemble_deep_path(
            course,
            course.x_mm,
            course.y_mm,
            (_edge(1, 2, 5), _edge(2, 7, 9)),
            PlannerConfig(),
            "試験",
        )
        self.assertEqual([(item[1], item[2]) for item in path.selected_edges], [(2, 5), (7, 9)])
        self.assertLessEqual(float(np.max(np.hypot(np.diff(path.x_mm), np.diff(path.y_mm)))), 10.01)
        self.assertAlmostEqual(float(path.x_mm[0]), 0.0)
        self.assertAlmostEqual(float(path.x_mm[-1]), 100.0)

    def test_overlapping_edges_are_rejected(self) -> None:
        course = _straight_course()
        with self.assertRaises(ValueError):
            assemble_deep_path(
                course,
                course.x_mm,
                course.y_mm,
                (_edge(1, 2, 6), _edge(2, 5, 9)),
                PlannerConfig(),
                "重複",
            )

    def test_hard_constraints_remain_unchanged(self) -> None:
        config = PlannerConfig()
        self.assertEqual(config.gfcp_reference_speed_mps, 3.6)
        self.assertEqual(config.min_speed_mps, 3.6)
        self.assertEqual(config.max_speed_mps, 13.0)
        self.assertEqual(config.speed_scan_iterations, 4)
        self.assertEqual(config.legal_pose_step_mm, 2.0)
        self.assertEqual(config.legal_yaw_step_deg, 1.0)
        self.assertEqual(config.contact_dp_max_progress_step_segments, 1)


if __name__ == "__main__":
    unittest.main()
