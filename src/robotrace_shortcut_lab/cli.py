from __future__ import annotations

from pathlib import Path

from .course import load_course
from .model import COURSE_FILE, PlannerConfig
from .portable import run_comparison
from .report import write_result_png


def main() -> int:
    config = PlannerConfig()
    course = load_course(COURSE_FILE)
    comparison = run_comparison(course, config)
    output = write_result_png(Path("outputs/result.png"), comparison, config)

    for item in (
        comparison.original,
        comparison.elastic,
        comparison.legacy_time,
        comparison.best,
    ):
        print(
            f"{item.path.label}: {item.metrics.predicted_time_s:.3f}s, "
            f"GFCP単独={item.metrics.gfcp_only_time_s:.3f}s, "
            f"{item.metrics.length_m:.3f}m, offset={item.metrics.max_offset_mm:.1f}mm, "
            f"radius={item.metrics.min_radius_mm:.1f}mm, "
            f"generation={item.path.generation_s * 1000.0:.1f}ms, "
            f"speed_plan={item.speed_plan_s * 1000.0:.1f}ms"
        )
    print(f"選択候補: #{comparison.selected_candidate_id} {comparison.selected_candidate_name}")
    print(
        f"候補評価: {comparison.candidate_evaluation_count}本, "
        f"探索={comparison.candidate_search_s * 1000.0:.1f}ms, "
        f"固定O(N)走査概算={comparison.approximate_o_n_scans}回"
    )
    print(output.resolve())
    return 0
