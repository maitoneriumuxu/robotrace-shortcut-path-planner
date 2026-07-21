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

    for item in (comparison.original, comparison.elastic, comparison.best):
        print(
            f"{item.path.label}: {item.metrics.predicted_time_s:.3f}s, "
            f"{item.metrics.length_m:.3f}m, offset={item.metrics.max_offset_mm:.1f}mm, "
            f"radius={item.metrics.min_radius_mm:.1f}mm, "
            f"generation={item.path.generation_s * 1000.0:.1f}ms"
        )
    print(f"選択候補: #{comparison.selected_candidate_id} {comparison.selected_candidate_name}")
    print(output.resolve())
    return 0
