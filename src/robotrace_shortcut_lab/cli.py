from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from .course import load_course
from .evaluation import evaluate, write_result_png
from .algorithm import generate_paths
from .models import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LN5.xショートカット経路の高速幾何検討")
    parser.add_argument(
        "--course",
        type=Path,
        default=Path("data/courses/normalized/2025alljapan.tsv"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/result.png"))
    args = parser.parse_args(argv)

    started = perf_counter()
    settings = Settings()
    course = load_course(args.course)
    baseline, sharp, smooth = generate_paths(course, settings)
    baseline_result = evaluate(course, baseline, settings)
    sharp_result = evaluate(course, sharp, settings)
    smooth_result = evaluate(course, smooth, settings)
    image = write_result_png(
        args.output,
        course,
        baseline,
        sharp,
        smooth,
        baseline_result,
        sharp_result,
        smooth_result,
        settings,
    )
    metrics = smooth_result.metrics
    print(
        f"{course.course_id}: {'PASS' if metrics.valid else 'REJECTED'} | "
        f"short={metrics.shortening_percent:.3f}% | offset={metrics.max_offset_mm:.1f}mm | "
        f"R={metrics.min_radius_mm:.1f}mm"
    )
    print(
        f"join dκ/ds: sharp={sharp_result.metrics.join_curvature_slew_per_m2:.1f} -> "
        f"smooth={metrics.join_curvature_slew_per_m2:.1f} 1/m^2"
    )
    print(f"overall dκ/ds peak={metrics.max_curvature_slew_per_m2:.1f} 1/m^2")
    print(f"image: {image}")
    print(f"elapsed: {perf_counter() - started:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
