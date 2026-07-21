from __future__ import annotations

import argparse
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robotrace_shortcut_lab.algorithm import _fast_firmware_style
from robotrace_shortcut_lab.course import load_course
from robotrace_shortcut_lab.evaluation import evaluate, write_result_png
from robotrace_shortcut_lab.models import Path as GeneratedPath
from robotrace_shortcut_lab.models import Settings


def write_input(destination: Path, course: object, baseline: GeneratedPath) -> None:
    with destination.open("wb") as output:
        output.write(struct.pack("<i", course.point_count))
        for values in (course.x_mm, course.y_mm, baseline.x_mm, baseline.y_mm):
            output.write(np.asarray(values, dtype="<f4").tobytes())


def read_output(
    source: Path,
) -> tuple[tuple[float, float, int, int, int, int, int, int], np.ndarray, np.ndarray]:
    data = source.read_bytes()
    count = struct.unpack_from("<i", data, 0)[0]
    result = struct.unpack_from("<ffiiiQQQ", data, 4)
    arrays = np.frombuffer(data, dtype="<f4", count=count * 2, offset=48).copy()
    return result, arrays[:count], arrays[count:]


def main() -> int:
    parser = argparse.ArgumentParser(description="RX651向けC曲率リミッタを全コースで実行")
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--teardrop-center", type=int, default=69)
    parser.add_argument("--teardrop-shift-mm", type=float, default=94.0)
    parser.add_argument("--teardrop-max-slew-ratio", type=float, default=1.0)
    parser.add_argument("--regularization", type=float, default=10000.0)
    parser.add_argument("--teardrop-pass-limit", type=int, default=2)
    parser.add_argument("--max-passes", type=int, default=12)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    settings = Settings()
    failures: list[str] = []
    started = perf_counter()

    with tempfile.TemporaryDirectory() as temporary_text:
        temporary = Path(temporary_text)
        executable = temporary / "scl_host.exe"
        subprocess.run(
            [
                str(args.compiler),
                f"-DSCL_TEARDROP_CENTER={args.teardrop_center}",
                f"-DSCL_TEARDROP_SHIFT_MM={args.teardrop_shift_mm}f",
                f"-DSCL_TEARDROP_MAX_SLEW_RATIO={args.teardrop_max_slew_ratio}f",
                f"-DSCL_REGULARIZATION={args.regularization}f",
                f"-DSCL_TEARDROP_PASS_LIMIT={args.teardrop_pass_limit}",
                f"-DSCL_HOST_MAX_PASS_COUNT={args.max_passes}",
                "-I" + str(root / "embedded"),
                "-o",
                str(executable),
                str(root / "embedded/host_harness.c"),
                str(root / "embedded/shortcut_curvature_limiter.c"),
            ],
            check=True,
        )
        rows: list[tuple[str, float, float, float, float, float, int, int, int, float]] = []
        maximum_c_ms = 0.0
        maximum_reads = 0
        maximum_writes = 0
        maximum_sqrts = 0
        for course_file in sorted((root / "data/courses/normalized").glob("*.tsv")):
            course = load_course(course_file)
            baseline = _fast_firmware_style(course, settings)
            baseline_metrics = evaluate(course, baseline, settings).metrics
            input_file = temporary / "input.bin"
            output_file = temporary / "output.bin"
            write_input(input_file, course, baseline)
            c_started = perf_counter()
            subprocess.run([str(executable), str(input_file), str(output_file)], check=True)
            c_ms = (perf_counter() - c_started) * 1000.0
            maximum_c_ms = max(maximum_c_ms, c_ms)
            c_result, x_mm, y_mm = read_output(output_file)
            path = GeneratedPath("RX651 C", x_mm, y_mm)
            metrics = evaluate(course, path, settings).metrics
            before, after, before_index, after_index, accepted, reads, writes, sqrts = c_result
            maximum_reads = max(maximum_reads, reads)
            maximum_writes = max(maximum_writes, writes)
            maximum_sqrts = max(maximum_sqrts, sqrts)
            rows.append(
                (
                    course_file.stem,
                    metrics.shortening_percent,
                    metrics.max_offset_mm,
                    metrics.min_radius_mm,
                    baseline_metrics.max_curvature_slew_per_m2,
                    metrics.max_curvature_slew_per_m2,
                    accepted,
                    before_index,
                    after_index,
                    c_ms,
                )
            )
            if args.image is not None and course_file.stem == "2025alljapan":
                image = args.image if args.image.is_absolute() else root / args.image
                write_result_png(
                    image,
                    course,
                    baseline,
                    baseline,
                    path,
                    evaluate(course, baseline, settings),
                    evaluate(course, baseline, settings),
                    evaluate(course, path, settings),
                    settings,
                )
            if (
                not metrics.valid
                or metrics.shortening_percent <= 0.0
                or after > before + 0.01
                or metrics.max_curvature_slew_per_m2
                > baseline_metrics.max_curvature_slew_per_m2 + 0.1
            ):
                failures.append(course_file.stem)

    for (
        name,
        shortening,
        offset,
        radius,
        before_slew,
        slew,
        accepted,
        before_index,
        after_index,
        c_ms,
    ) in rows:
        print(
            f"{name:24s} short={shortening:7.3f}% off={offset:6.1f} "
            f"R={radius:6.1f} dK={before_slew:7.1f}->{slew:7.1f} "
            f"accepted={accepted} peak={before_index}->{after_index} C={c_ms:5.1f}ms"
        )
    print(
        f"courses={len(rows)} failures={len(failures)} max_C={maximum_c_ms:.1f}ms "
        f"max_reads={maximum_reads} max_writes={maximum_writes} "
        f"max_sqrts={maximum_sqrts} "
        f"elapsed={perf_counter() - started:.3f}s"
    )
    if failures:
        print("failed: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
