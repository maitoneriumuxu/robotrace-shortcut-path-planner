from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path as FilePath

import numpy as np

from .geometry import curvature_slew_per_m2, path_length_mm, radius_mm
from .models import CoursePath, Metrics, Path, Settings


@dataclass(frozen=True)
class Result:
    metrics: Metrics
    slew_per_m2: np.ndarray
    peak_index: int
    join_peak_index: int


def evaluate(course: CoursePath, path: Path, settings: Settings) -> Result:
    source_length = path_length_mm(course.x_mm, course.y_mm)
    length = path_length_mm(path.x_mm, path.y_mm)
    offset = np.hypot(path.x_mm - course.x_mm, path.y_mm - course.y_mm)
    radius = radius_mm(path.x_mm, path.y_mm, settings.radius_window_segments)
    finite_radius = radius[(radius > 0.0) & (radius < 900_000.0)]
    min_radius = float(np.min(finite_radius)) if finite_radius.size else 999_999.0
    slew = curvature_slew_per_m2(path.x_mm, path.y_mm)
    valid_indices = np.arange(100, max(101, path.x_mm.size - 100))
    peak_index = int(valid_indices[np.argmax(np.abs(slew[valid_indices]))])

    join_indices: list[int] = []
    for start, end in path.straight_cores:
        join_indices.extend(range(max(3, start - 20), min(path.x_mm.size - 3, start + 21)))
        join_indices.extend(range(max(3, end - 20), min(path.x_mm.size - 3, end + 21)))
    if join_indices:
        unique = np.unique(join_indices)
        join_peak_index = int(unique[np.argmax(np.abs(slew[unique]))])
        join_peak = float(abs(slew[join_peak_index]))
    else:
        join_peak_index = peak_index
        join_peak = 0.0
    max_offset = float(np.max(offset))
    metrics = Metrics(
        path_length_mm=length,
        shortening_percent=(source_length - length) / source_length * 100.0,
        max_offset_mm=max_offset,
        min_radius_mm=min_radius,
        max_curvature_slew_per_m2=float(abs(slew[peak_index])),
        join_curvature_slew_per_m2=join_peak,
        valid=max_offset <= settings.offset_limit_mm + 0.001
        and min_radius >= settings.min_radius_mm,
    )
    return Result(metrics, slew, peak_index, join_peak_index)


def write_result_png(
    destination: str | FilePath,
    course: CoursePath,
    baseline: Path,
    sharp: Path,
    smooth: Path,
    baseline_result: Result,
    sharp_result: Result,
    smooth_result: Result,
    settings: Settings,
) -> FilePath:
    """経路と直線接続部の評価を、日本語のPNG画像へ出力する。"""

    import matplotlib.pyplot as plt

    output = FilePath(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "DejaVu Sans"]
    figure = plt.figure(figsize=(14, 9), dpi=150, facecolor="white")
    grid = figure.add_gridspec(2, 2, width_ratios=(1.45, 1.0), wspace=0.25, hspace=0.22)
    whole = figure.add_subplot(grid[:, 0])
    join = figure.add_subplot(grid[0, 1])
    notes = figure.add_subplot(grid[1, 1])

    whole.plot(course.x_mm, course.y_mm, color="#c8ccd1", lw=6, label="原コース")
    whole.plot(
        baseline.x_mm,
        baseline.y_mm,
        color="#d97706",
        lw=1.4,
        ls="--",
        label="現行高速近似",
    )
    whole.plot(
        smooth.x_mm,
        smooth.y_mm,
        color="#087f5b",
        lw=2.0,
        label="直線優先＋dκ/ds正則化経路",
    )
    hotspot = smooth_result.peak_index
    whole.scatter(
        [smooth.x_mm[hotspot]],
        [smooth.y_mm[hotspot]],
        marker="x",
        s=75,
        linewidths=2.2,
        color="#c2410c",
        label="次の改善対象",
        zorder=5,
    )
    year = course.course_id[:4]
    whole.set_title(f"{year}年 全日本 ― 直線優先＋dκ/ds正則化", fontsize=14)
    whole.set_aspect("equal")
    whole.set_xlabel("X [mm]")
    whole.set_ylabel("Y [mm]")
    whole.grid(alpha=0.2)
    whole.legend(loc="best")

    half_window = 30

    def join_window(result: Result) -> tuple[np.ndarray, np.ndarray]:
        center = result.join_peak_index
        start = max(0, center - half_window)
        end = min(result.slew_per_m2.size, center + half_window + 1)
        relative_mm = (np.arange(start, end) - center) * 10.0
        return relative_mm, np.abs(result.slew_per_m2[start:end])

    sharp_distance, sharp_slew = join_window(sharp_result)
    smooth_distance, smooth_slew = join_window(smooth_result)
    sharp_peak = sharp_result.metrics.join_curvature_slew_per_m2
    smooth_peak = smooth_result.metrics.join_curvature_slew_per_m2
    join.plot(
        sharp_distance,
        sharp_slew,
        color="#c2410c",
        ls="--",
        lw=1.8,
        label=f"鋭角接続 {sharp_peak:.1f}",
    )
    join.plot(
        smooth_distance,
        smooth_slew,
        color="#087f5b",
        lw=2.2,
        label=f"曲率連続接続 {smooth_peak:.1f}",
    )
    join.set_title("最も厳しい直線―曲線接続（正則化案）")
    join.set_xlabel("接続点からの距離 [mm]")
    join.set_ylabel("|dκ/ds| [1/m²]")
    join.grid(alpha=0.2)
    join.legend(loc="best")

    metrics = smooth_result.metrics
    reduction = (1.0 - smooth_peak / sharp_peak) * 100.0 if sharp_peak else 0.0
    status = "合格" if metrics.valid else "制約違反"
    summary = (
        f"{status}\n"
        f"接続ピーク: {sharp_peak:.1f} → {smooth_peak:.1f} [1/m²] "
        f"（{reduction:.1f}%低減）\n"
        f"経路短縮率: {metrics.shortening_percent:.3f}%\n"
        f"最大オフセット: {metrics.max_offset_mm:.1f} / "
        f"{settings.offset_limit_mm:.0f} mm\n"
        f"最小半径: {metrics.min_radius_mm:.1f} / "
        f"{settings.min_radius_mm:.0f} mm\n"
        f"経路全体ピーク: {metrics.max_curvature_slew_per_m2:.1f} [1/m²]\n"
        f"直線コア数: {len(smooth.straight_cores)}\n"
        f"次の改善対象: 中央の曲線区間（点番号 {hotspot}）\n\n"
        "一定速度時: 角加速度 = v² × dκ/ds"
    )
    notes.axis("off")
    notes.text(0.02, 0.95, summary, va="top", ha="left", fontsize=12, linespacing=1.55)
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output.resolve()
