from __future__ import annotations

from pathlib import Path as FilePath

import numpy as np

from .evaluation import evaluate
from .geometry import cumulative_distance_mm, path_length_mm
from .models import CoursePath, Path, Settings


ITERATIONS = 420
SOURCE_WEIGHT = np.float32(0.0005)
SMOOTH_WEIGHT = np.float32(0.36)
MIN_RADIUS_MM = np.float32(100.0)
RADIUS_OFFSET = 20
RADIUS_RELAX_BLEND = np.float32(0.012)
EDGE_KEEP_SEG = 30
EDGE_BLEND_SEG = 30
EDGE_SOURCE_WEIGHT = np.float32(0.05)
OFFSET_WARNING_MM = 75.0
LOCAL_SLEW_LIMIT_PER_M2 = 300.0
STRAIGHT_RADIUS_MM = np.float32(999_999.0)
EPSILON = np.float32(0.001)


def _radius_mm(x: np.ndarray, y: np.ndarray, index: int) -> np.float32:
    x0, y0 = x[index - RADIUS_OFFSET], y[index - RADIUS_OFFSET]
    x1, y1 = x[index], y[index]
    x2, y2 = x[index + RADIUS_OFFSET], y[index + RADIUS_OFFSET]
    dx01, dy01 = np.float32(x1 - x0), np.float32(y1 - y0)
    dx12, dy12 = np.float32(x2 - x1), np.float32(y2 - y1)
    dx20, dy20 = np.float32(x0 - x2), np.float32(y0 - y2)
    cross = np.float32(dx01 * np.float32(y2 - y0) - dy01 * np.float32(x2 - x0))
    if abs(float(cross)) <= float(EPSILON):
        return STRAIGHT_RADIUS_MM
    length01 = np.float32(np.sqrt(np.float32(dx01 * dx01 + dy01 * dy01)))
    length12 = np.float32(np.sqrt(np.float32(dx12 * dx12 + dy12 * dy12)))
    length20 = np.float32(np.sqrt(np.float32(dx20 * dx20 + dy20 * dy20)))
    if min(float(length01), float(length12), float(length20)) <= float(EPSILON):
        return np.float32(0.0)
    return np.float32(
        np.float32(length01 * length12 * length20)
        / np.float32(np.float32(2.0) * abs(cross))
    )


def _edge_weight(index: int, max_index: int) -> np.float32:
    edge_distance = min(index, max_index - index)
    if edge_distance <= EDGE_KEEP_SEG:
        return EDGE_SOURCE_WEIGHT
    if EDGE_KEEP_SEG + EDGE_BLEND_SEG <= edge_distance:
        return SOURCE_WEIGHT
    ratio = np.float32(edge_distance - EDGE_KEEP_SEG) / np.float32(EDGE_BLEND_SEG)
    return np.float32(
        EDGE_SOURCE_WEIGHT + np.float32((SOURCE_WEIGHT - EDGE_SOURCE_WEIGHT) * ratio)
    )


def generate_current_firmware_path(course: CoursePath) -> Path:
    """現行C実装と同じ定数・走査順でElastic Band経路を生成する。"""

    source_x = course.x_mm.astype(np.float32, copy=True)
    source_y = course.y_mm.astype(np.float32, copy=True)
    x, y = source_x.copy(), source_y.copy()
    max_index = course.point_count - 1

    for _iteration in range(ITERATIONS):
        if RADIUS_OFFSET * 2 <= max_index:
            for center in range(RADIUS_OFFSET, max_index - RADIUS_OFFSET + 1):
                radius = _radius_mm(x, y, center)
                if np.float32(0.0) < radius < MIN_RADIUS_MM:
                    start = max(1, center - RADIUS_OFFSET)
                    end = min(max_index - 1, center + RADIUS_OFFSET)
                    for index in range(start, end + 1):
                        x[index] = np.float32(
                            x[index] + np.float32((source_x[index] - x[index]) * RADIUS_RELAX_BLEND)
                        )
                        y[index] = np.float32(
                            y[index] + np.float32((source_y[index] - y[index]) * RADIUS_RELAX_BLEND)
                        )

        for index in range(1, max_index):
            weight = _edge_weight(index, max_index)
            old_x, old_y = x[index], y[index]
            data_x = np.float32((source_x[index] - old_x) * weight)
            data_y = np.float32((source_y[index] - old_y) * weight)
            smooth_x = np.float32(
                np.float32(x[index - 1] + x[index + 1] - np.float32(2.0) * old_x)
                * SMOOTH_WEIGHT
            )
            smooth_y = np.float32(
                np.float32(y[index - 1] + y[index + 1] - np.float32(2.0) * old_y)
                * SMOOTH_WEIGHT
            )
            x[index] = np.float32(old_x + data_x + smooth_x)
            y[index] = np.float32(old_y + data_y + smooth_y)

    return Path("現行ファーム Elastic Band", x, y)


def generate_curvature_continuous_path(course: CoursePath, current_path: Path) -> Path:
    """コース全体から候補を自動抽出し、曲率連続ショートカットを生成する。"""

    from .algorithm import _connect_straights, _straight_corridors

    settings = Settings(transition_segments=50)
    corridors = _straight_corridors(course, settings)
    path = _connect_straights(course, current_path, corridors, settings, smooth=True)
    local_candidates = _find_local_shortcut_candidates(course, path, settings, corridors)
    if not local_candidates:
        return path

    selected_cores: list[tuple[int, int]] = []
    working = path
    working_length = path_length_mm(working.x_mm, working.y_mm)
    base_slew = evaluate(course, path, settings).metrics.max_curvature_slew_per_m2
    slew_limit = max(LOCAL_SLEW_LIMIT_PER_M2, 1.25 * base_slew)
    for _score, interval, core in local_candidates:
        core_start, core_end = core
        if any(
            core_start <= other_end
            and other_start <= core_end
            for other_start, other_end in selected_cores
        ):
            continue
        candidate = _connect_straights(course, working, (interval,), settings, smooth=True)
        if not candidate.straight_cores:
            continue
        metrics = evaluate(course, candidate, settings).metrics
        candidate_length = path_length_mm(candidate.x_mm, candidate.y_mm)
        if metrics.max_curvature_slew_per_m2 > slew_limit:
            continue
        if candidate_length >= working_length - 5.0:
            continue
        selected_cores.append(core)
        working = candidate
        working_length = candidate_length
    return working


def _find_local_shortcut_candidates(
    course: CoursePath,
    base_path: Path,
    settings: Settings,
    excluded: tuple[tuple[int, int], ...],
) -> list[tuple[float, tuple[int, int], tuple[int, int]]]:
    """座標番号に依存せず、弧長と弦長の差から局所ショートカット候補を探す。"""

    from .algorithm import _connect_straights

    count = course.point_count
    distance = cumulative_distance_mm(base_path.x_mm, base_path.y_mm)
    base_length = path_length_mm(base_path.x_mm, base_path.y_mm)
    base_slew = evaluate(course, base_path, settings).metrics.max_curvature_slew_per_m2
    slew_limit = max(LOCAL_SLEW_LIMIT_PER_M2, 1.25 * base_slew)
    candidates: list[tuple[float, tuple[int, int], tuple[int, int]]] = []
    min_span = 100
    max_span = 320
    step = 20
    for start in range(80, count - min_span - 80, step):
        for span in range(min_span, max_span + 1, step):
            end = start + span
            if end >= count - 80:
                continue
            if any(start < other_end and other_start < end for other_start, other_end in excluded):
                continue
            arc = float(distance[end] - distance[start])
            chord = float(
                np.hypot(
                    base_path.x_mm[end] - base_path.x_mm[start],
                    base_path.y_mm[end] - base_path.y_mm[start],
                )
            )
            if arc <= 0.0 or chord / arc > 0.94:
                continue
            trial = _connect_straights(course, base_path, ((start, end),), settings, smooth=True)
            if not trial.straight_cores:
                continue
            trial_metrics = evaluate(course, trial, settings).metrics
            if trial_metrics.max_curvature_slew_per_m2 > slew_limit:
                continue
            trial_length = path_length_mm(trial.x_mm, trial.y_mm)
            shortening = base_length - trial_length
            if shortening <= 5.0:
                continue
            core = trial.straight_cores[-1]
            # 短縮量を主目的にしつつ、同等ならオフセットを使う候補を優先する。
            offset = float(
                np.max(
                    np.hypot(
                        trial.x_mm[start : end + 1] - course.x_mm[start : end + 1],
                        trial.y_mm[start : end + 1] - course.y_mm[start : end + 1],
                    )
                )
            )
            score = shortening + 0.02 * offset
            candidates.append((score, (start, end), core))
    return sorted(candidates, reverse=True)


def write_current_firmware_png(
    destination: str | FilePath,
    course: CoursePath,
    path: Path,
) -> FilePath:
    """現行ファーム経路と評価値を日本語PNGへまとめる。"""

    import matplotlib.pyplot as plt

    settings = Settings(offset_limit_mm=OFFSET_WARNING_MM, min_radius_mm=float(MIN_RADIUS_MM))
    result = evaluate(course, path, settings)
    metrics = result.metrics
    output = FilePath(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "DejaVu Sans"]
    figure = plt.figure(figsize=(14, 9), dpi=150, facecolor="white")
    grid = figure.add_gridspec(2, 2, width_ratios=(1.45, 1.0), wspace=0.25, hspace=0.25)
    whole = figure.add_subplot(grid[:, 0])
    slew = figure.add_subplot(grid[0, 1])
    notes = figure.add_subplot(grid[1, 1])

    whole.plot(course.x_mm, course.y_mm, color="#7d858d", lw=1.8, label="原コース")
    whole.plot(path.x_mm, path.y_mm, color="#2563a6", lw=2.0, label="現行ファーム経路")
    peak = result.peak_index
    whole.scatter(
        [path.x_mm[peak]], [path.y_mm[peak]], marker="x", s=75,
        linewidths=2.2, color="#c2410c", label="最大曲率変化位置", zorder=5,
    )
    year = course.course_id[:4]
    whole.set_title(f"{year}年 全日本 ― 現行ファーム Elastic Band", fontsize=14)
    whole.set_aspect("equal")
    whole.set_xlabel("X [mm]")
    whole.set_ylabel("Y [mm]")
    whole.grid(alpha=0.2)
    whole.legend(loc="best")

    distance_m = cumulative_distance_mm(path.x_mm, path.y_mm) * np.float32(0.001)
    slew.plot(distance_m, np.abs(result.slew_per_m2), color="#2563a6", lw=1.4)
    slew.scatter(
        [distance_m[peak]], [abs(result.slew_per_m2[peak])],
        marker="x", s=55, linewidths=2.0, color="#c2410c",
    )
    slew.set_title("経路全体の曲率変化率")
    slew.set_xlabel("経路距離 [m]")
    slew.set_ylabel("|dκ/ds| [1/m²]")
    slew.grid(alpha=0.2)

    offset_ok = metrics.max_offset_mm <= OFFSET_WARNING_MM
    radius_ok = metrics.min_radius_mm >= float(MIN_RADIUS_MM)
    status = "警告なし" if offset_ok and radius_ok else "要確認"
    summary = (
        f"{status}\n"
        f"経路短縮率: {metrics.shortening_percent:.3f}%\n"
        f"最大オフセット: {metrics.max_offset_mm:.1f} mm\n"
        f"オフセット警告値: {OFFSET_WARNING_MM:.0f} mm\n"
        f"最小半径: {metrics.min_radius_mm:.1f} mm\n"
        f"生成時の半径目標: {float(MIN_RADIUS_MM):.0f} mm\n"
        f"最大曲率変化率: {metrics.max_curvature_slew_per_m2:.1f} [1/m²]\n\n"
        f"現行設定\n"
        f"Elastic Band反復: {ITERATIONS}回\n"
        f"端部固定: {EDGE_KEEP_SEG * 10} mm\n"
        f"端部遷移: {EDGE_BLEND_SEG * 10} mm\n\n"
        "※75 mmは生成時の強制制限ではなく、\n"
        "  DUMP上の警告判定です。"
    )
    notes.axis("off")
    notes.text(0.02, 0.97, summary, va="top", ha="left", fontsize=12, linespacing=1.45)
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output.resolve()


def write_comparison_png(
    destination: str | FilePath,
    course: CoursePath,
    current_path: Path,
    continuous_path: Path,
) -> FilePath:
    """原コース・現行ファーム・曲率連続経路を同じ線幅で比較する。"""

    import matplotlib.pyplot as plt

    settings = Settings()
    current_result = evaluate(course, current_path, settings)
    continuous_result = evaluate(course, continuous_path, settings)
    output = FilePath(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "DejaVu Sans"]
    figure = plt.figure(figsize=(14, 9), dpi=150, facecolor="white")
    grid = figure.add_gridspec(2, 2, width_ratios=(1.45, 1.0), wspace=0.25, hspace=0.25)
    whole = figure.add_subplot(grid[:, 0])
    summary_axis = figure.add_subplot(grid[:, 1])

    whole.plot(course.x_mm, course.y_mm, color="#7d858d", lw=1.8, label="原コース")
    whole.plot(current_path.x_mm, current_path.y_mm, color="#2563a6", lw=1.8, label="現行ファーム")
    whole.plot(
        continuous_path.x_mm,
        continuous_path.y_mm,
        color="#087f5b",
        lw=1.8,
        label="曲率連続方式",
    )
    peak = continuous_result.peak_index
    whole.scatter(
        [continuous_path.x_mm[peak]], [continuous_path.y_mm[peak]],
        marker="x", s=65, linewidths=2.0, color="#c2410c",
        label="曲率変化の最大位置", zorder=5,
    )
    year = course.course_id[:4]
    whole.set_title(f"{year}年 全日本 ― 3経路比較", fontsize=14)
    whole.set_aspect("equal")
    whole.set_xlabel("X [mm]")
    whole.set_ylabel("Y [mm]")
    whole.grid(alpha=0.2)
    whole.legend(loc="best")

    current_metrics = current_result.metrics
    continuous_metrics = continuous_result.metrics
    def judgment(metrics: object) -> str:
        return "OK" if metrics.max_offset_mm <= 100.0 and metrics.min_radius_mm >= 60.0 else "要確認"

    summary_axis.axis("off")
    summary_axis.set_title("比較結果", loc="left", fontsize=14, pad=12)
    summary_axis.text(
        0.02,
        0.93,
        "線の意味",
        va="top",
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    summary_axis.text(0.05, 0.885, "━  原コース", color="#7d858d", fontsize=11, va="top")
    summary_axis.text(0.05, 0.845, "━  現行ファーム Elastic Band", color="#2563a6", fontsize=11, va="top")
    summary_axis.text(0.05, 0.805, "━  曲率連続方式", color="#087f5b", fontsize=11, va="top")

    table_rows = [
        [
            "現行\nファーム",
            f"{current_metrics.shortening_percent:.3f}%",
            f"{current_metrics.max_offset_mm:.1f} mm",
            f"{current_metrics.min_radius_mm:.1f} mm",
            judgment(current_metrics),
        ],
        [
            "曲率連続\n方式",
            f"{continuous_metrics.shortening_percent:.3f}%",
            f"{continuous_metrics.max_offset_mm:.1f} mm",
            f"{continuous_metrics.min_radius_mm:.1f} mm",
            judgment(continuous_metrics),
        ],
    ]
    table = summary_axis.table(
        cellText=table_rows,
        colLabels=["経路", "短縮率", "最大\nオフセット", "最小半径", "判定"],
        colWidths=[0.22, 0.18, 0.23, 0.19, 0.16],
        cellLoc="center",
        bbox=[0.02, 0.48, 0.96, 0.27],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#c8ccd1")
        if row == 0:
            cell.set_facecolor("#eef1f4")
            cell.set_text_props(fontweight="bold")
        elif row == 1:
            cell.set_facecolor("#eef5fb")
        elif row == 2:
            cell.set_facecolor("#edf8f3")

    summary_axis.text(
        0.02,
        0.40,
        "評価の目安",
        va="top",
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    summary_axis.text(
        0.04,
        0.35,
        "最大オフセット: 100 mm以下\n"
        "最小半径: 60 mm以上\n"
        "曲率連続方式は、現行ファーム経路を\n"
        "基準に長い直線を滑らかに接続した経路です。",
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.5,
    )
    summary_axis.text(
        0.02,
        0.11,
        "※ファーム側の75 mmは警告判定であり、\n"
        "　経路生成を止める強制制限ではありません。",
        va="top",
        ha="left",
        fontsize=10,
        color="#555b61",
        linespacing=1.4,
    )
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output.resolve()
