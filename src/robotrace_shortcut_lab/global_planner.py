from __future__ import annotations

from dataclasses import dataclass, replace
from math import cos, floor, pi, sin
from time import perf_counter

import numpy as np

from .course import load_board_boundary
from .geometry import (
    cumulative_distance_m,
    curvature_slew_per_m2,
    radius_mm,
    self_intersection_count,
    signed_curvature_per_m,
)
from .model import (
    BoardBoundary,
    Comparison,
    Course,
    EvaluatedGlobalPath,
    GlobalComparison,
    GlobalMetrics,
    GlobalPath,
    GlobalSearchResult,
    GlobalSearchStats,
    PlannerConfig,
)
from .portable import plan_speed, run_comparison


@dataclass(frozen=True)
class _ShortcutEdge:
    edge_id: int
    start_node: int
    end_node: int
    start_index: int
    end_index: int
    kind: str
    x_mm: np.ndarray
    y_mm: np.ndarray
    source_progress_index: np.ndarray
    approximate_time_s: float
    skipped_source_mm: float
    crossing_count: int
    shallow_crossing_count: int
    crossing_angles_deg: tuple[float, ...]
    crossing_source_indices: tuple[int, ...]
    past_crossing_count: int
    future_crossing_count: int
    parallel_distance_mm: float
    reference_valid: bool
    reference_violation: str


@dataclass(frozen=True)
class LineInteractionDetails:
    """他白線との交差を候補辺ごとに保持するPC評価値。"""

    crossing_count: int
    shallow_crossing_count: int
    parallel_distance_mm: float
    source_indices: tuple[int, ...]
    crossing_angles_deg: tuple[float, ...]
    past_crossing_count: int
    future_crossing_count: int
    crossing_path: np.ndarray


class _PolylineIndex:
    """候補と白線の距離・交差を汎用ライブラリなしで調べる空間格子。"""

    def __init__(self, x_mm: np.ndarray, y_mm: np.ndarray, cell_mm: float = 150.0):
        self.x = np.asarray(x_mm, dtype=np.float64)
        self.y = np.asarray(y_mm, dtype=np.float64)
        self.cell_mm = float(cell_mm)
        self.buckets: dict[tuple[int, int], list[int]] = {}
        for index in range(self.x.size - 1):
            min_x = min(self.x[index], self.x[index + 1])
            max_x = max(self.x[index], self.x[index + 1])
            min_y = min(self.y[index], self.y[index + 1])
            max_y = max(self.y[index], self.y[index + 1])
            x0 = int(floor(min_x / self.cell_mm))
            x1 = int(floor(max_x / self.cell_mm))
            y0 = int(floor(min_y / self.cell_mm))
            y1 = int(floor(max_y / self.cell_mm))
            for cell_y in range(y0, y1 + 1):
                for cell_x in range(x0, x1 + 1):
                    self.buckets.setdefault((cell_x, cell_y), []).append(index)

    def nearby_segments(self, x: float, y: float, rings: int = 2) -> tuple[int, ...]:
        cell_x = int(floor(x / self.cell_mm))
        cell_y = int(floor(y / self.cell_mm))
        found: set[int] = set()
        for offset_y in range(-rings, rings + 1):
            for offset_x in range(-rings, rings + 1):
                found.update(self.buckets.get((cell_x + offset_x, cell_y + offset_y), ()))
        return tuple(sorted(found))

    def nearby_bbox(
        self, min_x: float, max_x: float, min_y: float, max_y: float
    ) -> tuple[int, ...]:
        x0 = int(floor(min_x / self.cell_mm))
        x1 = int(floor(max_x / self.cell_mm))
        y0 = int(floor(min_y / self.cell_mm))
        y1 = int(floor(max_y / self.cell_mm))
        found: set[int] = set()
        for cell_y in range(y0 - 1, y1 + 2):
            for cell_x in range(x0 - 1, x1 + 2):
                found.update(self.buckets.get((cell_x, cell_y), ()))
        return tuple(sorted(found))

    def nearest(self, x: float, y: float) -> tuple[float, int]:
        candidates = self.nearby_segments(x, y, 2)
        if not candidates:
            candidates = tuple(range(self.x.size - 1))
        best_distance = float("inf")
        best_index = -1
        for index in candidates:
            distance = _point_segment_distance(
                x,
                y,
                self.x[index],
                self.y[index],
                self.x[index + 1],
                self.y[index + 1],
            )
            if distance < best_distance:
                best_distance = distance
                best_index = index
        return best_distance, best_index


class _LineTubeGrid:
    """規定最大車体でも白線へ届かない候補を早期棄却する固定格子。"""

    def __init__(self, x_mm: np.ndarray, y_mm: np.ndarray, limit_mm: float):
        self.cell_mm = 25.0
        padding = limit_mm + self.cell_mm * 2.0
        self.min_x = floor((float(np.min(x_mm)) - padding) / self.cell_mm) * self.cell_mm
        self.min_y = floor((float(np.min(y_mm)) - padding) / self.cell_mm) * self.cell_mm
        max_x = float(np.max(x_mm)) + padding
        max_y = float(np.max(y_mm)) + padding
        width = int(np.ceil((max_x - self.min_x) / self.cell_mm)) + 1
        height = int(np.ceil((max_y - self.min_y) / self.cell_mm)) + 1
        self.mask = np.zeros((height, width), dtype=np.bool_)
        radius_cells = int(np.ceil((limit_mm + self.cell_mm * np.sqrt(2.0)) / self.cell_mm))
        offsets = [
            (ox, oy)
            for oy in range(-radius_cells, radius_cells + 1)
            for ox in range(-radius_cells, radius_cells + 1)
            if np.hypot(ox * self.cell_mm, oy * self.cell_mm)
            <= limit_mm + self.cell_mm * np.sqrt(2.0)
        ]
        point_x = np.floor((np.asarray(x_mm, dtype=np.float64) - self.min_x) / self.cell_mm).astype(int)
        point_y = np.floor((np.asarray(y_mm, dtype=np.float64) - self.min_y) / self.cell_mm).astype(int)
        for ox, oy in offsets:
            ix = point_x + ox
            iy = point_y + oy
            valid = (
                (ix >= 0)
                & (ix < width)
                & (iy >= 0)
                & (iy < height)
            )
            self.mask[iy[valid], ix[valid]] = True

    def contains_path(self, x_mm: np.ndarray, y_mm: np.ndarray) -> bool:
        ix = np.floor((np.asarray(x_mm, dtype=np.float64) - self.min_x) / self.cell_mm).astype(int)
        iy = np.floor((np.asarray(y_mm, dtype=np.float64) - self.min_y) / self.cell_mm).astype(int)
        valid = (
            (ix >= 0)
            & (ix < self.mask.shape[1])
            & (iy >= 0)
            & (iy < self.mask.shape[0])
        )
        return bool(np.all(valid) and np.all(self.mask[iy, ix]))


def _point_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    dx = bx - ax
    dy = by - ay
    denominator = dx * dx + dy * dy
    if denominator <= 1.0e-12:
        return float(np.hypot(px - ax, py - ay))
    ratio = ((px - ax) * dx + (py - ay) * dy) / denominator
    ratio = min(1.0, max(0.0, ratio))
    return float(np.hypot(px - (ax + ratio * dx), py - (ay + ratio * dy)))


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _segment_intersection(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> bool:
    abx, aby = bx - ax, by - ay
    cdx, cdy = dx - cx, dy - cy
    return bool(
        _cross(abx, aby, cx - ax, cy - ay)
        * _cross(abx, aby, dx - ax, dy - ay)
        < -1.0e-8
        and _cross(cdx, cdy, ax - cx, ay - cy)
        * _cross(cdx, cdy, bx - cx, by - cy)
        < -1.0e-8
    )


def _unit_tangents(x_mm: np.ndarray, y_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = np.gradient(np.asarray(x_mm, dtype=np.float64))
    dy = np.gradient(np.asarray(y_mm, dtype=np.float64))
    length = np.maximum(np.hypot(dx, dy), 1.0e-9)
    return dx / length, dy / length


def _angle_deg(ax: float, ay: float, bx: float, by: float) -> float:
    dot = ax * bx + ay * by
    cross = ax * by - ay * bx
    return float(abs(np.rad2deg(np.arctan2(cross, dot))))


def _resample_xy(
    x_mm: np.ndarray, y_mm: np.ndarray, interval_mm: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x_mm, dtype=np.float64)
    y = np.asarray(y_mm, dtype=np.float64)
    segment = np.hypot(np.diff(x), np.diff(y))
    cumulative = np.concatenate((np.zeros(1), np.cumsum(segment)))
    keep = np.concatenate(([True], np.diff(cumulative) > 1.0e-6))
    x = x[keep]
    y = y[keep]
    cumulative = cumulative[keep]
    if cumulative.size < 2 or cumulative[-1] <= 1.0e-6:
        return x.astype(np.float32), y.astype(np.float32), cumulative, cumulative
    target = np.arange(0.0, cumulative[-1], interval_mm, dtype=np.float64)
    if target.size == 0 or target[-1] < cumulative[-1] - 1.0e-6:
        target = np.append(target, cumulative[-1])
    return (
        np.interp(target, cumulative, x).astype(np.float32),
        np.interp(target, cumulative, y).astype(np.float32),
        target,
        cumulative,
    )


def _cubic_hermite(
    p0: np.ndarray,
    p1: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    scale: float,
    count: int,
) -> np.ndarray:
    u = np.linspace(0.0, 1.0, count, dtype=np.float64)[:, None]
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2
    return h00 * p0 + h10 * (t0 * scale) + h01 * p1 + h11 * (t1 * scale)


def _quintic_hermite(
    p0: np.ndarray,
    p1: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    scale: float,
    count: int,
    second0: np.ndarray | None = None,
    second1: np.ndarray | None = None,
) -> np.ndarray:
    u = np.linspace(0.0, 1.0, count, dtype=np.float64)[:, None]
    h00 = 1.0 - 10.0 * u**3 + 15.0 * u**4 - 6.0 * u**5
    h10 = u - 6.0 * u**3 + 8.0 * u**4 - 3.0 * u**5
    h20 = 0.5 * (u**2 - 3.0 * u**3 + 3.0 * u**4 - u**5)
    h01 = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    h11 = -4.0 * u**3 + 7.0 * u**4 - 3.0 * u**5
    h21 = 0.5 * (u**3 - 2.0 * u**4 + u**5)
    a0 = np.zeros(2) if second0 is None else second0
    a1 = np.zeros(2) if second1 is None else second1
    return (
        h00 * p0
        + h10 * (t0 * scale)
        + h20 * a0
        + h01 * p1
        + h11 * (t1 * scale)
        + h21 * a1
    )


def _lightweight_biarc(
    p0: np.ndarray,
    p1: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    count: int,
) -> np.ndarray:
    chord = p1 - p0
    chord_length = float(np.hypot(chord[0], chord[1]))
    chord_unit = chord / max(chord_length, 1.0e-9)
    middle_tangent = t0 + t1 + chord_unit
    middle_tangent /= max(float(np.hypot(*middle_tangent)), 1.0e-9)
    middle = 0.5 * (p0 + p1)
    first_count = max(3, count // 2 + 1)
    second_count = max(3, count - first_count + 2)
    first = _cubic_hermite(p0, middle, t0, middle_tangent, chord_length * 0.28, first_count)
    second = _cubic_hermite(middle, p1, middle_tangent, t1, chord_length * 0.28, second_count)
    return np.vstack((first[:-1], second))


def generate_edge_shapes(
    p0: np.ndarray,
    p1: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    curvature0_per_m: float,
    curvature1_per_m: float,
    interval_mm: float = 20.0,
) -> tuple[tuple[str, np.ndarray], ...]:
    """同じアンカー対へ決定論的な5種類以上の接続曲線を作る。"""

    chord = p1 - p0
    length = float(np.hypot(chord[0], chord[1]))
    count = max(3, int(np.ceil(length / interval_mm)) + 1)
    chord_unit = chord / max(length, 1.0e-9)
    line = np.linspace(p0, p1, count, dtype=np.float64)
    cubic = _cubic_hermite(p0, p1, t0, t1, length * 0.48, count)
    quintic = _quintic_hermite(p0, p1, t0, t1, length * 0.52, count)
    transition = _quintic_hermite(p0, p1, t0, t1, length * 0.34, count)
    normal0 = np.array([-t0[1], t0[0]])
    normal1 = np.array([-t1[1], t1[0]])
    scale = length * 0.48
    second0 = normal0 * (curvature0_per_m * 0.001) * scale**2
    second1 = normal1 * (curvature1_per_m * 0.001) * scale**2
    g2 = _quintic_hermite(
        p0, p1, t0, t1, scale, count, second0, second1
    )
    biarc = _lightweight_biarc(p0, p1, t0, t1, count)
    # 直線候補は接線不一致を別の幾何検査で棄却する。
    _ = chord_unit
    return (
        ("直線", line),
        ("入口遷移＋直線＋出口遷移", transition),
        ("3次Hermite", cubic),
        ("5次Hermite", quintic),
        ("G2近似", g2),
        ("軽量biarc", biarc),
    )


def _rdp_indices(x_mm: np.ndarray, y_mm: np.ndarray, tolerance_mm: float) -> set[int]:
    """再帰を使わないRDP代表点抽出。"""

    keep = {0, x_mm.size - 1}
    stack: list[tuple[int, int]] = [(0, x_mm.size - 1)]
    while stack:
        start, finish = stack.pop()
        if finish - start <= 1:
            continue
        ax, ay = float(x_mm[start]), float(y_mm[start])
        bx, by = float(x_mm[finish]), float(y_mm[finish])
        maximum = -1.0
        maximum_index = -1
        for index in range(start + 1, finish):
            distance = _point_segment_distance(
                float(x_mm[index]), float(y_mm[index]), ax, ay, bx, by
            )
            if distance > maximum:
                maximum = distance
                maximum_index = index
        if maximum > tolerance_mm and maximum_index > start:
            keep.add(maximum_index)
            stack.append((start, maximum_index))
            stack.append((maximum_index, finish))
    return keep


def extract_anchor_indices(
    course: Course,
    baseline_x_mm: np.ndarray,
    baseline_y_mm: np.ndarray,
    config: PlannerConfig,
    mode: str,
) -> np.ndarray:
    """距離、RDP、曲率、速度支配、自己近接から進行順アンカーを選ぶ。"""

    limit = (
        config.reference_anchor_limit
        if mode == "reference"
        else config.embedded_anchor_limit
    )
    base_spacing_mm = 220.0 if mode == "reference" else 500.0
    spacing_mm = max(
        base_spacing_mm,
        float(course.distance_mm[-1]) / max(2.0, limit * 0.70),
    )
    scores: dict[int, float] = {0: 1.0e9, course.point_count - 1: 1.0e9}
    mandatory = {0, course.point_count - 1}

    for distance in np.arange(spacing_mm, course.distance_mm[-1], spacing_mm):
        index = int(np.searchsorted(course.distance_mm, distance))
        scores[index] = max(scores.get(index, 0.0), 50.0)
        mandatory.add(index)

    rdp_tolerance = 22.0 if mode == "reference" else 45.0
    for index in _rdp_indices(course.x_mm, course.y_mm, rdp_tolerance):
        scores[index] = max(scores.get(index, 0.0), 65.0)

    curvature = signed_curvature_per_m(
        course.x_mm, course.y_mm, config.radius_window
    )
    magnitude = np.abs(curvature)
    threshold = float(np.percentile(magnitude, 70.0))
    half_width = max(config.radius_window, 10)
    for index in range(half_width, course.point_count - half_width):
        if curvature[index - 1] * curvature[index + 1] < 0.0:
            scores[index] = max(scores.get(index, 0.0), 95.0)
        if magnitude[index] >= max(threshold, 1.0e-5) and magnitude[index] >= np.max(
            magnitude[index - half_width : index + half_width + 1]
        ):
            scores[index] = max(scores.get(index, 0.0), 105.0 + magnitude[index])

    plan = plan_speed(baseline_x_mm, baseline_y_mm, config)
    low = plan.gfcp_limit_mps <= config.min_speed_mps * 1.12
    changes = np.flatnonzero(np.diff(low.astype(np.int8)) != 0) + 1
    for index in changes:
        scores[int(index)] = max(scores.get(int(index), 0.0), 100.0)
    starts = np.flatnonzero(low & np.concatenate(([True], ~low[:-1])))
    finishes = np.flatnonzero(low & np.concatenate((~low[1:], [True])))
    for start, finish in zip(starts, finishes, strict=True):
        for index in (int(start), int((start + finish) // 2), int(finish)):
            scores[index] = max(scores.get(index, 0.0), 100.0)
    reason_changes = np.flatnonzero(np.diff(plan.limit_reason.astype(np.int16)) != 0) + 1
    for index in reason_changes:
        scores[int(index)] = max(scores.get(int(index), 0.0), 72.0)

    # 遠い進行index同士が空間的に近づく箇所は、乗り換え候補として優先する。
    point_cell = 250.0
    buckets: dict[tuple[int, int], list[int]] = {}
    stride = 4 if mode == "reference" else 9
    for index in range(0, course.point_count, stride):
        cell = (
            int(floor(float(course.x_mm[index]) / point_cell)),
            int(floor(float(course.y_mm[index]) / point_cell)),
        )
        best: tuple[float, int] | None = None
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                for previous in buckets.get((cell[0] + ox, cell[1] + oy), ()):
                    progress = float(course.distance_mm[index] - course.distance_mm[previous])
                    if progress < 500.0:
                        continue
                    distance = float(
                        np.hypot(
                            course.x_mm[index] - course.x_mm[previous],
                            course.y_mm[index] - course.y_mm[previous],
                        )
                    )
                    if distance <= 420.0 and (best is None or distance < best[0]):
                        best = (distance, previous)
        if best is not None:
            previous = best[1]
            scores[previous] = max(scores.get(previous, 0.0), 125.0 - best[0] * 0.02)
            scores[index] = max(scores.get(index, 0.0), 125.0 - best[0] * 0.02)
        buckets.setdefault(cell, []).append(index)

    remaining = max(0, limit - len(mandatory))
    additional = [index for index in scores if index not in mandatory]
    additional.sort(key=lambda index: (-scores[index], index))
    mandatory.update(additional[:remaining])
    return np.asarray(sorted(mandatory), dtype=np.int32)


def _inside_board_union(x: float, y: float, boundary: BoardBoundary) -> bool:
    return any(
        min_x - 1.0e-6 <= x <= max_x + 1.0e-6
        and min_y - 1.0e-6 <= y <= max_y + 1.0e-6
        for min_x, max_x, min_y, max_y in boundary.rectangles_mm
    )


def board_path_is_inside(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    boundary: BoardBoundary,
    margin_mm: float,
) -> bool:
    """中心と車体外周16点が板セル和集合の中にあるか調べる。"""

    angles = np.arange(16, dtype=np.float64) * (2.0 * pi / 16.0)
    offsets = [(0.0, 0.0)] + [
        (margin_mm * cos(angle), margin_mm * sin(angle)) for angle in angles
    ]
    stride = max(1, x_mm.size // 500)
    indices = list(range(0, x_mm.size, stride))
    if indices[-1] != x_mm.size - 1:
        indices.append(x_mm.size - 1)
    for index in indices:
        x = float(x_mm[index])
        y = float(y_mm[index])
        for offset_x, offset_y in offsets:
            if not _inside_board_union(x + offset_x, y + offset_y, boundary):
                return False
    return True


def _line_coverage_is_possible(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    source_index: _PolylineIndex,
    config: PlannerConfig,
) -> bool:
    absolute_limit = config.rule_max_robot_radius_mm + config.rule_line_half_width_mm
    for x, y in zip(x_mm, y_mm, strict=True):
        distance, _ = source_index.nearest(float(x), float(y))
        if distance > absolute_limit + 1.0e-5:
            return False
    return True


def line_interaction_metrics(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    source_progress_index: np.ndarray,
    source_index: _PolylineIndex,
    config: PlannerConfig,
) -> tuple[int, int, float, tuple[int, ...]]:
    """他白線との交差、浅角交差、平行近接距離、交差先indexを返す。"""

    details = _line_interaction_details(
        x_mm, y_mm, source_progress_index, source_index, config
    )
    return (
        details.crossing_count,
        details.shallow_crossing_count,
        details.parallel_distance_mm,
        details.source_indices,
    )


def analyze_line_interactions(
    source_x_mm: np.ndarray,
    source_y_mm: np.ndarray,
    path_x_mm: np.ndarray,
    path_y_mm: np.ndarray,
    source_progress_index: np.ndarray,
    config: PlannerConfig,
) -> tuple[int, int, float, tuple[int, ...]]:
    """テスト・PC解析向けの他白線交差評価入口。"""

    return line_interaction_metrics(
        path_x_mm,
        path_y_mm,
        source_progress_index,
        _PolylineIndex(source_x_mm, source_y_mm),
        config,
    )


def analyze_line_interactions_detailed(
    source_x_mm: np.ndarray,
    source_y_mm: np.ndarray,
    path_x_mm: np.ndarray,
    path_y_mm: np.ndarray,
    source_progress_index: np.ndarray,
    config: PlannerConfig,
) -> LineInteractionDetails:
    """交差角、交差先index、過去／未来分類を含む詳細値を返す。"""

    return _line_interaction_details(
        path_x_mm,
        path_y_mm,
        source_progress_index,
        _PolylineIndex(source_x_mm, source_y_mm),
        config,
    )


def edge_connection_is_valid(
    points_mm: np.ndarray,
    start_tangent: np.ndarray,
    end_tangent: np.ndarray,
    max_angle_deg: float,
) -> bool:
    """入口・出口の接線角が異常な候補を棄却する共通判定。"""

    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        return False
    first = points[1] - points[0]
    last = points[-1] - points[-2]
    if np.hypot(*first) <= 1.0e-6 or np.hypot(*last) <= 1.0e-6:
        return False
    return bool(
        _angle_deg(first[0], first[1], start_tangent[0], start_tangent[1])
        <= max_angle_deg
        and _angle_deg(last[0], last[1], end_tangent[0], end_tangent[1])
        <= max_angle_deg
    )


def _line_interaction_details(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    source_progress_index: np.ndarray,
    source_index: _PolylineIndex,
    config: PlannerConfig,
) -> LineInteractionDetails:
    """交差した大域経路点も含む詳細値を返す。"""

    crossings: dict[int, tuple[float, bool]] = {}
    crossing_path = np.zeros(x_mm.size, dtype=np.bool_)
    shallow = 0
    parallel_mm = 0.0
    for index in range(x_mm.size - 1):
        ax, ay = float(x_mm[index]), float(y_mm[index])
        bx, by = float(x_mm[index + 1]), float(y_mm[index + 1])
        path_dx, path_dy = bx - ax, by - ay
        path_length = float(np.hypot(path_dx, path_dy))
        if path_length <= 1.0e-6:
            continue
        expected = 0.5 * float(
            source_progress_index[index] + source_progress_index[index + 1]
        )
        nearby = source_index.nearby_bbox(
            min(ax, bx), max(ax, bx), min(ay, by), max(ay, by)
        )
        for source_segment in nearby:
            if abs(source_segment - expected) <= 28.0:
                continue
            cx = float(source_index.x[source_segment])
            cy = float(source_index.y[source_segment])
            dx = float(source_index.x[source_segment + 1])
            dy = float(source_index.y[source_segment + 1])
            if _segment_intersection(ax, ay, bx, by, cx, cy, dx, dy):
                if source_segment not in crossings:
                    angle = _angle_deg(path_dx, path_dy, dx - cx, dy - cy)
                    angle = min(angle, 180.0 - angle)
                    if angle < 35.0:
                        shallow += 1
                    crossings[source_segment] = (angle, source_segment < expected)
                crossing_path[index : index + 2] = True
        middle_x = 0.5 * (ax + bx)
        middle_y = 0.5 * (ay + by)
        distance, near_segment = source_index.nearest(middle_x, middle_y)
        if near_segment >= 0 and abs(near_segment - expected) > 28.0:
            source_dx = source_index.x[near_segment + 1] - source_index.x[near_segment]
            source_dy = source_index.y[near_segment + 1] - source_index.y[near_segment]
            angle = _angle_deg(path_dx, path_dy, source_dx, source_dy)
            angle = min(angle, 180.0 - angle)
            if (
                distance < config.line_parallel_warning_distance_mm
                and angle < config.line_parallel_warning_angle_deg
            ):
                parallel_mm += path_length
    source_indices = tuple(sorted(crossings))
    crossing_angles = tuple(crossings[index][0] for index in source_indices)
    past = sum(crossings[index][1] for index in source_indices)
    return LineInteractionDetails(
        len(source_indices),
        shallow,
        parallel_mm,
        source_indices,
        crossing_angles,
        past,
        len(source_indices) - past,
        crossing_path,
    )


def _approximate_time(x_mm: np.ndarray, y_mm: np.ndarray, config: PlannerConfig) -> float:
    distance_m = cumulative_distance_m(x_mm, y_mm)
    curvature = np.abs(signed_curvature_per_m(x_mm, y_mm, config.radius_window))
    radius_ratio = np.divide(
        1.0,
        0.1 * curvature,
        out=np.full_like(curvature, np.inf),
        where=curvature > 1.0e-9,
    )
    speed = np.clip(
        config.gfcp_reference_speed_mps
        * np.power(radius_ratio, config.gfcp_exponent),
        config.min_speed_mps,
        config.max_speed_mps,
    )
    if speed.size < 2:
        return float("inf")
    return float(
        np.sum(2.0 * np.diff(distance_m) / np.maximum(speed[:-1] + speed[1:], 1.0e-6))
    )


def _make_edges(
    course: Course,
    comparison: Comparison,
    anchors: np.ndarray,
    config: PlannerConfig,
    mode: str,
    boundary: BoardBoundary | None,
    *,
    apply_theoretical_line_tube: bool = True,
    prefilter_line_radius_mm: float | None = None,
) -> tuple[list[_ShortcutEdge], float, int]:
    start_time = perf_counter()
    baseline = comparison.best.path
    tangent_x, tangent_y = _unit_tangents(baseline.x_mm, baseline.y_mm)
    curvature = signed_curvature_per_m(
        baseline.x_mm, baseline.y_mm, config.radius_window
    )
    line_tube = _LineTubeGrid(
        course.x_mm,
        course.y_mm,
        (
            prefilter_line_radius_mm
            if prefilter_line_radius_mm is not None
            else config.rule_max_robot_radius_mm + config.rule_line_half_width_mm
        ),
    )
    source_index = _PolylineIndex(course.x_mm, course.y_mm)
    max_skip_mm = (
        config.reference_max_skip_mm
        if mode == "reference"
        else config.embedded_max_skip_mm
    )
    edge_limit = (
        config.reference_edge_limit if mode == "reference" else config.embedded_edge_limit
    )
    max_pairs_per_start = 42 if mode == "reference" else 10
    edges: list[_ShortcutEdge] = []
    considered = 0
    edge_id = 1

    for start_node in range(anchors.size - 2):
        start_index = int(anchors[start_node])
        pair_scores: list[tuple[float, int]] = []
        for end_node in range(start_node + 2, anchors.size):
            end_index = int(anchors[end_node])
            skipped = float(course.distance_mm[end_index] - course.distance_mm[start_index])
            if skipped > max_skip_mm:
                break
            chord = float(
                np.hypot(
                    baseline.x_mm[end_index] - baseline.x_mm[start_index],
                    baseline.y_mm[end_index] - baseline.y_mm[start_index],
                )
            )
            saving = skipped - chord
            if skipped < config.shortcut_min_skip_mm or saving < config.shortcut_min_saving_mm:
                continue
            # 大きな短縮と自己近接を優先し、固定本数だけ詳細形状を作る。
            score = saving + max(0.0, 500.0 - chord) * 0.8
            pair_scores.append((score, end_node))
        pair_scores.sort(key=lambda item: (-item[0], item[1]))
        for _, end_node in pair_scores[:max_pairs_per_start]:
            end_index = int(anchors[end_node])
            p0 = np.array(
                [baseline.x_mm[start_index], baseline.y_mm[start_index]], dtype=np.float64
            )
            p1 = np.array(
                [baseline.x_mm[end_index], baseline.y_mm[end_index]], dtype=np.float64
            )
            t0 = np.array([tangent_x[start_index], tangent_y[start_index]])
            t1 = np.array([tangent_x[end_index], tangent_y[end_index]])
            skipped = float(course.distance_mm[end_index] - course.distance_mm[start_index])
            for kind, points in generate_edge_shapes(
                p0,
                p1,
                t0,
                t1,
                float(curvature[start_index]),
                float(curvature[end_index]),
            ):
                considered += 1
                if len(edges) >= edge_limit:
                    break
                x, y, target, _ = _resample_xy(points[:, 0], points[:, 1], 20.0)
                if x.size < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
                    continue
                if not edge_connection_is_valid(
                    np.column_stack((x, y)),
                    t0,
                    t1,
                    config.connector_max_angle_deg,
                ):
                    continue
                # 辺生成中は中心だけを粗検査し、車体外形込みの板境界は
                # 上位K経路を結合した後の完全評価で確認する。
                if boundary is not None and not all(
                    _inside_board_union(float(px), float(py), boundary)
                    for px, py in zip(x[::5], y[::5], strict=True)
                ):
                    continue
                source_progress = np.interp(
                    target,
                    (0.0, max(float(target[-1]), 1.0e-6)),
                    (float(start_index), float(end_index)),
                ).astype(np.float32)
                line_valid = (
                    line_tube.contains_path(x, y)
                    if apply_theoretical_line_tube or prefilter_line_radius_mm is not None
                    else True
                )
                crossing_count = 0
                shallow = 0
                crossing_angles: tuple[float, ...] = ()
                crossing_indices: tuple[int, ...] = ()
                past_crossings = 0
                future_crossings = 0
                parallel_mm = 0.0
                reference_violations: list[str] = []
                if not line_valid:
                    reference_violations.append("白線投影")
                else:
                    # 20 mm候補点列を200 mmごとに間引いて候補辺全数を
                    # 評価する。上位Kと最終経路は10 mm点列で厳密再評価する。
                    sample = np.arange(0, x.size, 10, dtype=np.int32)
                    if sample.size == 0 or sample[-1] != x.size - 1:
                        sample = np.append(sample, x.size - 1)
                    interactions = _line_interaction_details(
                        x[sample],
                        y[sample],
                        source_progress[sample],
                        source_index,
                        config,
                    )
                    crossing_count = interactions.crossing_count
                    shallow = interactions.shallow_crossing_count
                    crossing_angles = interactions.crossing_angles_deg
                    crossing_indices = interactions.source_indices
                    past_crossings = interactions.past_crossing_count
                    future_crossings = interactions.future_crossing_count
                    parallel_mm = interactions.parallel_distance_mm
                approximate = _approximate_time(x, y, config)
                approximate += (
                    crossing_count * config.line_crossing_penalty_s
                    + shallow * config.line_shallow_crossing_penalty_s
                    + parallel_mm * 0.001 * config.line_parallel_penalty_s_per_m
                )
                edges.append(
                    _ShortcutEdge(
                        edge_id,
                        start_node,
                        end_node,
                        start_index,
                        end_index,
                        kind,
                        x,
                        y,
                        source_progress,
                        approximate,
                        skipped,
                        crossing_count,
                        shallow,
                        crossing_angles,
                        crossing_indices,
                        past_crossings,
                        future_crossings,
                        parallel_mm,
                        not reference_violations,
                        "、".join(reference_violations),
                    )
                )
                edge_id += 1
            if len(edges) >= edge_limit:
                break
        if len(edges) >= edge_limit:
            break
    return edges, perf_counter() - start_time, considered


def _base_edge(
    node: int,
    anchors: np.ndarray,
    baseline_x: np.ndarray,
    baseline_y: np.ndarray,
    baseline_distance_m: np.ndarray,
    baseline_time_s: np.ndarray,
) -> _ShortcutEdge:
    start_index = int(anchors[node])
    end_index = int(anchors[node + 1])
    indices = np.arange(start_index, end_index + 1, dtype=np.float32)
    cost = float(baseline_time_s[end_index] - baseline_time_s[start_index])
    return _ShortcutEdge(
        -(node + 1),
        node,
        node + 1,
        start_index,
        end_index,
        "基準経路",
        baseline_x[start_index : end_index + 1].copy(),
        baseline_y[start_index : end_index + 1].copy(),
        indices,
        max(cost, 0.0),
        0.0,
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


def _k_best_edge_paths(
    anchors: np.ndarray,
    edges: list[_ShortcutEdge],
    base_edges: list[_ShortcutEdge],
    top_k: int,
    *,
    reference_only: bool,
) -> tuple[list[tuple[int, ...]], float]:
    start_time = perf_counter()
    all_edges = base_edges + [
        edge for edge in edges if edge.reference_valid or not reference_only
    ]
    by_start: list[list[_ShortcutEdge]] = [[] for _ in range(anchors.size)]
    edge_map = {edge.edge_id: edge for edge in all_edges}
    for edge in all_edges:
        by_start[edge.start_node].append(edge)
    for outgoing in by_start:
        outgoing.sort(key=lambda edge: (edge.end_node, edge.approximate_time_s, edge.edge_id))

    states: list[list[tuple[float, tuple[int, ...]]]] = [
        [] for _ in range(anchors.size)
    ]
    states[0] = [(0.0, ())]
    for node in range(anchors.size - 1):
        states[node].sort(key=lambda item: (item[0], item[1]))
        states[node] = states[node][:top_k]
        for cost, sequence in states[node]:
            for edge in by_start[node]:
                bucket = states[edge.end_node]
                bucket.append((cost + edge.approximate_time_s, sequence + (edge.edge_id,)))
                if len(bucket) > top_k * 4:
                    bucket.sort(key=lambda item: (item[0], item[1]))
                    del bucket[top_k:]
    final = sorted(states[-1], key=lambda item: (item[0], item[1]))[:top_k]
    # sequenceの復元側でも同じedge mapを再構築できるようIDだけ返す。
    _ = edge_map
    return [sequence for _, sequence in final], perf_counter() - start_time


def _geometry_arrays(
    x_mm: np.ndarray, y_mm: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = np.gradient(x_mm.astype(np.float64))
    dy = np.gradient(y_mm.astype(np.float64))
    yaw = np.unwrap(np.arctan2(dy, dx)).astype(np.float32)
    curvature = signed_curvature_per_m(x_mm, y_mm).astype(np.float32)
    slew = curvature_slew_per_m2(x_mm, y_mm).astype(np.float32)
    return yaw, curvature, slew


def _build_global_path(
    course: Course,
    sequence: tuple[int, ...],
    edge_map: dict[int, _ShortcutEdge],
    config: PlannerConfig,
    label: str,
    generation_s: float,
    *,
    allow_synthetic_progress: bool = True,
) -> GlobalPath:
    pieces_x: list[np.ndarray] = []
    pieces_y: list[np.ndarray] = []
    pieces_progress: list[np.ndarray] = []
    pieces_edge_id: list[np.ndarray] = []
    selected_edges: list[tuple[int, int, int, str]] = []
    crossing_edge_ids: set[int] = set()
    for position, edge_id in enumerate(sequence):
        edge = edge_map[edge_id]
        start = 0 if position == 0 else 1
        pieces_x.append(edge.x_mm[start:])
        pieces_y.append(edge.y_mm[start:])
        pieces_progress.append(edge.source_progress_index[start:])
        pieces_edge_id.append(
            np.full(edge.x_mm.size - start, edge.edge_id, dtype=np.int32)
        )
        if edge.edge_id > 0:
            selected_edges.append(
                (edge.edge_id, edge.start_index, edge.end_index, edge.kind)
            )
            if edge.crossing_count:
                crossing_edge_ids.add(edge.edge_id)
    raw_x = np.concatenate(pieces_x).astype(np.float64)
    raw_y = np.concatenate(pieces_y).astype(np.float64)
    raw_progress = np.concatenate(pieces_progress).astype(np.float64)
    raw_edge_id = np.concatenate(pieces_edge_id)
    segment = np.hypot(np.diff(raw_x), np.diff(raw_y))
    raw_distance = np.concatenate((np.zeros(1), np.cumsum(segment)))
    keep = np.concatenate(([True], np.diff(raw_distance) > 1.0e-6))
    raw_x = raw_x[keep]
    raw_y = raw_y[keep]
    raw_progress = raw_progress[keep]
    raw_edge_id = raw_edge_id[keep]
    raw_distance = raw_distance[keep]
    target = np.arange(
        0.0, raw_distance[-1], config.global_resample_interval_mm, dtype=np.float64
    )
    if target.size == 0 or target[-1] < raw_distance[-1] - 1.0e-6:
        target = np.append(target, raw_distance[-1])
    x = np.interp(target, raw_distance, raw_x).astype(np.float32)
    y = np.interp(target, raw_distance, raw_y).astype(np.float32)
    if allow_synthetic_progress:
        # 旧理論下限の再現専用。競技合法経路はこの値を使わず、
        # swept-footprintの実接触DPを完了してから値を入れる。
        progress = np.maximum.accumulate(
            np.interp(target, raw_distance, raw_progress)
        ).astype(np.float32)
        source_distance = np.interp(
            progress,
            np.arange(course.point_count, dtype=np.float64),
            course.distance_mm.astype(np.float64),
        ).astype(np.float32)
    else:
        progress = np.full(target.size, -1.0, dtype=np.float32)
        source_distance = np.full(target.size, -1.0, dtype=np.float32)
    nearest_raw = np.minimum(
        np.searchsorted(raw_distance, target, side="right") - 1,
        raw_edge_id.size - 1,
    )
    nearest_raw = np.maximum(nearest_raw, 0)
    shortcut_edge_id = raw_edge_id[nearest_raw].astype(np.int32)
    deliberate = np.isin(shortcut_edge_id, tuple(crossing_edge_ids))
    x[0], y[0] = course.x_mm[0], course.y_mm[0]
    x[-1], y[-1] = course.x_mm[-1], course.y_mm[-1]
    yaw, curvature, slew = _geometry_arrays(x, y)
    return GlobalPath(
        label,
        x,
        y,
        target.astype(np.float32),
        progress,
        source_distance,
        shortcut_edge_id,
        deliberate.astype(np.bool_),
        yaw,
        curvature,
        slew,
        np.zeros(x.size, dtype=np.float32),
        generation_s,
        tuple(selected_edges),
    )


def global_path_from_local(
    course: Course,
    comparison: Comparison,
    config: PlannerConfig,
    label: str = "現在4.471秒経路",
) -> GlobalPath:
    baseline = comparison.best
    # 基準経路は元から約10 mm間隔なので、再補間による曲率差を作らず
    # 4.470922秒をそのままフォールバック値として保持する。
    x = baseline.path.x_mm.astype(np.float32, copy=True)
    y = baseline.path.y_mm.astype(np.float32, copy=True)
    target = cumulative_distance_m(x, y) * 1000.0
    progress = np.arange(course.point_count, dtype=np.float32)
    source_distance = course.distance_mm.astype(np.float32, copy=True)
    yaw, curvature, slew = _geometry_arrays(x, y)
    return GlobalPath(
        label,
        x,
        y,
        target.astype(np.float32),
        progress,
        source_distance,
        np.full(x.size, -1, dtype=np.int32),
        np.zeros(x.size, dtype=np.bool_),
        yaw,
        curvature,
        slew,
        np.zeros(x.size, dtype=np.float32),
        baseline.path.generation_s,
        (),
    )


def evaluate_global_path(
    course: Course,
    path: GlobalPath,
    config: PlannerConfig,
    original_length_m: float,
    boundary: BoardBoundary | None,
    *,
    enforce_line_rule: bool = True,
    enforce_geometry: bool = True,
    detailed_interactions: bool = True,
    line_tube: _LineTubeGrid | None = None,
    enforce_board: bool = True,
) -> EvaluatedGlobalPath:
    plan = plan_speed(path.x_mm, path.y_mm, config)
    segment_m = np.diff(plan.distance_m)
    segment_time = 2.0 * segment_m / np.maximum(
        plan.speed_mps[:-1] + plan.speed_mps[1:], 1.0e-6
    )
    cumulative_time = np.concatenate((np.zeros(1), np.cumsum(segment_time)))
    source_index = _PolylineIndex(course.x_mm, course.y_mm) if detailed_interactions else None
    if source_index is None:
        interactions = LineInteractionDetails(
            0,
            0,
            0.0,
            (),
            (),
            0,
            0,
            np.zeros(path.x_mm.size, dtype=np.bool_),
        )
        crossing_path = np.zeros(path.x_mm.size, dtype=np.bool_)
    else:
        interactions = _line_interaction_details(
            path.x_mm,
            path.y_mm,
            path.source_progress_index,
            source_index,
            config,
        )
        crossing_path = interactions.crossing_path
    violations: list[str] = []
    warnings: list[str] = []
    if not (
        np.isfinite(path.x_mm).all()
        and np.isfinite(path.y_mm).all()
        and np.isfinite(plan.speed_mps).all()
    ):
        violations.append("非有限値")
    if np.any(np.diff(path.source_progress_index) < -1.0e-5):
        violations.append("進行index逆行")
    if not (
        np.allclose(path.x_mm[[0, -1]], course.x_mm[[0, -1]], atol=1.0e-4)
        and np.allclose(path.y_mm[[0, -1]], course.y_mm[[0, -1]], atol=1.0e-4)
    ):
        violations.append("端部接続")
    max_segment = float(np.max(np.hypot(np.diff(path.x_mm), np.diff(path.y_mm))))
    if max_segment > config.max_segment_mm:
        violations.append("点間距離")
    if detailed_interactions and self_intersection_count(
        path.x_mm, path.y_mm
    ) > self_intersection_count(course.x_mm, course.y_mm):
        violations.append("自己交差")
    max_slew = float(np.max(np.abs(path.curvature_slew_per_m2)))
    if enforce_geometry and max_slew > config.max_curvature_slew_per_m2:
        violations.append("曲率変化")
    if line_tube is not None:
        line_possible = line_tube.contains_path(path.x_mm, path.y_mm)
    else:
        if source_index is None:
            source_index = _PolylineIndex(course.x_mm, course.y_mm)
        line_possible = _line_coverage_is_possible(
            path.x_mm, path.y_mm, source_index, config
        )
    if enforce_line_rule:
        if not line_possible:
            violations.append("規定最大車体でも白線から完全離脱")
        else:
            warnings.append("実車外形未登録（白線重なり未保証）")
    if enforce_board:
        if boundary is None:
            warnings.append("板境界未確認")
        elif not board_path_is_inside(
            path.x_mm, path.y_mm, boundary, config.board_robot_margin_mm
        ):
            violations.append("板外")
    if interactions.shallow_crossing_count:
        warnings.append(f"浅角交差{interactions.shallow_crossing_count}回")
    if interactions.parallel_distance_mm > 1.0:
        warnings.append(
            f"他ライン平行近接{interactions.parallel_distance_mm * 0.001:.2f}m"
        )
    if interactions.crossing_count:
        warnings.append(
            f"他ライン交差{interactions.crossing_count}回"
            f"（最小角{min(interactions.crossing_angles_deg):.1f}deg、"
            f"過去{interactions.past_crossing_count}/未来{interactions.future_crossing_count}）"
        )
    radii = radius_mm(path.x_mm, path.y_mm, config.radius_window)
    inner = radii[config.radius_window : -config.radius_window]
    min_radius = float(np.min(inner)) if inner.size else float(np.min(radii))
    omega = np.abs(np.rad2deg(plan.speed_mps * path.curvature_per_m))
    skipped_source = sum(
        float(course.distance_mm[end] - course.distance_mm[start])
        for _, start, end, _ in path.selected_edges
    )
    metrics = GlobalMetrics(
        float(cumulative_time[-1]),
        float(plan.distance_m[-1]),
        (original_length_m - float(plan.distance_m[-1])) / original_length_m * 100.0,
        float(np.max(plan.speed_mps)),
        float(np.max(omega)),
        min_radius,
        max_slew,
        len(path.selected_edges),
        skipped_source * 0.001,
        interactions.crossing_count,
        interactions.shallow_crossing_count,
        (
            min(interactions.crossing_angles_deg)
            if interactions.crossing_angles_deg
            else float("nan")
        ),
        interactions.past_crossing_count,
        interactions.future_crossing_count,
        interactions.parallel_distance_mm * 0.001,
        not violations,
        "、".join(warnings),
        "、".join(violations),
    )
    completed_path = replace(
        path,
        speed_mps=plan.speed_mps.copy(),
        deliberate_line_crossing=(
            crossing_path & (path.shortcut_edge_id > 0)
        ),
    )
    return EvaluatedGlobalPath(
        completed_path,
        metrics,
        plan.distance_m,
        plan.speed_mps,
        plan.limit_reason,
        cumulative_time.astype(np.float32),
        plan.elapsed_s,
    )


def _local_finish(
    course: Course,
    evaluated: EvaluatedGlobalPath,
    config: PlannerConfig,
    original_length_m: float,
    boundary: BoardBoundary | None,
    line_tube: _LineTubeGrid | None = None,
) -> tuple[EvaluatedGlobalPath, float]:
    """各大域辺の入口・出口を独立したraised-cosine窓で微調整する。"""

    start_time = perf_counter()
    best = evaluated
    if not evaluated.path.selected_edges:
        return best, perf_counter() - start_time
    edge_id = evaluated.path.shortcut_edge_id
    changed = np.flatnonzero(np.diff(edge_id) != 0) + 1
    # 基準辺はアンカーごとに別IDを持つ。その境界は形状接続では
    # ないので、採用ショートカット（正ID）の出入り口だけを仕上げる。
    transition_indices = changed[
        (edge_id[changed - 1] > 0) | (edge_id[changed] > 0)
    ]
    for center in transition_indices:
        for half_width in (10, 20):
            left = max(1, int(center) - half_width)
            right = min(best.path.x_mm.size - 1, int(center) + half_width + 1)
            if right - left < 5:
                continue
            x = best.path.x_mm.astype(np.float64, copy=True)
            y = best.path.y_mm.astype(np.float64, copy=True)
            smooth_x = np.convolve(x, np.ones(5) / 5.0, mode="same")
            smooth_y = np.convolve(y, np.ones(5) / 5.0, mode="same")
            phase = np.linspace(-1.0, 1.0, right - left)
            weight = 0.5 + 0.5 * np.cos(pi * phase)
            x[left:right] += (smooth_x[left:right] - x[left:right]) * weight
            y[left:right] += (smooth_y[left:right] - y[left:right]) * weight
            yaw, curvature, slew = _geometry_arrays(
                x.astype(np.float32), y.astype(np.float32)
            )
            candidate_path = replace(
                best.path,
                label=(
                    best.path.label
                    if best.path.label.endswith("＋局所仕上げ")
                    else best.path.label + "＋局所仕上げ"
                ),
                x_mm=x.astype(np.float32),
                y_mm=y.astype(np.float32),
                yaw_rad=yaw,
                curvature_per_m=curvature,
                curvature_slew_per_m2=slew,
            )
            candidate = evaluate_global_path(
                course,
                candidate_path,
                config,
                original_length_m,
                boundary,
                detailed_interactions=False,
                line_tube=line_tube,
            )
            if (
                candidate.metrics.valid
                and candidate.metrics.predicted_time_s
                < best.metrics.predicted_time_s - 1.0e-9
            ):
                best = candidate
    return best, perf_counter() - start_time


def _run_mode(
    course: Course,
    comparison: Comparison,
    config: PlannerConfig,
    mode: str,
    boundary: BoardBoundary | None,
    fallback: EvaluatedGlobalPath,
) -> GlobalSearchResult:
    total_start = perf_counter()
    anchors = extract_anchor_indices(
        course,
        comparison.best.path.x_mm,
        comparison.best.path.y_mm,
        config,
        mode,
    )
    edges, geometry_s, considered = _make_edges(
        course, comparison, anchors, config, mode, boundary
    )
    baseline_plan = plan_speed(
        comparison.best.path.x_mm, comparison.best.path.y_mm, config
    )
    segment_time = 2.0 * np.diff(baseline_plan.distance_m) / np.maximum(
        baseline_plan.speed_mps[:-1] + baseline_plan.speed_mps[1:], 1.0e-6
    )
    baseline_time = np.concatenate((np.zeros(1), np.cumsum(segment_time)))
    base_edges = [
        _base_edge(
            node,
            anchors,
            comparison.best.path.x_mm,
            comparison.best.path.y_mm,
            baseline_plan.distance_m,
            baseline_time,
        )
        for node in range(anchors.size - 1)
    ]
    top_k = config.reference_top_k if mode == "reference" else config.embedded_top_k
    sequences, graph_s = _k_best_edge_paths(
        anchors, edges, base_edges, top_k, reference_only=True
    )
    edge_map = {edge.edge_id: edge for edge in base_edges + edges}
    evaluation_start = perf_counter()
    fast_candidates: list[EvaluatedGlobalPath] = []
    evaluated_count = 0
    original_length = comparison.original.metrics.length_m
    line_tube = _LineTubeGrid(
        course.x_mm,
        course.y_mm,
        config.rule_max_robot_radius_mm + config.rule_line_half_width_mm,
    )
    for sequence in sequences:
        path = _build_global_path(
            course,
            sequence,
            edge_map,
            config,
            f"{mode}大域経路",
            perf_counter() - total_start,
        )
        evaluated = evaluate_global_path(
            course,
            path,
            config,
            original_length,
            boundary,
            detailed_interactions=False,
            line_tube=line_tube,
        )
        evaluated_count += 1
        if evaluated.metrics.valid:
            fast_candidates.append(evaluated)
    fast_candidates.sort(key=lambda item: item.metrics.predicted_time_s)
    best = fallback
    detailed_limit = 8 if mode == "reference" else 2
    for candidate in fast_candidates[:detailed_limit]:
        if candidate.metrics.predicted_time_s >= best.metrics.predicted_time_s:
            break
        detailed = evaluate_global_path(
            course,
            candidate.path,
            config,
            original_length,
            boundary,
        )
        if detailed.metrics.valid:
            best = detailed
            break
    evaluation_s = perf_counter() - evaluation_start
    finished, finish_s = _local_finish(
        course, best, config, original_length, boundary, line_tube
    )
    if finished.metrics.predicted_time_s < best.metrics.predicted_time_s:
        exact_finished = evaluate_global_path(
            course, finished.path, config, original_length, boundary
        )
        if exact_finished.metrics.valid:
            best = exact_finished
    fallback_used = best.path.selected_edges == ()
    max_points = max((edge.x_mm.size for edge in edges), default=0)
    memory = int(
        anchors.nbytes
        + sum(edge.x_mm.nbytes + edge.y_mm.nbytes + edge.source_progress_index.nbytes for edge in edges)
        + max_points * 8 * 6
    )
    stats = GlobalSearchStats(
        mode,
        int(anchors.size),
        len(edges),
        sum(1 for edge in edges if edge.reference_valid),
        geometry_s,
        graph_s,
        evaluated_count,
        evaluation_s,
        finish_s,
        perf_counter() - total_start,
        considered * 4 + evaluated_count * (config.speed_scan_iterations * 2 + 8),
        memory,
    )
    return GlobalSearchResult(mode, anchors, best, best, stats, fallback_used)


def _run_geometric_lower_bound(
    course: Course,
    comparison: Comparison,
    config: PlannerConfig,
    boundary: BoardBoundary | None,
    fallback: EvaluatedGlobalPath,
) -> EvaluatedGlobalPath:
    anchors = extract_anchor_indices(
        course,
        comparison.best.path.x_mm,
        comparison.best.path.y_mm,
        config,
        "reference",
    )
    edges, _, _ = _make_edges(
        course, comparison, anchors, config, "reference", boundary
    )
    baseline_plan = plan_speed(
        comparison.best.path.x_mm, comparison.best.path.y_mm, config
    )
    baseline_segment_time = 2.0 * np.diff(baseline_plan.distance_m) / np.maximum(
        baseline_plan.speed_mps[:-1] + baseline_plan.speed_mps[1:], 1.0e-6
    )
    baseline_time = np.concatenate((np.zeros(1), np.cumsum(baseline_segment_time)))
    base_edges = [
        _base_edge(
            node,
            anchors,
            comparison.best.path.x_mm,
            comparison.best.path.y_mm,
            baseline_plan.distance_m,
            baseline_time,
        )
        for node in range(anchors.size - 1)
    ]
    sequences, _ = _k_best_edge_paths(
        anchors,
        edges,
        base_edges,
        min(config.reference_top_k, 32),
        reference_only=False,
    )
    edge_map = {edge.edge_id: edge for edge in base_edges + edges}
    best = fallback
    for sequence in sequences:
        path = _build_global_path(
            course, sequence, edge_map, config, "幾何下限", 0.0
        )
        evaluated = evaluate_global_path(
            course,
            path,
            config,
            comparison.original.metrics.length_m,
            boundary,
            enforce_line_rule=False,
            enforce_geometry=False,
            detailed_interactions=False,
        )
        if (
            evaluated.metrics.valid
            and evaluated.metrics.predicted_time_s < best.metrics.predicted_time_s
        ):
            best = evaluated
    if best is fallback:
        return best
    return evaluate_global_path(
        course,
        best.path,
        config,
        comparison.original.metrics.length_m,
        boundary,
        enforce_line_rule=False,
        enforce_geometry=False,
    )


def run_global_comparison(
    course: Course,
    config: PlannerConfig | None = None,
    *,
    local_comparison: Comparison | None = None,
) -> GlobalComparison:
    """LN5実車外形をハードゲートにした比較へ委譲する。"""

    # 循環importを避けるため、旧大域関数の定義完了後に読む。
    from .legal_planner import run_legal_comparison

    return run_legal_comparison(
        course,
        config,
        local_comparison=local_comparison,
    )


def run_global_mode(
    course: Course,
    mode: str,
    config: PlannerConfig | None = None,
    *,
    local_comparison: Comparison | None = None,
) -> tuple[Comparison, GlobalSearchResult, str]:
    """全コース回帰用にreferenceまたはembedded-liteだけを実行する。"""

    if mode not in {"reference", "embedded-lite"}:
        raise ValueError(f"未対応モードです: {mode}")
    config = config or PlannerConfig()
    comparison = local_comparison or run_comparison(course, config)
    boundary = load_board_boundary(course)
    fallback_path = global_path_from_local(course, comparison, config)
    fallback = evaluate_global_path(
        course,
        fallback_path,
        config,
        comparison.original.metrics.length_m,
        boundary,
        detailed_interactions=False,
        line_tube=_LineTubeGrid(
            course.x_mm,
            course.y_mm,
            config.rule_max_robot_radius_mm + config.rule_line_half_width_mm,
        ),
    )
    result = _run_mode(
        course, comparison, config, mode, boundary, fallback
    )
    board_status = (
        f"CAD板境界確認済み（{boundary.source}）"
        if boundary is not None and boundary.confirmed
        else "板境界未確認"
    )
    return comparison, result, board_status
