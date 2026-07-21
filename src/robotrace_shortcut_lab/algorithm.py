from __future__ import annotations

import numpy as np

from .geometry import (
    cumulative_distance_mm,
    curvature_slew_per_m2,
    quintic_hermite,
    radius_mm,
)
from .models import CoursePath, Path, Settings


def generate_paths(course: CoursePath, settings: Settings) -> tuple[Path, Path, Path]:
    """現行近似、鋭角接続比較、曲率連続の改善経路をまとめて生成する。"""

    baseline = _fast_firmware_style(course, settings)
    corridors = _straight_corridors(course, settings)
    sharp = _connect_straights(course, baseline, corridors, settings, smooth=False)
    smooth = _connect_straights(course, baseline, corridors, settings, smooth=True)
    # 曲率変化ピークを固定窓で正則化し、直線と円弧の接続を滑らかにする。
    smooth = _regularize_hotspots(course, smooth, settings)
    return baseline, sharp, smooth


def _move_hairpin_forward(course: CoursePath, path: Path, settings: Settings) -> Path:
    """最大ピークの折返しを手前へ寄せ、3階差分最小の涙滴形にする。"""

    before_slew = curvature_slew_per_m2(path.x_mm, path.y_mm)
    peak = 100 + int(np.argmax(np.abs(before_slew[100:-100])))
    # Python比較経路は直線化後に適用するため、C後処理とは別に調整する。
    start, end, center = peak - 100, peak + 60, peak - 30
    if start < 0 or end >= course.point_count:
        return path

    tangent = np.array(
        [
            path.x_mm[center + 1] - path.x_mm[center - 1],
            path.y_mm[center + 1] - path.y_mm[center - 1],
        ],
        dtype=np.float64,
    )
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 0.001:
        return path
    forward_shift = -tangent / tangent_norm

    count = end - start + 1
    index = np.arange(start, end + 1)
    normalized = (index - center) / 70.0
    envelope = np.where(np.abs(normalized) < 1.0, (1.0 - normalized**2) ** 3, 0.0)
    target_x = path.x_mm[start : end + 1] + envelope * 84.0 * forward_shift[0]
    target_y = path.y_mm[start : end + 1] + envelope * 84.0 * forward_shift[1]

    weights = np.ones(count)
    weights[:12] = weights[-12:] = 100_000.0
    factor = _third_difference_factor(weights, 10_000.0)
    trial_x, trial_y = path.x_mm.copy(), path.y_mm.copy()
    trial_x[start : end + 1] = _solve_spd_banded(factor, weights * target_x)
    trial_y[start : end + 1] = _solve_spd_banded(factor, weights * target_y)

    offset = np.hypot(trial_x - course.x_mm, trial_y - course.y_mm)
    trial_radius = radius_mm(trial_x, trial_y, settings.radius_window_segments)
    finite_radius = trial_radius[(trial_radius > 0.0) & (trial_radius < 900_000.0)]
    after_slew = curvature_slew_per_m2(trial_x, trial_y)
    if float(np.max(offset)) > settings.offset_limit_mm + 0.001:
        return path
    if finite_radius.size and float(np.min(finite_radius)) < settings.min_radius_mm:
        return path
    if float(np.max(np.abs(after_slew[start:end]))) >= float(
        np.max(np.abs(before_slew[start:end]))
    ):
        return path
    return Path("直線＋曲率連続＋涙滴形折返し", trial_x, trial_y, path.straight_cores)


def _regularize_hotspots(course: CoursePath, path: Path, settings: Settings) -> Path:
    """残った曲率変化ピークだけを固定窓で平滑化する。"""

    weights = np.ones(161)
    weights[:12] = weights[-12:] = 100_000.0
    factor = _third_difference_factor(weights, 10_000.0)
    current = path
    for _pass in range(12):
        before_slew = curvature_slew_per_m2(current.x_mm, current.y_mm)
        peak = 100 + int(np.argmax(np.abs(before_slew[100:-100])))
        if float(abs(before_slew[peak])) < 150.0:
            break
        start, end = peak - 80, peak + 80
        if start < 0 or end >= course.point_count:
            break
        trial_x, trial_y = current.x_mm.copy(), current.y_mm.copy()
        trial_x[start : end + 1] = _solve_spd_banded(
            factor, weights * current.x_mm[start : end + 1]
        )
        trial_y[start : end + 1] = _solve_spd_banded(
            factor, weights * current.y_mm[start : end + 1]
        )
        offset = np.hypot(trial_x - course.x_mm, trial_y - course.y_mm)
        trial_radius = radius_mm(trial_x, trial_y, settings.radius_window_segments)
        finite_radius = trial_radius[(trial_radius > 0.0) & (trial_radius < 900_000.0)]
        after_slew = curvature_slew_per_m2(trial_x, trial_y)
        if float(np.max(offset)) > settings.offset_limit_mm + 0.001:
            improved = False
        elif finite_radius.size and float(np.min(finite_radius)) < settings.min_radius_mm:
            improved = False
        else:
            improved = float(np.max(np.abs(after_slew[100:-100]))) < float(
                np.max(np.abs(before_slew[100:-100]))
            )
        if float(np.max(offset)) <= settings.offset_limit_mm + 0.001 and improved:
            current = Path(
                "直線優先＋dκ/ds正則化", trial_x, trial_y, current.straight_cores
            )
            continue

        # タイトな折返しは座標平滑化だけでは点間隔が詰まるため、元ライン側へ少し開く。
        peak_value = float(abs(before_slew[peak]))
        half_window = 40 if peak_value > 300.0 else 160
        blend = 0.20 if peak_value > 300.0 else 0.05
        start, end = peak - half_window, peak + half_window
        if start < 0 or end >= course.point_count:
            break
        u = np.linspace(-1.0, 1.0, end - start + 1)
        envelope = (1.0 - u * u) ** 3
        trial_x, trial_y = current.x_mm.copy(), current.y_mm.copy()
        trial_x[start : end + 1] += (
            course.x_mm[start : end + 1] - trial_x[start : end + 1]
        ) * envelope * blend
        trial_y[start : end + 1] += (
            course.y_mm[start : end + 1] - trial_y[start : end + 1]
        ) * envelope * blend
        candidate = Path(current.label, trial_x, trial_y, current.straight_cores)
        candidate_slew = curvature_slew_per_m2(candidate.x_mm, candidate.y_mm)
        candidate_radius = radius_mm(
            candidate.x_mm, candidate.y_mm, settings.radius_window_segments
        )
        finite_radius = candidate_radius[
            (candidate_radius > 0.0) & (candidate_radius < 900_000.0)
        ]
        candidate_offset = np.hypot(
            candidate.x_mm - course.x_mm, candidate.y_mm - course.y_mm
        )
        if float(np.max(candidate_offset)) > settings.offset_limit_mm + 0.001:
            break
        if finite_radius.size and float(np.min(finite_radius)) < settings.min_radius_mm:
            break
        if float(np.max(np.abs(candidate_slew[100:-100]))) >= peak_value:
            break
        current = candidate
    return current


def _third_difference_factor(weights: np.ndarray, regularization: float) -> np.ndarray:
    """D3転置D3の4本の下三角帯からCholesky因子を作る。"""

    count = weights.size
    matrix = np.zeros((4, count), dtype=np.float64)
    main = np.full(count, 20.0)
    main[:3], main[-3:] = (1.0, 10.0, 19.0), (19.0, 10.0, 1.0)
    matrix[0] = weights + regularization * main
    matrix[1, 1:] = regularization * np.r_[-3.0, -12.0, np.full(count - 5, -15.0), -12.0, -3.0]
    matrix[2, 2:] = regularization * np.r_[3.0, np.full(count - 4, 6.0), 3.0]
    matrix[3, 3:] = -regularization
    factor = np.zeros_like(matrix)
    for row in range(count):
        for distance in range(min(3, row), -1, -1):
            column = row - distance
            product = 0.0
            for index in range(max(0, column - 3, row - 3), column):
                product += factor[row - index, row] * factor[column - index, column]
            if distance == 0:
                factor[0, row] = np.sqrt(matrix[0, row] - product)
            else:
                factor[distance, row] = (matrix[distance, row] - product) / factor[0, column]
    return factor


def _solve_spd_banded(factor: np.ndarray, right: np.ndarray) -> np.ndarray:
    """帯幅3のCholesky因子を使い、追加配列2本で解く。"""

    count = right.size
    intermediate = np.zeros(count)
    for row in range(count):
        product = sum(
            factor[row - column, row] * intermediate[column]
            for column in range(max(0, row - 3), row)
        )
        intermediate[row] = (right[row] - product) / factor[0, row]
    result = np.zeros(count)
    for row in range(count - 1, -1, -1):
        product = sum(
            factor[column - row, column] * result[column]
            for column in range(row + 1, min(count, row + 4))
        )
        result[row] = (intermediate[row] - product) / factor[0, row]
    return result


def _fast_firmware_style(course: CoursePath, settings: Settings) -> Path:
    """現行Elastic Bandの形をNumPyの同時更新で高速近似する。"""

    source_x, source_y = course.x_mm, course.y_mm
    x, y = source_x.copy(), source_y.copy()
    edge = np.minimum(np.arange(course.point_count), np.arange(course.point_count)[::-1])
    source_weight = np.full(course.point_count, settings.source_weight, dtype=np.float32)
    source_weight[edge <= 30] = np.float32(0.05)
    blend = (edge > 30) & (edge < 60)
    source_weight[blend] = np.float32(0.05) + (
        np.float32(settings.source_weight - 0.05) * (edge[blend] - 30) / 30.0
    )
    relax_kernel = np.ones(41, dtype=np.float32)

    for iteration in range(settings.elastic_iterations):
        previous_x, previous_y = x.copy(), y.copy()
        x[1:-1] = (
            previous_x[1:-1]
            + (source_x[1:-1] - previous_x[1:-1]) * source_weight[1:-1]
            + (previous_x[:-2] + previous_x[2:] - 2.0 * previous_x[1:-1])
            * settings.smooth_weight
        )
        y[1:-1] = (
            previous_y[1:-1]
            + (source_y[1:-1] - previous_y[1:-1]) * source_weight[1:-1]
            + (previous_y[:-2] + previous_y[2:] - 2.0 * previous_y[1:-1])
            * settings.smooth_weight
        )
        if (iteration + 1) % settings.radius_relax_every == 0:
            radius = radius_mm(x, y, settings.radius_window_segments)
            low = ((radius > 0.0) & (radius < settings.radius_relax_target_mm)).astype(
                np.float32
            )
            affected = np.convolve(low, relax_kernel, mode="same") > 0.0
            x[affected] += (source_x[affected] - x[affected]) * settings.radius_relax_blend
            y[affected] += (source_y[affected] - y[affected]) * settings.radius_relax_blend
        offset_x, offset_y = x - source_x, y - source_y
        offset_squared = offset_x * offset_x + offset_y * offset_y
        near_limit = offset_squared > (settings.offset_limit_mm - 5.0) ** 2
        x[near_limit] += (source_x[near_limit] - x[near_limit]) * 0.02
        y[near_limit] += (source_y[near_limit] - y[near_limit]) * 0.02
        offset_x, offset_y = x - source_x, y - source_y
        offset_squared = offset_x * offset_x + offset_y * offset_y
        outside = offset_squared > settings.offset_limit_mm**2
        if np.any(outside):
            scale = settings.offset_limit_mm / np.sqrt(offset_squared[outside])
            x[outside] = source_x[outside] + offset_x[outside] * scale
            y[outside] = source_y[outside] + offset_y[outside] * scale
    offset_x, offset_y = x - source_x, y - source_y
    offset = np.hypot(offset_x, offset_y)
    outside = offset > settings.offset_limit_mm
    x[outside] = source_x[outside] + offset_x[outside] * (
        settings.offset_limit_mm / offset[outside]
    )
    y[outside] = source_y[outside] + offset_y[outside] * (
        settings.offset_limit_mm / offset[outside]
    )
    return Path("現行ファーム高速近似", x, y)


def _connect_straights(
    course: CoursePath,
    baseline: Path,
    corridors: tuple[tuple[int, int], ...],
    settings: Settings,
    *,
    smooth: bool,
) -> Path:
    x, y = baseline.x_mm.copy(), baseline.y_mm.copy()
    dx, dy = np.gradient(baseline.x_mm), np.gradient(baseline.y_mm)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    accepted: list[tuple[int, int]] = []
    transition = settings.transition_segments

    for start, end in corridors:
        core_start, core_end = start + transition, end - transition
        if core_start >= core_end:
            continue
        chord_dx = float(baseline.x_mm[core_end] - baseline.x_mm[core_start])
        chord_dy = float(baseline.y_mm[core_end] - baseline.y_mm[core_start])
        chord_count = core_end - core_start
        chord_vx, chord_vy = chord_dx / chord_count, chord_dy / chord_count
        ratio = np.arange(chord_count + 1, dtype=np.float64) / chord_count
        chord_x = baseline.x_mm[core_start] + ratio * chord_dx
        chord_y = baseline.y_mm[core_start] + ratio * chord_dy

        trial_x, trial_y = x.copy(), y.copy()
        trial_x[core_start : core_end + 1] = chord_x
        trial_y[core_start : core_end + 1] = chord_y
        if smooth:
            u = np.linspace(0.0, 1.0, transition + 1)
            trial_x[start : core_start + 1] = quintic_hermite(
                float(baseline.x_mm[start]),
                float(baseline.x_mm[core_start]),
                float(dx[start] * transition),
                chord_vx * transition,
                float(ddx[start] * transition**2),
                0.0,
                u,
            )
            trial_y[start : core_start + 1] = quintic_hermite(
                float(baseline.y_mm[start]),
                float(baseline.y_mm[core_start]),
                float(dy[start] * transition),
                chord_vy * transition,
                float(ddy[start] * transition**2),
                0.0,
                u,
            )
            trial_x[core_end : end + 1] = quintic_hermite(
                float(baseline.x_mm[core_end]),
                float(baseline.x_mm[end]),
                chord_vx * transition,
                float(dx[end] * transition),
                0.0,
                float(ddx[end] * transition**2),
                u,
            )
            trial_y[core_end : end + 1] = quintic_hermite(
                float(baseline.y_mm[core_end]),
                float(baseline.y_mm[end]),
                chord_vy * transition,
                float(dy[end] * transition),
                0.0,
                float(ddy[end] * transition**2),
                u,
            )

        offset = np.hypot(trial_x - course.x_mm, trial_y - course.y_mm)
        trial_radius = radius_mm(trial_x, trial_y, settings.radius_window_segments)
        finite_radius = trial_radius[(trial_radius > 0.0) & (trial_radius < 900_000.0)]
        if float(np.max(offset)) > settings.offset_limit_mm + 0.001:
            continue
        if finite_radius.size and float(np.min(finite_radius)) < settings.min_radius_mm:
            continue
        x, y = trial_x.astype(np.float32), trial_y.astype(np.float32)
        accepted.append((core_start, core_end))

    label = "直線優先＋鋭角接続" if not smooth else "直線優先＋曲率連続接続"
    return Path(label, x, y, tuple(accepted))


def _straight_corridors(course: CoursePath, settings: Settings) -> tuple[tuple[int, int], ...]:
    anchors = _rdp_anchors(course.x_mm, course.y_mm, settings)
    cumulative = cumulative_distance_mm(course.x_mm, course.y_mm)
    candidates: list[tuple[float, int, int]] = []
    for position, start in enumerate(anchors[:-1]):
        farthest: int | None = None
        for end in anchors[position + 1 :]:
            arc = float(cumulative[end] - cumulative[start])
            if not _is_straight(course, cumulative, start, end, settings):
                if arc >= settings.straight_min_length_mm:
                    break
                continue
            if arc >= settings.straight_min_length_mm:
                farthest = end
        if farthest is not None:
            candidates.append((float(cumulative[farthest] - cumulative[start]), start, farthest))
    selected: list[tuple[int, int]] = []
    for _length, start, end in sorted(candidates, reverse=True):
        if not any(start < other_end and other_start < end for other_start, other_end in selected):
            selected.append((start, end))
    return tuple(sorted(selected))


def _rdp_anchors(x: np.ndarray, y: np.ndarray, settings: Settings) -> list[int]:
    keep = np.zeros(x.size, dtype=np.bool_)
    keep[[0, -1]] = True
    stack = [(0, x.size - 1)]
    count = 2
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        dx, dy = float(x[end] - x[start]), float(y[end] - y[start])
        chord = float(np.hypot(dx, dy))
        relative_x = x[start + 1 : end] - x[start]
        relative_y = y[start + 1 : end] - y[start]
        deviation = (
            np.hypot(relative_x, relative_y)
            if chord <= 0.001
            else np.abs(dy * relative_x - dx * relative_y) / chord
        )
        local = int(np.argmax(deviation))
        if float(deviation[local]) <= settings.rdp_tolerance_mm:
            continue
        if count >= settings.max_anchor_count:
            raise ValueError("RDPアンカー上限を超えました")
        split = start + 1 + local
        keep[split] = True
        count += 1
        stack.extend(((split, end), (start, split)))
    return np.flatnonzero(keep).astype(int).tolist()


def _is_straight(
    course: CoursePath,
    cumulative: np.ndarray,
    start: int,
    end: int,
    settings: Settings,
) -> bool:
    dx = float(course.x_mm[end] - course.x_mm[start])
    dy = float(course.y_mm[end] - course.y_mm[start])
    chord = float(np.hypot(dx, dy))
    arc = float(cumulative[end] - cumulative[start])
    if chord <= 0.001 or chord / arc < settings.straight_min_chord_ratio:
        return False
    relative_x = course.x_mm[start : end + 1] - course.x_mm[start]
    relative_y = course.y_mm[start : end + 1] - course.y_mm[start]
    deviation = np.abs(dy * relative_x - dx * relative_y) / chord
    progress = (relative_x * dx + relative_y * dy) / chord
    return bool(
        float(np.max(deviation)) <= settings.straight_max_deviation_mm
        and float(np.min(np.diff(progress))) >= -2.0
    )
