from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from .geometry import (
    cumulative_distance_m,
    curvature_slew_per_m2,
    expanded_mask,
    frenet_normals,
    path_order_is_forward,
    path_length_m,
    radius_mm,
    self_intersection_count,
    signed_curvature_per_m,
)
from .model import (
    CANDIDATES,
    CandidateWeights,
    Comparison,
    Course,
    EvaluatedPath,
    GeneratedPath,
    Metrics,
    PlannerConfig,
    SpeedPlan,
)


LIMIT_MAX_SPEED = np.uint8(0)
LIMIT_GFCP = np.uint8(1)
LIMIT_AALP = np.uint8(2)
LIMIT_ACCELERATION = np.uint8(3)
LIMIT_DECELERATION = np.uint8(4)


def _edge_activity(count: int, config: PlannerConfig) -> np.ndarray:
    """端部を0、通常区間を1とする固定長の遷移係数。"""

    index = np.arange(count)
    edge_distance = np.minimum(index, count - 1 - index)
    activity = np.ones(count, dtype=np.float32)
    activity[edge_distance <= config.edge_keep_points] = 0.0
    blend = (edge_distance > config.edge_keep_points) & (
        edge_distance < config.edge_keep_points + config.edge_blend_points
    )
    ratio = (
        edge_distance[blend] - config.edge_keep_points
    ) / float(config.edge_blend_points)
    activity[blend] = (0.5 - 0.5 * np.cos(np.pi * ratio)).astype(np.float32)
    return activity


def _repair_min_radius(
    source_x: np.ndarray,
    source_y: np.ndarray,
    normal_x: np.ndarray,
    normal_y: np.ndarray,
    offset: np.ndarray,
    config: PlannerConfig,
    passes: int = 80,
) -> np.ndarray:
    """半径不足区間の横オフセットだけを元ライン側へ戻す。"""

    repaired = offset.astype(np.float32, copy=True)
    for _ in range(passes):
        x = source_x + normal_x * repaired
        y = source_y + normal_y * repaired
        radii = radius_mm(x, y, config.radius_window)
        bad = radii < np.float32(config.legacy_min_radius_mm)
        if not bool(np.any(bad)):
            break
        affected = expanded_mask(bad, config.radius_window)
        repaired[affected] *= np.float32(0.88)
    return repaired


def generate_original(course: Course) -> GeneratedPath:
    zeros = np.zeros(course.point_count, dtype=np.float32)
    return GeneratedPath(
        "原コース", course.x_mm.copy(), course.y_mm.copy(), zeros, 0.0
    )


def generate_elastic_band(course: Course, config: PlannerConfig) -> GeneratedPath:
    """現行ファームの力と制約をJacobi更新で再現した比較基準。"""

    start = perf_counter()
    source_x = course.x_mm.astype(np.float32, copy=True)
    source_y = course.y_mm.astype(np.float32, copy=True)
    x, y = source_x.copy(), source_y.copy()
    count = course.point_count
    normal_x, normal_y = frenet_normals(source_x, source_y)
    edge_distance = np.minimum(np.arange(count), count - 1 - np.arange(count))
    source_weight = np.full(count, 0.0005, dtype=np.float32)
    source_weight[edge_distance <= 30] = 0.05
    blend = (edge_distance > 30) & (edge_distance < 60)
    ratio = (edge_distance[blend] - 30) / 30.0
    source_weight[blend] = (0.05 + (0.0005 - 0.05) * ratio).astype(np.float32)

    for iteration in range(420):
        old_x, old_y = x.copy(), y.copy()
        x[1:-1] = old_x[1:-1] + (
            (source_x[1:-1] - old_x[1:-1]) * source_weight[1:-1]
            + (old_x[:-2] + old_x[2:] - 2.0 * old_x[1:-1]) * 0.36
        )
        y[1:-1] = old_y[1:-1] + (
            (source_y[1:-1] - old_y[1:-1]) * source_weight[1:-1]
            + (old_y[:-2] + old_y[2:] - 2.0 * old_y[1:-1]) * 0.36
        )

        displacement_x = x - source_x
        displacement_y = y - source_y
        magnitude = np.hypot(displacement_x, displacement_y)
        scale = np.minimum(1.0, config.offset_limit_mm / np.maximum(magnitude, 1.0e-6))
        x = source_x + displacement_x * scale
        y = source_y + displacement_y * scale

        if iteration % 7 == 6:
            bad = radius_mm(x, y, config.radius_window) < config.legacy_min_radius_mm
            if bool(np.any(bad)):
                affected = expanded_mask(bad, config.radius_window)
                x[affected] += (source_x[affected] - x[affected]) * 0.08
                y[affected] += (source_y[affected] - y[affected]) * 0.08

        x[0], y[0], x[-1], y[-1] = source_x[0], source_y[0], source_x[-1], source_y[-1]

    # 最終制約処理も同一indexの原ライン方向へ戻すだけにする。
    for _ in range(80):
        bad = radius_mm(x, y, config.radius_window) < config.legacy_min_radius_mm
        if not bool(np.any(bad)):
            break
        affected = expanded_mask(bad, config.radius_window)
        x[affected] += (source_x[affected] - x[affected]) * 0.15
        y[affected] += (source_y[affected] - y[affected]) * 0.15

    displacement_x = x - source_x
    displacement_y = y - source_y
    signed_offset = displacement_x * normal_x + displacement_y * normal_y
    elapsed = perf_counter() - start
    return GeneratedPath(
        "現行Elastic Band相当",
        x.astype(np.float32),
        y.astype(np.float32),
        signed_offset.astype(np.float32),
        elapsed,
        frenet_locked=False,
    )


def generate_time_candidate(
    course: Course,
    config: PlannerConfig,
    weights: CandidateWeights,
) -> GeneratedPath:
    """原ライン法線方向の横オフセットだけを固定幅局所更新する。"""

    start = perf_counter()
    source_x = course.x_mm.astype(np.float32, copy=False)
    source_y = course.y_mm.astype(np.float32, copy=False)
    normal_x, normal_y = frenet_normals(source_x, source_y)
    activity = _edge_activity(course.point_count, config)
    offset = np.zeros(course.point_count, dtype=np.float32)

    for iteration in range(weights.iterations):
        x = source_x + normal_x * offset
        y = source_y + normal_y * offset
        update = np.zeros_like(offset)

        # 経路長勾配: 前後点の弦中点へ寄せる変位を法線へ射影する。
        target_x = 0.5 * (x[:-2] + x[2:])
        target_y = 0.5 * (y[:-2] + y[2:])
        length_delta = (target_x - x[1:-1]) * normal_x[1:-1] + (
            target_y - y[1:-1]
        ) * normal_y[1:-1]
        update[1:-1] += np.float32(weights.length_weight) * length_delta

        # 4階差分を0へ寄せ、一定曲率を残したまま曲率の凹凸をならす。
        curve_target_x = (
            4.0 * (x[1:-3] + x[3:-1]) - (x[:-4] + x[4:])
        ) / 6.0
        curve_target_y = (
            4.0 * (y[1:-3] + y[3:-1]) - (y[:-4] + y[4:])
        ) / 6.0
        curve_delta = (curve_target_x - x[2:-2]) * normal_x[2:-2] + (
            curve_target_y - y[2:-2]
        ) * normal_y[2:-2]
        update[2:-2] += np.float32(weights.curvature_weight) * curve_delta

        # 6階差分を0へ寄せ、曲率変化dκ/dsの局所ピークを抑える。
        slew_target_x = (
            x[:-6]
            - 6.0 * x[1:-5]
            + 15.0 * x[2:-4]
            + 15.0 * x[4:-2]
            - 6.0 * x[5:-1]
            + x[6:]
        ) / 20.0
        slew_target_y = (
            y[:-6]
            - 6.0 * y[1:-5]
            + 15.0 * y[2:-4]
            + 15.0 * y[4:-2]
            - 6.0 * y[5:-1]
            + y[6:]
        ) / 20.0
        slew_delta = (slew_target_x - x[3:-3]) * normal_x[3:-3] + (
            slew_target_y - y[3:-3]
        ) * normal_y[3:-3]
        update[3:-3] += np.float32(weights.slew_weight) * slew_delta

        update -= np.float32(weights.source_weight) * offset
        update = np.clip(update, -weights.step_limit_mm, weights.step_limit_mm)
        offset += activity * update
        offset = np.clip(offset, -config.offset_limit_mm, config.offset_limit_mm)

        if iteration % 12 == 11:
            x = source_x + normal_x * offset
            y = source_y + normal_y * offset
            bad = radius_mm(x, y, config.radius_window) < config.legacy_min_radius_mm
            if bool(np.any(bad)):
                offset[expanded_mask(bad, config.radius_window)] *= np.float32(0.94)

    offset = _repair_min_radius(
        source_x, source_y, normal_x, normal_y, offset, config
    )
    offset[: config.edge_keep_points + 1] = 0.0
    offset[-config.edge_keep_points - 1 :] = 0.0
    x = (source_x + normal_x * offset).astype(np.float32)
    y = (source_y + normal_y * offset).astype(np.float32)
    elapsed = perf_counter() - start
    return GeneratedPath(
        f"時間選択型 #{weights.candidate_id}",
        x,
        y,
        offset.astype(np.float32),
        elapsed,
        weights.candidate_id,
        weights.name,
    )


def _project_elastic_offset(
    course: Course, elastic: GeneratedPath
) -> np.ndarray:
    normal_x, normal_y = frenet_normals(course.x_mm, course.y_mm)
    return (
        (elastic.x_mm - course.x_mm) * normal_x
        + (elastic.y_mm - course.y_mm) * normal_y
    ).astype(np.float32)


def _detect_long_window_centers(course: Course, config: PlannerConfig) -> tuple[int, ...]:
    """高曲率・低速・符号反転が密集するヘアピングループを自動検出する。"""

    curvature = signed_curvature_per_m(
        course.x_mm, course.y_mm, config.radius_window
    )
    magnitude = np.abs(curvature)
    maximum = float(np.max(magnitude))
    if maximum <= 1.0e-6:
        return ()
    radius_ratio = np.divide(
        1.0,
        0.1 * magnitude,
        out=np.full_like(magnitude, np.inf),
        where=magnitude > 1.0e-9,
    )
    gfcp_speed = np.clip(
        config.gfcp_reference_speed_mps
        * np.power(radius_ratio, config.gfcp_exponent),
        config.min_speed_mps,
        config.max_speed_mps,
    )

    local_half_width = config.radius_window
    raw_peaks: list[int] = []
    for index in range(
        config.edge_keep_points + local_half_width,
        course.point_count - config.edge_keep_points - local_half_width,
    ):
        if magnitude[index] < 0.85 * maximum:
            continue
        if gfcp_speed[index] > config.min_speed_mps + 1.0e-4:
            continue
        if magnitude[index] < np.max(
            magnitude[
                index - local_half_width : index + local_half_width + 1
            ]
        ):
            continue
        if raw_peaks and index - raw_peaks[-1] < local_half_width * 2:
            if magnitude[index] > magnitude[raw_peaks[-1]]:
                raw_peaks[-1] = index
            continue
        raw_peaks.append(index)

    if len(raw_peaks) < 3:
        ordered = np.argsort(magnitude)[::-1]
        raw_peaks = []
        for value in ordered:
            index = int(value)
            if all(abs(index - previous) >= local_half_width * 2 for previous in raw_peaks):
                raw_peaks.append(index)
            if len(raw_peaks) == 3:
                break
        raw_peaks.sort()

    best: tuple[int, ...] = ()
    best_score = -1.0
    for start in range(len(raw_peaks)):
        group: list[int] = []
        for index in raw_peaks[start:]:
            if course.distance_mm[index] - course.distance_mm[raw_peaks[start]] > 2500.0:
                break
            group.append(index)
        if len(group) < 3:
            continue
        alternating = sum(
            1
            for left, right in zip(group[:-1], group[1:], strict=True)
            if curvature[left] * curvature[right] < 0.0
        )
        score = (
            len(group) * 10.0
            + alternating * 5.0
            + sum(float(magnitude[index] / maximum) for index in group)
        )
        if score > best_score:
            best = tuple(group[:6])
            best_score = score

    if best:
        return best
    return tuple(raw_peaks[:3])


def _apply_long_windows(
    course: Course,
    config: PlannerConfig,
    base_offset: np.ndarray,
    centers: tuple[int, ...],
    curvature: np.ndarray,
    width_mm: float,
    amplitude_mm: float,
) -> np.ndarray:
    """各中心へ端点0のraised cosineを同一index法線オフセットとして加える。"""

    offset = base_offset.astype(np.float32, copy=True)
    typical_segment_mm = float(np.median(np.diff(course.distance_mm)))
    half_points = max(2, int(round(0.5 * width_mm / typical_segment_mm)))
    shape_index = np.arange(-half_points, half_points + 1, dtype=np.float32)
    shape = (
        0.5 + 0.5 * np.cos(np.pi * shape_index / float(half_points))
    ).astype(np.float32)
    for center in centers:
        start = center - half_points
        finish = center + half_points + 1
        if start < 0 or finish > course.point_count:
            continue
        direction = np.float32(1.0 if curvature[center] >= 0.0 else -1.0)
        offset[start:finish] += np.float32(amplitude_mm) * direction * shape
    offset = np.clip(
        offset, -config.offset_limit_mm, config.offset_limit_mm
    ).astype(np.float32)
    offset[: config.edge_keep_points + 1] = 0.0
    offset[-config.edge_keep_points - 1 :] = 0.0
    return offset


def _path_from_offset(
    course: Course,
    offset: np.ndarray,
    label: str,
    generation_s: float = 0.0,
    candidate_id: int | None = None,
    candidate_name: str | None = None,
) -> GeneratedPath:
    normal_x, normal_y = frenet_normals(course.x_mm, course.y_mm)
    return GeneratedPath(
        label,
        (course.x_mm + normal_x * offset).astype(np.float32),
        (course.y_mm + normal_y * offset).astype(np.float32),
        offset.astype(np.float32),
        generation_s,
        candidate_id,
        candidate_name,
    )


def _candidate_time(
    course: Course,
    path: GeneratedPath,
    config: PlannerConfig,
) -> float | None:
    """候補を1本だけ局所制約確認・ATTACK速度評価し、合格時だけ時間を返す。"""

    if not (
        np.isfinite(path.x_mm).all()
        and np.isfinite(path.y_mm).all()
        and np.isfinite(path.offset_mm).all()
    ):
        return None
    if float(np.max(np.abs(path.offset_mm))) > config.offset_limit_mm + 0.01:
        return None
    if float(np.max(np.abs(np.diff(path.offset_mm)))) > config.max_offset_step_mm:
        return None
    segment_mm = np.hypot(np.diff(path.x_mm), np.diff(path.y_mm))
    if float(np.max(segment_mm)) > config.max_segment_mm:
        return None
    if not path_order_is_forward(
        course.x_mm, course.y_mm, path.x_mm, path.y_mm
    ):
        return None
    plan = plan_speed(path.x_mm, path.y_mm, config)
    segment_m = np.diff(plan.distance_m)
    return float(
        np.sum(2.0 * segment_m / (plan.speed_mps[:-1] + plan.speed_mps[1:]))
    )


def acceleration_limit_from_omega(
    omega_deg_s: np.ndarray, config: PlannerConfig
) -> np.ndarray:
    """過去ATTACKのomegaガードと同じ線形補間で許容縦加速度を作る。"""

    absolute = np.abs(np.asarray(omega_deg_s, dtype=np.float64))
    ratio = (
        absolute - config.max_acceleration_omega_deg_s
    ) / (
        config.min_acceleration_omega_deg_s
        - config.max_acceleration_omega_deg_s
    )
    ratio = np.clip(ratio, 0.0, 1.0)
    return (
        config.max_acceleration_mps2
        + ratio
        * (config.min_acceleration_mps2 - config.max_acceleration_mps2)
    )


def _aalp_limit_mps(
    curvature_slew: np.ndarray, config: PlannerConfig
) -> np.ndarray:
    """探索走行の1 ms角速度差を再現し、元AALPの平方根則を適用する。"""

    # 元実装のyawAccel_segはgyro[deg/s]の1 ms差を時間で割らない。
    # 定速探索なのでalpha=a*kappa+v^2*dκ/dsのaは0とする。
    alpha_rad_s2 = (
        config.search_run_speed_mps**2 * np.abs(curvature_slew)
    )
    alpha_deg_s_per_ms = (
        np.rad2deg(alpha_rad_s2) * config.firmware_sample_s
    )
    scale = np.sqrt(
        np.divide(
            config.max_aalp_deg_s_per_ms,
            alpha_deg_s_per_ms,
            out=np.full_like(alpha_deg_s_per_ms, np.inf),
            where=alpha_deg_s_per_ms > 1.0e-9,
        )
    )
    return np.clip(
        config.search_run_speed_mps * scale,
        config.min_speed_mps,
        config.max_speed_mps,
    )


def plan_speed(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    config: PlannerConfig,
    *,
    use_aalp: bool = True,
) -> SpeedPlan:
    """ATTACK相当のGFCP・AALP・omega依存加減速を固定回数で計画する。"""

    start = perf_counter()
    distance_m = cumulative_distance_m(x_mm, y_mm)
    segment_m = np.diff(distance_m)
    signed_curvature = signed_curvature_per_m(x_mm, y_mm)
    curvature = np.abs(signed_curvature)
    slew = curvature_slew_per_m2(x_mm, y_mm)

    radius_ratio = np.divide(
        1.0,
        0.1 * curvature,
        out=np.full_like(curvature, np.inf),
        where=curvature > 1.0e-9,
    )
    gfcp_limit = np.clip(
        config.gfcp_reference_speed_mps
        * np.power(radius_ratio, config.gfcp_exponent),
        config.min_speed_mps,
        config.max_speed_mps,
    )
    aalp_limit = _aalp_limit_mps(slew, config)
    if not use_aalp:
        aalp_limit = np.full_like(aalp_limit, config.max_speed_mps)

    base_limit = np.minimum.reduce(
        (
            np.full_like(curvature, config.max_speed_mps),
            gfcp_limit,
            aalp_limit,
        )
    )
    speed = np.clip(base_limit, config.min_speed_mps, config.max_speed_mps)
    speed[0] = config.min_speed_mps
    speed[-1] = config.min_speed_mps

    acceleration_limit = np.full_like(speed, config.max_acceleration_mps2)
    deceleration = config.deceleration_mps2 / config.break_kp
    for _ in range(config.speed_scan_iterations):
        omega_deg_s = np.rad2deg(speed * signed_curvature)
        acceleration_limit = acceleration_limit_from_omega(omega_deg_s, config)
        for index in range(1, speed.size):
            segment_acceleration = min(
                acceleration_limit[index - 1], acceleration_limit[index]
            )
            limit = np.sqrt(
                speed[index - 1] ** 2
                + 2.0 * segment_acceleration * segment_m[index - 1]
            )
            if limit < speed[index]:
                speed[index] = limit
        for index in range(speed.size - 2, -1, -1):
            limit = np.sqrt(
                speed[index + 1] ** 2
                + 2.0 * deceleration * segment_m[index]
            )
            if limit < speed[index]:
                speed[index] = limit

    omega_deg_s = np.rad2deg(speed * signed_curvature)
    acceleration_limit = acceleration_limit_from_omega(omega_deg_s, config)
    limit_reason = np.argmin(
        np.vstack(
            (
                np.full_like(speed, config.max_speed_mps),
                gfcp_limit,
                aalp_limit,
            )
        ),
        axis=0,
    ).astype(np.uint8)
    tolerance = 1.0e-5
    for index in range(speed.size):
        if speed[index] >= base_limit[index] - tolerance:
            continue
        acceleration_bound = np.inf
        deceleration_bound = np.inf
        if index > 0:
            segment_acceleration = min(
                acceleration_limit[index - 1], acceleration_limit[index]
            )
            acceleration_bound = np.sqrt(
                speed[index - 1] ** 2
                + 2.0 * segment_acceleration * segment_m[index - 1]
            )
        if index + 1 < speed.size:
            deceleration_bound = np.sqrt(
                speed[index + 1] ** 2 + 2.0 * deceleration * segment_m[index]
            )
        if acceleration_bound <= deceleration_bound:
            limit_reason[index] = LIMIT_ACCELERATION
        else:
            limit_reason[index] = LIMIT_DECELERATION

    elapsed = perf_counter() - start
    return SpeedPlan(
        distance_m,
        speed.astype(np.float32),
        gfcp_limit.astype(np.float32),
        aalp_limit.astype(np.float32),
        acceleration_limit.astype(np.float32),
        limit_reason,
        elapsed,
    )


def evaluate_path(
    course: Course,
    path: GeneratedPath,
    config: PlannerConfig,
    original_length_m: float,
) -> EvaluatedPath:
    plan = plan_speed(path.x_mm, path.y_mm, config)
    gfcp_plan = plan_speed(path.x_mm, path.y_mm, config, use_aalp=False)
    distance_m = plan.distance_m
    speed = plan.speed_mps
    segment_m = np.diff(distance_m)
    predicted_time = float(np.sum(2.0 * segment_m / (speed[:-1] + speed[1:])))
    gfcp_only_time = float(
        np.sum(
            2.0
            * segment_m
            / (gfcp_plan.speed_mps[:-1] + gfcp_plan.speed_mps[1:])
        )
    )
    length_m = float(distance_m[-1])
    radii = radius_mm(path.x_mm, path.y_mm, config.radius_window)
    min_radius = float(np.min(radii[config.radius_window : -config.radius_window]))
    max_offset = float(np.max(np.abs(path.offset_mm)))
    max_slew = float(np.max(np.abs(curvature_slew_per_m2(path.x_mm, path.y_mm))))
    finite = bool(
        np.isfinite(path.x_mm).all()
        and np.isfinite(path.y_mm).all()
        and np.isfinite(speed).all()
    )
    violations: list[str] = []
    if not finite:
        violations.append("非有限値")
    if max_offset > config.offset_limit_mm + 0.01:
        violations.append("オフセット")
    segment_mm = np.hypot(np.diff(path.x_mm), np.diff(path.y_mm))
    if float(np.max(segment_mm)) > config.max_segment_mm:
        violations.append("点間距離")
    if float(np.max(np.abs(np.diff(path.offset_mm)))) > config.max_offset_step_mm:
        violations.append("オフセット変化")
    if max_slew > config.max_curvature_slew_per_m2:
        violations.append("曲率変化")
    if not path_order_is_forward(
        course.x_mm, course.y_mm, path.x_mm, path.y_mm
    ):
        violations.append("経路順序")
    if self_intersection_count(path.x_mm, path.y_mm) > self_intersection_count(
        course.x_mm, course.y_mm
    ):
        violations.append("自己交差")
    if not (
        np.allclose(path.x_mm[[0, -1]], course.x_mm[[0, -1]], atol=1.0e-5)
        and np.allclose(path.y_mm[[0, -1]], course.y_mm[[0, -1]], atol=1.0e-5)
    ):
        violations.append("端部復帰")
    if path.frenet_locked:
        normal_x, normal_y = frenet_normals(course.x_mm, course.y_mm)
        if not (
            np.allclose(
                path.x_mm,
                course.x_mm + normal_x * path.offset_mm,
                atol=1.0e-4,
            )
            and np.allclose(
                path.y_mm,
                course.y_mm + normal_y * path.offset_mm,
                atol=1.0e-4,
            )
        ):
            violations.append("Frenet対応")
    violation = "、".join(violations)
    metrics = Metrics(
        predicted_time,
        length_m,
        (original_length_m - length_m) / original_length_m * 100.0,
        max_offset,
        min_radius,
        max_slew,
        float(np.max(speed)),
        gfcp_only_time,
        not violations,
        violation,
    )
    return EvaluatedPath(
        path,
        metrics,
        distance_m,
        speed,
        gfcp_plan.speed_mps,
        plan.limit_reason,
        plan.elapsed_s + gfcp_plan.elapsed_s,
    )


def run_comparison(course: Course, config: PlannerConfig | None = None) -> Comparison:
    """旧#7と2初期値以上の長窓候補を比較し、最良記述子だけで再生成する。"""

    config = config or PlannerConfig()
    original_path = generate_original(course)
    original_length = path_length_m(original_path.x_mm, original_path.y_mm)
    original = evaluate_path(course, original_path, config, original_length)
    elastic_path = generate_elastic_band(course, config)
    elastic = evaluate_path(course, elastic_path, config, original_length)
    legacy_path = generate_time_candidate(course, config, CANDIDATES[-1])
    legacy_path = replace(
        legacy_path,
        label="旧時間選択型（#7 最大短縮）",
    )
    legacy_time = evaluate_path(course, legacy_path, config, original_length)

    elastic_offset = _project_elastic_offset(course, elastic_path)
    centers = _detect_long_window_centers(course, config)
    curvature = signed_curvature_per_m(
        course.x_mm, course.y_mm, config.radius_window
    )
    def seed_offset(seed_id: int) -> np.ndarray:
        if seed_id == 0:
            return np.zeros(course.point_count, dtype=np.float32)
        if seed_id == 1:
            return elastic_offset.copy()
        return legacy_path.offset_mm.copy()

    search_start = perf_counter()
    best_time = float("inf")
    best_seed = 0
    best_coarse: int | None = None
    best_refine: int | None = None
    candidate_count = 0
    for seed_id in range(3):
        initial = seed_offset(seed_id)
        candidate = _path_from_offset(course, initial, "長窓候補")
        candidate_time = _candidate_time(
            course, candidate, config
        )
        candidate_count += 1
        if candidate_time is not None and candidate_time < best_time:
            best_time = candidate_time
            best_seed = seed_id
            best_coarse = None

        for coarse_index, (width_mm, amplitude_mm) in enumerate(
            config.long_window_coarse
        ):
            offset = _apply_long_windows(
                course,
                config,
                initial,
                centers,
                curvature,
                width_mm,
                amplitude_mm,
            )
            candidate = _path_from_offset(course, offset, "長窓候補")
            candidate_time = _candidate_time(
                course, candidate, config
            )
            candidate_count += 1
            if candidate_time is not None and candidate_time < best_time:
                best_time = candidate_time
                best_seed = seed_id
                best_coarse = coarse_index

    best_offset = seed_offset(best_seed)
    if best_coarse is not None:
        width_mm, amplitude_mm = config.long_window_coarse[best_coarse]
        best_offset = _apply_long_windows(
            course,
            config,
            best_offset,
            centers,
            curvature,
            width_mm,
            amplitude_mm,
        )
    refine_source = best_offset.copy()
    for refine_index, (width_mm, amplitude_mm) in enumerate(
        config.long_window_refine
    ):
        offset = _apply_long_windows(
            course,
            config,
            refine_source,
            centers,
            curvature,
            width_mm,
            amplitude_mm,
        )
        candidate = _path_from_offset(course, offset, "長窓微調整候補")
        candidate_time = _candidate_time(
            course, candidate, config
        )
        candidate_count += 1
        if candidate_time is not None and candidate_time < best_time:
            best_time = candidate_time
            best_refine = refine_index

    if best_refine is not None:
        width_mm, amplitude_mm = config.long_window_refine[best_refine]
        best_offset = _apply_long_windows(
            course,
            config,
            refine_source,
            centers,
            curvature,
            width_mm,
            amplitude_mm,
        )

    seed_names = ("原ライン初期", "Elastic射影初期", "旧#7初期")
    description = seed_names[best_seed]
    selected_candidate_id = best_seed * 100
    if best_coarse is not None:
        width_mm, amplitude_mm = config.long_window_coarse[best_coarse]
        description += f"・{width_mm:.0f}mm/{amplitude_mm:+.0f}mm"
        selected_candidate_id += (best_coarse + 1) * 10
    if best_refine is not None:
        width_mm, amplitude_mm = config.long_window_refine[best_refine]
        description += f"・微調整{width_mm:.0f}mm/{amplitude_mm:+.1f}mm"
        selected_candidate_id += best_refine + 1

    search_elapsed = perf_counter() - search_start
    improved_path = _path_from_offset(
        course,
        best_offset,
        f"改善後最良経路（{description}）",
        search_elapsed,
        selected_candidate_id,
        description,
    )
    improved = evaluate_path(course, improved_path, config, original_length)

    # Elastic Band自身と旧#7も最終候補に残し、予測時間の悪化を許さない。
    best = min(
        (elastic, legacy_time, improved),
        key=lambda item: (
            not item.metrics.valid,
            item.metrics.predicted_time_s,
        ),
    )
    selected_name = description
    if best is elastic:
        selected_candidate_id = -1
        selected_name = "Elastic Bandフォールバック"
        best = replace(
            elastic,
            path=replace(elastic.path, label="改善後採用経路（Elastic Band）"),
        )
    elif best is legacy_time:
        selected_candidate_id = 7
        selected_name = "旧時間選択型#7フォールバック"
        best = replace(
            legacy_time,
            path=replace(legacy_time.path, label="改善後採用経路（旧#7）"),
        )

    approximate_scans = (
        CANDIDATES[-1].iterations * 4
        + candidate_count * (config.speed_scan_iterations * 3 + 10)
    )
    return Comparison(
        original,
        elastic,
        legacy_time,
        best,
        selected_candidate_id,
        selected_name,
        search_elapsed,
        candidate_count,
        centers,
        approximate_scans,
    )
