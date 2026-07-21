from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from .geometry import (
    cumulative_distance_m,
    curvature_slew_per_m2,
    expanded_mask,
    frenet_normals,
    path_length_m,
    radius_mm,
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
)


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
        bad = radii < np.float32(config.min_radius_mm)
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
            bad = radius_mm(x, y, config.radius_window) < config.min_radius_mm
            if bool(np.any(bad)):
                affected = expanded_mask(bad, config.radius_window)
                x[affected] += (source_x[affected] - x[affected]) * 0.08
                y[affected] += (source_y[affected] - y[affected]) * 0.08

        x[0], y[0], x[-1], y[-1] = source_x[0], source_y[0], source_x[-1], source_y[-1]

    # 最終制約処理も同一indexの原ライン方向へ戻すだけにする。
    for _ in range(80):
        bad = radius_mm(x, y, config.radius_window) < config.min_radius_mm
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
            bad = radius_mm(x, y, config.radius_window) < config.min_radius_mm
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


def plan_speed(
    x_mm: np.ndarray, y_mm: np.ndarray, config: PlannerConfig
) -> tuple[np.ndarray, np.ndarray]:
    """実点間距離を使うGFCP・角速度・角加速度ペナルティ付き速度計画。"""

    distance_m = cumulative_distance_m(x_mm, y_mm)
    segment_m = np.diff(distance_m)
    curvature = np.abs(signed_curvature_per_m(x_mm, y_mm))
    slew = np.abs(curvature_slew_per_m2(x_mm, y_mm))

    radius_ratio = np.divide(
        1.0,
        0.1 * curvature,
        out=np.full_like(curvature, np.inf),
        where=curvature > 1.0e-9,
    )
    gfcp_limit = config.gfcp_reference_speed_mps * np.power(
        radius_ratio, config.gfcp_exponent
    )
    omega_limit = np.divide(
        np.deg2rad(config.max_omega_deg_s),
        curvature,
        out=np.full_like(curvature, np.inf),
        where=curvature > 1.0e-9,
    )
    slew_limit = np.sqrt(
        np.divide(
            config.max_angular_accel_rad_s2,
            slew,
            out=np.full_like(slew, np.inf),
            where=slew > 1.0e-9,
        )
    )
    speed = np.minimum.reduce(
        (
            np.full_like(curvature, config.max_speed_mps),
            gfcp_limit,
            omega_limit,
            slew_limit,
        )
    )
    speed = np.clip(speed, config.min_speed_mps, config.max_speed_mps)
    speed[0] = config.min_speed_mps
    speed[-1] = config.min_speed_mps

    # 固定回数の前後スキャン。各区間の実際の長さを使用する。
    for _ in range(4):
        for index in range(1, speed.size):
            limit = np.sqrt(
                speed[index - 1] ** 2
                + 2.0 * config.acceleration_mps2 * segment_m[index - 1]
            )
            if limit < speed[index]:
                speed[index] = limit
        for index in range(speed.size - 2, -1, -1):
            limit = np.sqrt(
                speed[index + 1] ** 2
                + 2.0 * config.deceleration_mps2 * segment_m[index]
            )
            if limit < speed[index]:
                speed[index] = limit

    return distance_m, speed.astype(np.float32)


def evaluate_path(
    course: Course,
    path: GeneratedPath,
    config: PlannerConfig,
    original_length_m: float,
) -> EvaluatedPath:
    distance_m, speed = plan_speed(path.x_mm, path.y_mm, config)
    segment_m = np.diff(distance_m)
    predicted_time = float(np.sum(2.0 * segment_m / (speed[:-1] + speed[1:])))
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
    if min_radius < config.min_radius_mm - 0.01:
        violations.append("最小半径")
    violation = "、".join(violations)
    metrics = Metrics(
        predicted_time,
        length_m,
        (original_length_m - length_m) / original_length_m * 100.0,
        max_offset,
        min_radius,
        max_slew,
        not violations,
        violation,
    )
    return EvaluatedPath(path, metrics, distance_m, speed)


def run_comparison(course: Course, config: PlannerConfig | None = None) -> Comparison:
    """8候補を1本ずつ評価し、最良パラメータだけで再生成する。"""

    config = config or PlannerConfig()
    original_path = generate_original(course)
    original_length = path_length_m(original_path.x_mm, original_path.y_mm)
    original = evaluate_path(course, original_path, config, original_length)
    elastic = evaluate_path(
        course, generate_elastic_band(course, config), config, original_length
    )

    search_start = perf_counter()
    best_evaluated: EvaluatedPath | None = None
    best_weights: CandidateWeights | None = None
    generation_times: list[float] = []
    for weights in CANDIDATES:
        candidate = generate_time_candidate(course, config, weights)
        generation_times.append(candidate.generation_s)
        evaluated = evaluate_path(course, candidate, config, original_length)
        if evaluated.metrics.valid and (
            best_evaluated is None
            or evaluated.metrics.predicted_time_s < best_evaluated.metrics.predicted_time_s
        ):
            best_evaluated = evaluated
            best_weights = weights
    search_elapsed = perf_counter() - search_start
    if best_weights is None:
        raise RuntimeError("制約を満たす時間選択型候補がありません")

    regenerated = generate_time_candidate(course, config, best_weights)
    regenerated = replace(
        regenerated,
        label=f"時間選択型（#{best_weights.candidate_id} {best_weights.name}）",
        generation_s=search_elapsed + regenerated.generation_s,
    )
    best = evaluate_path(course, regenerated, config, original_length)
    return Comparison(
        original,
        elastic,
        best,
        best_weights.candidate_id,
        best_weights.name,
        search_elapsed,
        tuple(generation_times),
    )
