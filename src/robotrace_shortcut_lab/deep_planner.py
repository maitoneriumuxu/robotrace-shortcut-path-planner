from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

from .course import load_board_boundary
from .geometry import cumulative_distance_m, signed_curvature_per_m
from .global_planner import (
    _LineTubeGrid,
    _ShortcutEdge,
    _approximate_time,
    _base_edge,
    _build_global_path,
    _geometry_arrays,
    _inside_board_union,
    _k_best_edge_paths,
    _resample_xy,
    _unit_tangents,
    edge_connection_is_valid,
    evaluate_global_path,
    extract_anchor_indices,
)
from .legal_planner import _legalize_edge, run_legal_global_mode
from .legality import (
    WhiteLineContactEvaluator,
    apply_contact_progress_to_path,
    evaluate_contact_sensitivity,
    load_vehicle_footprint,
)
from .model import (
    ContactEvaluation,
    ContactSensitivity,
    Course,
    EvaluatedGlobalPath,
    GlobalPath,
    PlannerConfig,
)
from .portable import plan_speed, run_comparison


@dataclass(frozen=True)
class DeepStageSpec:
    anchor_limit: int
    edge_limit: int
    legal_edge_limit: int
    top_k: int


@dataclass(frozen=True)
class DeepStageRecord:
    name: str
    anchor_count: int
    generated_edge_count: int
    screened_edge_count: int
    legal_edge_count: int
    top_k_count: int
    full_path_count: int
    best_time_s: float
    elapsed_s: float
    cumulative_s: float


@dataclass(frozen=True)
class DeepSearchResult:
    current: EvaluatedGlobalPath
    current_contact: ContactEvaluation
    current_sensitivity: ContactSensitivity
    legal_best: EvaluatedGlobalPath
    legal_contact: ContactEvaluation
    legal_sensitivity: ContactSensitivity
    robust_best: EvaluatedGlobalPath
    robust_contact: ContactEvaluation
    robust_sensitivity: ContactSensitivity
    strong_robust_best: EvaluatedGlobalPath | None
    strong_robust_contact: ContactEvaluation | None
    strong_robust_sensitivity: ContactSensitivity | None
    geometric_lower_bound: EvaluatedGlobalPath
    geometric_lower_contact: ContactEvaluation
    stage_records: tuple[DeepStageRecord, ...]
    convergence_time_s: np.ndarray
    convergence_best_s: np.ndarray
    evaluated_candidate_count: int
    total_s: float
    most_effective_edge: tuple[int, int, str] | None
    most_effective_edge_saving_s: float


DEFAULT_DEEP_STAGES: tuple[DeepStageSpec, ...] = (
    DeepStageSpec(256, 20_000, 1_200, 16),
    DeepStageSpec(384, 50_000, 5_000, 64),
    DeepStageSpec(512, 100_000, 20_000, 256),
)


def _cubic_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    scale0: float,
    scale1: float,
    normal0_mm: float,
    normal1_mm: float,
    count: int,
) -> np.ndarray:
    length = float(np.hypot(*(p1 - p0)))
    n0 = np.array((-t0[1], t0[0]), dtype=np.float64)
    n1 = np.array((-t1[1], t1[0]), dtype=np.float64)
    c0 = p0 + t0 * (length * scale0) + n0 * normal0_mm
    c1 = p1 - t1 * (length * scale1) + n1 * normal1_mm
    u = np.linspace(0.0, 1.0, count, dtype=np.float64)[:, None]
    return (
        (1.0 - u) ** 3 * p0
        + 3.0 * (1.0 - u) ** 2 * u * c0
        + 3.0 * (1.0 - u) * u**2 * c1
        + u**3 * p1
    )


def _quintic_asymmetric(
    p0: np.ndarray,
    p1: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    scale0: float,
    scale1: float,
    count: int,
    second0: np.ndarray | None = None,
    second1: np.ndarray | None = None,
) -> np.ndarray:
    length = float(np.hypot(*(p1 - p0)))
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
        + h10 * (t0 * length * scale0)
        + h20 * a0
        + h01 * p1
        + h11 * (t1 * length * scale1)
        + h21 * a1
    )


def _transition_line_transition(
    p0: np.ndarray,
    p1: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    entry_fraction: float,
    exit_fraction: float,
    count: int,
) -> np.ndarray:
    chord = p1 - p0
    length = float(np.hypot(*chord))
    chord_unit = chord / max(length, 1.0e-9)
    q0 = p0 + chord * entry_fraction
    q1 = p1 - chord * exit_fraction
    first_count = max(4, int(round(count * entry_fraction)) + 1)
    last_count = max(4, int(round(count * exit_fraction)) + 1)
    middle_count = max(3, count - first_count - last_count + 4)
    entry = _cubic_bezier(
        p0, q0, t0, chord_unit, 0.34, 0.34, 0.0, 0.0, first_count
    )
    middle = np.linspace(q0, q1, middle_count, dtype=np.float64)
    exit_curve = _cubic_bezier(
        q1, p1, chord_unit, t1, 0.34, 0.34, 0.0, 0.0, last_count
    )
    return np.vstack((entry[:-1], middle[:-1], exit_curve))


def generate_deep_edge_shapes(
    p0: np.ndarray,
    p1: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    curvature0_per_m: float,
    curvature1_per_m: float,
    interval_mm: float = 20.0,
    *,
    compact: bool = False,
) -> tuple[tuple[str, np.ndarray], ...]:
    """接線尺度、左右非対称、法線制御点を決定論的に展開する。"""

    length = float(np.hypot(*(p1 - p0)))
    count = max(3, int(np.ceil(length / interval_mm)) + 1)
    scales = (0.30, 0.50, 0.70) if compact else (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
    shapes: list[tuple[str, np.ndarray]] = [
        ("直線", np.linspace(p0, p1, count, dtype=np.float64))
    ]
    n0 = np.array((-t0[1], t0[0]), dtype=np.float64)
    n1 = np.array((-t1[1], t1[0]), dtype=np.float64)
    for scale in scales:
        tag = f"s{scale:.2f}"
        shapes.append(
            (f"3次Hermite {tag}", _cubic_bezier(p0, p1, t0, t1, scale, scale, 0.0, 0.0, count))
        )
        shapes.append(
            (f"5次Hermite {tag}", _quintic_asymmetric(p0, p1, t0, t1, scale, scale, count))
        )
        transition_scale = max(0.18, scale * 0.72)
        shapes.append(
            (
                f"入口遷移＋直線＋出口遷移 {tag}",
                _quintic_asymmetric(
                    p0, p1, t0, t1, transition_scale, transition_scale, count
                ),
            )
        )
        second0 = n0 * (curvature0_per_m * 0.001) * (length * scale) ** 2
        second1 = n1 * (curvature1_per_m * 0.001) * (length * scale) ** 2
        shapes.append(
            (
                f"G2近似 {tag}",
                _quintic_asymmetric(
                    p0, p1, t0, t1, scale, scale, count, second0, second1
                ),
            )
        )
        middle = 0.5 * (p0 + p1)
        chord_unit = (p1 - p0) / max(length, 1.0e-9)
        middle_tangent = t0 + t1 + chord_unit
        middle_tangent /= max(float(np.hypot(*middle_tangent)), 1.0e-9)
        first_count = max(3, count // 2 + 1)
        second_count = max(3, count - first_count + 2)
        first = _cubic_bezier(
            p0, middle, t0, middle_tangent, scale, scale, 0.0, 0.0, first_count
        )
        second = _cubic_bezier(
            middle, p1, middle_tangent, t1, scale, scale, 0.0, 0.0, second_count
        )
        shapes.append((f"軽量biarc {tag}", np.vstack((first[:-1], second))))
    asymmetric = ((0.25, 0.55), (0.35, 0.65)) if compact else (
        (0.20, 0.50),
        (0.30, 0.60),
        (0.40, 0.70),
        (0.50, 0.80),
    )
    for scale0, scale1 in asymmetric:
        for left, right in ((scale0, scale1), (scale1, scale0)):
            tag = f"s{left:.2f}/{right:.2f}"
            shapes.append(
                (f"3次Hermite非対称 {tag}", _cubic_bezier(p0, p1, t0, t1, left, right, 0.0, 0.0, count))
            )
            shapes.append(
                (f"5次Hermite非対称 {tag}", _quintic_asymmetric(p0, p1, t0, t1, left, right, count))
            )
    normal = min(20.0, length * 0.05)
    for normal0, normal1 in ((-normal, 0.0), (normal, 0.0), (0.0, -normal), (0.0, normal), (-normal, normal), (normal, -normal)):
        shapes.append(
            (
                f"3次Hermite法線 {normal0:+.0f}/{normal1:+.0f}mm",
                _cubic_bezier(p0, p1, t0, t1, 0.50, 0.50, normal0, normal1, count),
            )
        )
    transition_pairs = ((0.20, 0.20),) if compact else (
        (0.15, 0.15),
        (0.20, 0.20),
        (0.25, 0.15),
        (0.15, 0.25),
    )
    for entry_fraction, exit_fraction in transition_pairs:
        shapes.append(
            (
                f"遷移直線 e{entry_fraction:.2f}/x{exit_fraction:.2f}",
                _transition_line_transition(
                    p0, p1, t0, t1, entry_fraction, exit_fraction, count
                ),
            )
        )
    return tuple(shapes)


def _round_robin_pairs(
    course: Course,
    baseline_x: np.ndarray,
    baseline_y: np.ndarray,
    anchors: np.ndarray,
    config: PlannerConfig,
    pair_limit: int,
) -> list[tuple[int, int, float]]:
    per_start: list[list[tuple[float, int]]] = []
    for start_node in range(anchors.size - 2):
        start_index = int(anchors[start_node])
        pairs: list[tuple[float, int]] = []
        for end_node in range(start_node + 2, anchors.size):
            end_index = int(anchors[end_node])
            skipped = float(course.distance_mm[end_index] - course.distance_mm[start_index])
            if skipped > config.reference_max_skip_mm:
                break
            chord = float(
                np.hypot(
                    baseline_x[end_index] - baseline_x[start_index],
                    baseline_y[end_index] - baseline_y[start_index],
                )
            )
            saving = skipped - chord
            if skipped < config.shortcut_min_skip_mm or saving < config.shortcut_min_saving_mm:
                continue
            score = saving + max(0.0, 500.0 - chord) * 0.8
            pairs.append((score, end_node))
        pairs.sort(key=lambda item: (-item[0], item[1]))
        per_start.append(pairs)
    selected: list[tuple[int, int, float]] = []
    rank = 0
    while len(selected) < pair_limit:
        added = False
        for start_node, pairs in enumerate(per_start):
            if rank >= len(pairs):
                continue
            score, end_node = pairs[rank]
            selected.append((start_node, end_node, score))
            added = True
            if len(selected) >= pair_limit:
                break
        if not added:
            break
        rank += 1
    selected.sort(key=lambda item: (-item[2], item[0], item[1]))
    return selected


def _make_deep_edges(
    course: Course,
    baseline_x: np.ndarray,
    baseline_y: np.ndarray,
    anchors: np.ndarray,
    config: PlannerConfig,
    edge_limit: int,
    boundary,
    witness_radius_mm: float,
    *,
    compact_shapes: bool,
) -> tuple[list[_ShortcutEdge], int]:
    tangent_x, tangent_y = _unit_tangents(baseline_x, baseline_y)
    curvature = signed_curvature_per_m(baseline_x, baseline_y, config.radius_window)
    probe_shapes = generate_deep_edge_shapes(
        np.array((0.0, 0.0)),
        np.array((1000.0, 0.0)),
        np.array((1.0, 0.0)),
        np.array((1.0, 0.0)),
        0.0,
        0.0,
        compact=compact_shapes,
    )
    pair_limit = max(1, int(np.ceil(edge_limit / len(probe_shapes))))
    pairs = _round_robin_pairs(
        course, baseline_x, baseline_y, anchors, config, pair_limit
    )
    line_tube = _LineTubeGrid(
        course.x_mm,
        course.y_mm,
        witness_radius_mm + config.white_line_half_width_mm,
    )
    edges: list[_ShortcutEdge] = []
    edge_id = 1
    considered = 0
    for start_node, end_node, _ in pairs:
        start_index = int(anchors[start_node])
        end_index = int(anchors[end_node])
        p0 = np.array((baseline_x[start_index], baseline_y[start_index]), dtype=np.float64)
        p1 = np.array((baseline_x[end_index], baseline_y[end_index]), dtype=np.float64)
        t0 = np.array((tangent_x[start_index], tangent_y[start_index]), dtype=np.float64)
        t1 = np.array((tangent_x[end_index], tangent_y[end_index]), dtype=np.float64)
        skipped = float(course.distance_mm[end_index] - course.distance_mm[start_index])
        shapes = generate_deep_edge_shapes(
            p0,
            p1,
            t0,
            t1,
            float(curvature[start_index]),
            float(curvature[end_index]),
            compact=compact_shapes,
        )
        for kind, points in shapes:
            considered += 1
            if len(edges) >= edge_limit:
                break
            x, y, target, _ = _resample_xy(points[:, 0], points[:, 1], 20.0)
            if x.size < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
                continue
            if not edge_connection_is_valid(
                np.column_stack((x, y)), t0, t1, config.connector_max_angle_deg
            ):
                continue
            if boundary is not None and not all(
                _inside_board_union(float(px), float(py), boundary)
                for px, py in zip(x[::5], y[::5], strict=True)
            ):
                continue
            if not line_tube.contains_path(x, y):
                continue
            progress = np.interp(
                target,
                (0.0, max(float(target[-1]), 1.0e-6)),
                (float(start_index), float(end_index)),
            ).astype(np.float32)
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
                    progress,
                    _approximate_time(x, y, config),
                    skipped,
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
            )
            edge_id += 1
        if len(edges) >= edge_limit:
            break
    return edges, considered


def _evaluate_legal_geometry_pair(
    course: Course,
    path,
    evaluator: WhiteLineContactEvaluator,
    config: PlannerConfig,
    original_length_m: float,
) -> tuple[EvaluatedGlobalPath | None, EvaluatedGlobalPath | None, ContactEvaluation]:
    contact = evaluator.evaluate(
        path.x_mm,
        path.y_mm,
        path.yaw_rad,
        start_segment=0,
        end_segment=course.point_count - 2,
        compute_metrics=False,
    )
    if not contact.legal:
        return None, None, contact
    contacted = apply_contact_progress_to_path(path, contact, course)
    lower = evaluate_global_path(
        course,
        contacted,
        config,
        original_length_m,
        None,
        enforce_line_rule=False,
        enforce_geometry=False,
        detailed_interactions=False,
        enforce_board=False,
    )
    normal = evaluate_global_path(
        course,
        contacted,
        config,
        original_length_m,
        None,
        enforce_line_rule=False,
        enforce_geometry=True,
        detailed_interactions=False,
        enforce_board=False,
    )
    return normal if normal.metrics.valid else None, lower if lower.metrics.valid else None, contact


def _detailed_contact(
    course: Course,
    evaluated: EvaluatedGlobalPath,
    evaluator: WhiteLineContactEvaluator,
    config: PlannerConfig,
    original_length_m: float,
    *,
    enforce_geometry: bool = True,
) -> tuple[EvaluatedGlobalPath, ContactEvaluation, ContactSensitivity]:
    contact = evaluator.evaluate(
        evaluated.path.x_mm,
        evaluated.path.y_mm,
        evaluated.path.yaw_rad,
        start_segment=0,
        end_segment=course.point_count - 2,
        compute_metrics=True,
    )
    contacted = apply_contact_progress_to_path(evaluated.path, contact, course)
    detailed = evaluate_global_path(
        course,
        contacted,
        config,
        original_length_m,
        None,
        enforce_line_rule=False,
        enforce_geometry=enforce_geometry,
        detailed_interactions=True,
        enforce_board=False,
    )
    sensitivity = evaluate_contact_sensitivity(
        evaluator,
        detailed.path.x_mm,
        detailed.path.y_mm,
        detailed.path.yaw_rad,
        config,
    )
    return detailed, contact, sensitivity


def _seed_edges_from_path(
    path,
    course: Course,
    config: PlannerConfig,
) -> tuple[_ShortcutEdge, ...]:
    """現行合法経路の採用辺をdeep局所探索の初期値へ復元する。"""

    restored: list[_ShortcutEdge] = []
    for edge_id, start_index, end_index, kind in path.selected_edges:
        mask = path.shortcut_edge_id == edge_id
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            continue
        first = max(0, int(indices[0]) - 1)
        last = min(path.x_mm.size - 1, int(indices[-1]) + 1)
        x = np.asarray(path.x_mm[first : last + 1], dtype=np.float32)
        y = np.asarray(path.y_mm[first : last + 1], dtype=np.float32)
        distance = cumulative_distance_m(x, y) * 1000.0
        progress = np.interp(
            distance,
            (0.0, max(float(distance[-1]), 1.0e-6)),
            (float(start_index), float(end_index)),
        ).astype(np.float32)
        restored.append(
            _ShortcutEdge(
                int(edge_id),
                0,
                1,
                int(start_index),
                int(end_index),
                str(kind),
                x,
                y,
                progress,
                _approximate_time(x, y, config),
                float(course.distance_mm[end_index] - course.distance_mm[start_index]),
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
        )
    return tuple(sorted(restored, key=lambda edge: (edge.start_index, edge.end_index)))


def _strong_robust(sensitivity: ContactSensitivity) -> bool:
    position_index = int(np.where(sensitivity.position_error_mm == 5.0)[0][0])
    yaw_index = int(np.where(sensitivity.yaw_error_deg == 2.0)[0][0])
    return bool(
        sensitivity.position_all_legal[position_index]
        and sensitivity.yaw_all_legal[yaw_index]
    )


def _save_checkpoint(
    path: Path,
    best: EvaluatedGlobalPath,
    stage_records: list[DeepStageRecord],
    elapsed_s: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_mm=best.path.x_mm,
        y_mm=best.path.y_mm,
        speed_mps=best.speed_mps,
        source_progress_index=best.path.source_progress_index,
        predicted_time_s=np.asarray((best.metrics.predicted_time_s,)),
        elapsed_s=np.asarray((elapsed_s,)),
        stages=np.asarray(
            [
                (
                    record.anchor_count,
                    record.generated_edge_count,
                    record.screened_edge_count,
                    record.legal_edge_count,
                    record.top_k_count,
                    record.full_path_count,
                    record.best_time_s,
                    record.elapsed_s,
                )
                for record in stage_records
            ],
            dtype=np.float64,
        ),
    )


def _save_seed_path(path: Path, course: Course, seed: GlobalPath) -> None:
    """高価な合法seed探索の結果だけを再利用可能な中間結果として保存する。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    selected = seed.selected_edges
    np.savez_compressed(
        path,
        course_id=np.asarray((course.course_id,)),
        course_point_count=np.asarray((course.point_count,), dtype=np.int32),
        x_mm=seed.x_mm,
        y_mm=seed.y_mm,
        cumulative_distance_mm=seed.cumulative_distance_mm,
        source_progress_index=seed.source_progress_index,
        source_progress_distance_mm=seed.source_progress_distance_mm,
        shortcut_edge_id=seed.shortcut_edge_id,
        deliberate_line_crossing=seed.deliberate_line_crossing,
        yaw_rad=seed.yaw_rad,
        curvature_per_m=seed.curvature_per_m,
        curvature_slew_per_m2=seed.curvature_slew_per_m2,
        edge_number=np.asarray([item[0] for item in selected], dtype=np.int32),
        edge_start=np.asarray([item[1] for item in selected], dtype=np.int32),
        edge_end=np.asarray([item[2] for item in selected], dtype=np.int32),
        edge_kind=np.asarray([item[3] for item in selected], dtype=np.str_),
    )


def _load_seed_path(path: Path, course: Course) -> GlobalPath | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if (
                str(data["course_id"][0]) != course.course_id
                or int(data["course_point_count"][0]) != course.point_count
            ):
                return None
            selected = tuple(
                (
                    int(number),
                    int(start),
                    int(end),
                    str(kind),
                )
                for number, start, end, kind in zip(
                    data["edge_number"],
                    data["edge_start"],
                    data["edge_end"],
                    data["edge_kind"],
                    strict=True,
                )
            )
            return GlobalPath(
                "現行全LINE合法seed（保存済み探索結果）",
                data["x_mm"].astype(np.float32),
                data["y_mm"].astype(np.float32),
                data["cumulative_distance_mm"].astype(np.float32),
                data["source_progress_index"].astype(np.float32),
                data["source_progress_distance_mm"].astype(np.float32),
                data["shortcut_edge_id"].astype(np.int32),
                data["deliberate_line_crossing"].astype(np.bool_),
                data["yaw_rad"].astype(np.float32),
                data["curvature_per_m"].astype(np.float32),
                data["curvature_slew_per_m2"].astype(np.float32),
                np.zeros(data["x_mm"].size, dtype=np.float32),
                0.0,
                selected,
            )
    except (OSError, KeyError, ValueError):
        return None


def _rank_edges(
    edges: list[_ShortcutEdge],
    baseline_time: np.ndarray,
) -> list[_ShortcutEdge]:
    return sorted(
        edges,
        key=lambda edge: (
            -(
                float(baseline_time[edge.end_index] - baseline_time[edge.start_index])
                - edge.approximate_time_s
            ),
            edge.start_index,
            edge.end_index,
            edge.kind,
        ),
    )


def _legalize_batched(
    edges: list[_ShortcutEdge],
    evaluator: WhiteLineContactEvaluator,
    course: Course,
    config: PlannerConfig,
    limit: int,
    deadline: float,
) -> tuple[list[_ShortcutEdge], int]:
    legal: list[_ShortcutEdge] = []
    checked = 0
    selected = edges[:limit]
    for start in range(0, len(selected), 64):
        if perf_counter() >= deadline:
            break
        batch = selected[start : start + 64]
        with ThreadPoolExecutor(max_workers=4) as executor:
            evaluated = list(
                executor.map(
                    lambda edge: _legalize_edge(edge, evaluator, course, config),
                    batch,
                )
            )
        checked += len(batch)
        legal.extend(edge for edge in evaluated if edge.reference_valid)
    return legal, checked


def _evaluate_sequences(
    course: Course,
    sequences: list[tuple[int, ...]],
    edge_map: dict[int, _ShortcutEdge],
    evaluator: WhiteLineContactEvaluator,
    config: PlannerConfig,
    original_length_m: float,
    deadline: float,
    label: str,
) -> tuple[
    list[tuple[EvaluatedGlobalPath, ContactEvaluation, tuple[_ShortcutEdge, ...]]],
    list[tuple[EvaluatedGlobalPath, ContactEvaluation, tuple[_ShortcutEdge, ...]]],
    int,
]:
    previews: list[tuple[float, tuple[int, ...], object]] = []
    for sequence in sequences:
        if all(edge_id < 0 for edge_id in sequence):
            continue
        shortcut_count = sum(edge_id > 0 for edge_id in sequence)
        if not 1 <= shortcut_count <= 4:
            continue
        path = _build_global_path(
            course,
            sequence,
            edge_map,
            config,
            label,
            0.0,
            allow_synthetic_progress=False,
        )
        preview = evaluate_global_path(
            course,
            path,
            config,
            original_length_m,
            None,
            enforce_line_rule=False,
            enforce_geometry=False,
            detailed_interactions=False,
            enforce_board=False,
        )
        if preview.metrics.valid:
            previews.append((preview.metrics.predicted_time_s, sequence, path))
    previews.sort(key=lambda item: (item[0], item[1]))
    legal_routes: list[
        tuple[EvaluatedGlobalPath, ContactEvaluation, tuple[_ShortcutEdge, ...]]
    ] = []
    lower_routes: list[
        tuple[EvaluatedGlobalPath, ContactEvaluation, tuple[_ShortcutEdge, ...]]
    ] = []
    evaluated_count = 0
    for _, sequence, path in previews[:32]:
        if perf_counter() >= deadline:
            break
        normal, lower, contact = _evaluate_legal_geometry_pair(
            course, path, evaluator, config, original_length_m
        )
        evaluated_count += 1
        selected_edges = tuple(edge_map[edge_id] for edge_id in sequence if edge_id > 0)
        if lower is not None:
            lower_routes.append((lower, contact, selected_edges))
        if normal is not None:
            legal_routes.append((normal, contact, selected_edges))
    return legal_routes, lower_routes, evaluated_count


def run_deep_reference(
    course: Course,
    config: PlannerConfig | None = None,
    *,
    stages: tuple[DeepStageSpec, ...] = DEFAULT_DEEP_STAGES,
    budget_s: float = 1_500.0,
    checkpoint_path: str | Path = "outputs/deep_gate_checkpoint.npz",
    seed_cache_path: str | Path = "outputs/deep_gate_seed.npz",
    resume_cache_path: str | Path = "outputs/deep_gate_resume.npz",
    robust_cache_path: str | Path = "outputs/deep_gate_robust.npz",
    progress: Callable[[str], None] | None = None,
) -> DeepSearchResult:
    """固定モデルと全LINE合法性のまま2025用PC品質上限を段階探索する。"""

    config = config or PlannerConfig()
    emit = progress or (lambda _: None)
    total_start = perf_counter()
    deadline = total_start + max(budget_s, 1.0)
    comparison = run_comparison(course, config)
    boundary = load_board_boundary(course)
    footprint = load_vehicle_footprint()
    evaluator = WhiteLineContactEvaluator(course, footprint, config, boundary)
    cached_seed = _load_seed_path(Path(seed_cache_path), course)
    if cached_seed is None:
        _, seed_result, _ = run_legal_global_mode(
            course,
            "embedded-lite",
            config,
            local_comparison=comparison,
            footprint=footprint,
            boundary=boundary,
            evaluate_robust=False,
        )
        if not seed_result.legal or seed_result.contact is None:
            raise RuntimeError("現行の全LINE合法seedを再現できません")
        cached_seed = seed_result.adopted.path
        _save_seed_path(Path(seed_cache_path), course, cached_seed)
    seed_contact = evaluator.evaluate(
        cached_seed.x_mm,
        cached_seed.y_mm,
        cached_seed.yaw_rad,
        start_segment=0,
        end_segment=course.point_count - 2,
        compute_metrics=False,
    )
    if not seed_contact.legal:
        raise RuntimeError("保存されたseedが現在の全LINE合法性ゲートを通りません")
    seed_contacted = apply_contact_progress_to_path(cached_seed, seed_contact, course)
    seed_evaluated = evaluate_global_path(
        course,
        seed_contacted,
        config,
        comparison.original.metrics.length_m,
        None,
        enforce_line_rule=False,
        detailed_interactions=False,
        enforce_board=False,
    )
    if not seed_evaluated.metrics.valid:
        raise RuntimeError(f"保存されたseedの幾何制約違反: {seed_evaluated.metrics.violation}")
    current, current_contact, current_sensitivity = _detailed_contact(
        course,
        seed_evaluated,
        evaluator,
        config,
        comparison.original.metrics.length_m,
    )
    emit(f"deep seed監査: {current.metrics.predicted_time_s:.6f}s")
    legal_best = current
    lower_best = current
    convergence_time = [perf_counter() - total_start]
    convergence_best = [current.metrics.predicted_time_s]
    stage_records: list[DeepStageRecord] = []
    archive: list[tuple[EvaluatedGlobalPath, ContactEvaluation]] = [(current, current_contact)]
    baseline_x = comparison.best.path.x_mm
    baseline_y = comparison.best.path.y_mm
    baseline_plan = plan_speed(baseline_x, baseline_y, config)
    segment_time = 2.0 * np.diff(baseline_plan.distance_m) / np.maximum(
        baseline_plan.speed_mps[:-1] + baseline_plan.speed_mps[1:], 1.0e-6
    )
    baseline_time = np.concatenate((np.zeros(1), np.cumsum(segment_time)))
    witness_radius = max(
        float(np.max(np.hypot(component[:, 0], component[:, 1])))
        for component in footprint.contact_witness_components_mm
    )
    total_evaluated = 0
    best_route_edges = _seed_edges_from_path(current.path, course, config)
    checkpoint = Path(checkpoint_path)
    resume_cache = Path(resume_cache_path)
    resumed_path = _load_seed_path(resume_cache, course)
    if resumed_path is not None:
        resumed_contact = evaluator.evaluate(
            resumed_path.x_mm,
            resumed_path.y_mm,
            resumed_path.yaw_rad,
            start_segment=0,
            end_segment=course.point_count - 2,
            compute_metrics=False,
        )
        if resumed_contact.legal:
            resumed_contacted = apply_contact_progress_to_path(
                resumed_path, resumed_contact, course
            )
            resumed = evaluate_global_path(
                course,
                resumed_contacted,
                config,
                comparison.original.metrics.length_m,
                None,
                enforce_line_rule=False,
                detailed_interactions=False,
                enforce_board=False,
            )
            if (
                resumed.metrics.valid
                and resumed.metrics.predicted_time_s < legal_best.metrics.predicted_time_s
            ):
                legal_best = resumed
                lower_best = resumed
                archive.append((resumed, resumed_contact))
                best_route_edges = _seed_edges_from_path(resumed.path, course, config)
                convergence_time.append(perf_counter() - total_start)
                convergence_best.append(resumed.metrics.predicted_time_s)
                emit(f"deep再開候補: {resumed.metrics.predicted_time_s:.6f}s")
    robust_cache = Path(robust_cache_path)
    robust_warm_path = _load_seed_path(robust_cache, course)
    robust_warm_key: bytes | None = None
    if robust_warm_path is not None:
        robust_warm_contact = evaluator.evaluate(
            robust_warm_path.x_mm,
            robust_warm_path.y_mm,
            robust_warm_path.yaw_rad,
            start_segment=0,
            end_segment=course.point_count - 2,
            compute_metrics=False,
        )
        if robust_warm_contact.legal:
            robust_warm_contacted = apply_contact_progress_to_path(
                robust_warm_path, robust_warm_contact, course
            )
            robust_warm = evaluate_global_path(
                course,
                robust_warm_contacted,
                config,
                comparison.original.metrics.length_m,
                None,
                enforce_line_rule=False,
                detailed_interactions=False,
                enforce_board=False,
            )
            if robust_warm.metrics.valid:
                archive.append((robust_warm, robust_warm_contact))
                robust_warm_key = (
                    robust_warm.path.x_mm.tobytes()
                    + robust_warm.path.y_mm.tobytes()
                )
                emit(
                    "deep robust再開候補: "
                    f"{robust_warm.metrics.predicted_time_s:.6f}s"
                )

    for stage_index, spec in enumerate(stages):
        if perf_counter() >= deadline:
            break
        stage_start = perf_counter()
        stage_config = replace(
            config,
            reference_anchor_limit=spec.anchor_limit,
            reference_edge_limit=spec.edge_limit,
            legal_reference_edge_check_limit=spec.legal_edge_limit,
            legal_reference_top_k=spec.top_k,
        )
        anchors = extract_anchor_indices(
            course, baseline_x, baseline_y, stage_config, "reference"
        )
        raw_edges, _ = _make_deep_edges(
            course,
            baseline_x,
            baseline_y,
            anchors,
            stage_config,
            spec.edge_limit,
            boundary,
            witness_radius,
            compact_shapes=stage_index == 0,
        )
        ranked = _rank_edges(raw_edges, baseline_time)
        legal_edges, screened = _legalize_batched(
            ranked,
            evaluator,
            course,
            stage_config,
            spec.legal_edge_limit,
            deadline,
        )
        base_edges = [
            _base_edge(
                node,
                anchors,
                baseline_x,
                baseline_y,
                baseline_plan.distance_m,
                baseline_time,
            )
            for node in range(anchors.size - 1)
        ]
        sequences, _ = _k_best_edge_paths(
            anchors,
            legal_edges,
            base_edges,
            spec.top_k,
            reference_only=True,
        )
        edge_map = {edge.edge_id: edge for edge in base_edges + legal_edges}
        legal_routes, lower_routes, evaluated_count = _evaluate_sequences(
            course,
            sequences,
            edge_map,
            evaluator,
            stage_config,
            comparison.original.metrics.length_m,
            deadline,
            f"deep-reference stage {stage_index + 1}",
        )
        total_evaluated += screened + evaluated_count
        for evaluated, contact, edges in sorted(
            lower_routes, key=lambda item: item[0].metrics.predicted_time_s
        ):
            if evaluated.metrics.predicted_time_s < lower_best.metrics.predicted_time_s:
                lower_best = evaluated
        for evaluated, contact, edges in sorted(
            legal_routes, key=lambda item: item[0].metrics.predicted_time_s
        ):
            archive.append((evaluated, contact))
            if evaluated.metrics.predicted_time_s < legal_best.metrics.predicted_time_s - 1.0e-9:
                legal_best = evaluated
                best_route_edges = edges
                convergence_time.append(perf_counter() - total_start)
                convergence_best.append(legal_best.metrics.predicted_time_s)
                _save_checkpoint(checkpoint, legal_best, stage_records, perf_counter() - total_start)
                _save_seed_path(resume_cache, course, legal_best.path)
                emit(
                    f"stage-{stage_index + 1} 改善: "
                    f"{legal_best.metrics.predicted_time_s:.6f}s"
                )
        record = DeepStageRecord(
            f"stage-{stage_index + 1}",
            int(anchors.size),
            len(raw_edges),
            screened,
            len(legal_edges),
            len(sequences),
            evaluated_count,
            legal_best.metrics.predicted_time_s,
            perf_counter() - stage_start,
            perf_counter() - total_start,
        )
        stage_records.append(record)
        emit(
            f"{record.name}: anchors={record.anchor_count}, edges={record.generated_edge_count}, "
            f"checked={record.screened_edge_count}, legal={record.legal_edge_count}, "
            f"K={record.top_k_count}, full={record.full_path_count}, "
            f"best={record.best_time_s:.6f}s, {record.elapsed_s:.1f}s"
        )
        _save_checkpoint(checkpoint, legal_best, stage_records, perf_counter() - total_start)

    # 上位経路の各辺を交互に微調整する。入口・出口は距離差から汎用算出する。
    if best_route_edges and perf_counter() < deadline:
        local_start = perf_counter()
        local_checked = 0
        local_full = 0
        active_edges = list(best_route_edges[:4])
        for pass_index, offsets_mm in enumerate(
            ((-300.0, -200.0, -100.0, 0.0, 100.0, 200.0, 300.0),
             (-50.0, -20.0, 0.0, 20.0, 50.0))
        ):
            for edge_position, active in enumerate(tuple(active_edges)):
                if perf_counter() >= deadline:
                    break
                start_distance = float(course.distance_mm[active.start_index])
                end_distance = float(course.distance_mm[active.end_index])
                local_pairs: list[tuple[int, int]] = []
                for start_offset in offsets_mm:
                    start_index = int(
                        np.clip(
                            np.searchsorted(course.distance_mm, start_distance + start_offset),
                            0,
                            course.point_count - 2,
                        )
                    )
                    for end_offset in offsets_mm:
                        end_index = int(
                            np.clip(
                                np.searchsorted(course.distance_mm, end_distance + end_offset),
                                start_index + 2,
                                course.point_count - 1,
                            )
                        )
                        if start_index < end_index:
                            local_pairs.append((start_index, end_index))
                local_pairs = sorted(set(local_pairs))
                anchors = np.asarray(
                    sorted({value for pair in local_pairs for value in pair}), dtype=np.int32
                )
                node_for_index = {int(value): index for index, value in enumerate(anchors)}
                local_edges: list[_ShortcutEdge] = []
                edge_id = 1
                tangent_x, tangent_y = _unit_tangents(baseline_x, baseline_y)
                curvature = signed_curvature_per_m(
                    baseline_x, baseline_y, config.radius_window
                )
                for start_index, end_index in local_pairs:
                    p0 = np.array((baseline_x[start_index], baseline_y[start_index]))
                    p1 = np.array((baseline_x[end_index], baseline_y[end_index]))
                    t0 = np.array((tangent_x[start_index], tangent_y[start_index]))
                    t1 = np.array((tangent_x[end_index], tangent_y[end_index]))
                    for kind, points in generate_deep_edge_shapes(
                        p0,
                        p1,
                        t0,
                        t1,
                        float(curvature[start_index]),
                        float(curvature[end_index]),
                        compact=pass_index == 0,
                    ):
                        x, y, target, _ = _resample_xy(points[:, 0], points[:, 1], 20.0)
                        if not edge_connection_is_valid(
                            np.column_stack((x, y)), t0, t1, config.connector_max_angle_deg
                        ):
                            continue
                        progress = np.interp(
                            target,
                            (0.0, max(float(target[-1]), 1.0e-6)),
                            (float(start_index), float(end_index)),
                        ).astype(np.float32)
                        local_edges.append(
                            _ShortcutEdge(
                                edge_id,
                                node_for_index[start_index],
                                node_for_index[end_index],
                                start_index,
                                end_index,
                                kind,
                                x,
                                y,
                                progress,
                                _approximate_time(x, y, config),
                                float(course.distance_mm[end_index] - course.distance_mm[start_index]),
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
                        )
                        edge_id += 1
                ranked = _rank_edges(local_edges, baseline_time)
                legalized, checked = _legalize_batched(
                    ranked, evaluator, course, config, 500, deadline
                )
                local_checked += checked
                replacements = legalized[:12]
                route_options: list[
                    tuple[EvaluatedGlobalPath, ContactEvaluation, tuple[_ShortcutEdge, ...]]
                ] = []
                for replacement in replacements:
                    if perf_counter() >= deadline:
                        break
                    proposed = list(active_edges)
                    proposed[edge_position] = replacement
                    proposed.sort(key=lambda item: (item.start_index, item.end_index))
                    if any(
                        left.end_index >= right.start_index
                        for left, right in zip(proposed, proposed[1:])
                    ):
                        continue
                    route_path = assemble_deep_path(
                        course,
                        baseline_x,
                        baseline_y,
                        tuple(proposed),
                        config,
                        f"deep局所最適化 pass {pass_index + 1}",
                    )
                    normal, lower, contact = _evaluate_legal_geometry_pair(
                        course,
                        route_path,
                        evaluator,
                        config,
                        comparison.original.metrics.length_m,
                    )
                    local_full += 1
                    if lower is not None and lower.metrics.predicted_time_s < lower_best.metrics.predicted_time_s:
                        lower_best = lower
                    if normal is not None:
                        route_options.append((normal, contact, tuple(proposed)))
                if route_options:
                    candidate, contact, proposed_edges = min(
                        route_options, key=lambda item: item[0].metrics.predicted_time_s
                    )
                    archive.extend((item[0], item[1]) for item in route_options)
                    if candidate.metrics.predicted_time_s < legal_best.metrics.predicted_time_s - 1.0e-9:
                        legal_best = candidate
                        active_edges = list(proposed_edges)
                        convergence_time.append(perf_counter() - total_start)
                        convergence_best.append(legal_best.metrics.predicted_time_s)
                        _save_checkpoint(checkpoint, legal_best, stage_records, perf_counter() - total_start)
                        _save_seed_path(resume_cache, course, legal_best.path)
                        emit(
                            f"局所pass {pass_index + 1} 辺{edge_position + 1}改善: "
                            f"{legal_best.metrics.predicted_time_s:.6f}s"
                        )
        total_evaluated += local_checked + local_full
        stage_records.append(
            DeepStageRecord(
                "local-coordinate",
                0,
                local_checked,
                local_checked,
                0,
                0,
                local_full,
                legal_best.metrics.predicted_time_s,
                perf_counter() - local_start,
                perf_counter() - total_start,
            )
        )
        emit(
            f"local-coordinate: checked={local_checked}, full={local_full}, "
            f"best={legal_best.metrics.predicted_time_s:.6f}s"
        )

    # legal archiveを完全指標と取付誤差で再順位付けする。
    unique: dict[bytes, tuple[EvaluatedGlobalPath, ContactEvaluation]] = {}
    for evaluated, contact in archive:
        key = evaluated.path.x_mm.tobytes() + evaluated.path.y_mm.tobytes()
        previous = unique.get(key)
        if previous is None or evaluated.metrics.predicted_time_s < previous[0].metrics.predicted_time_s:
            unique[key] = (evaluated, contact)
    ranked_archive = sorted(unique.values(), key=lambda item: item[0].metrics.predicted_time_s)
    detailed_archive: list[
        tuple[EvaluatedGlobalPath, ContactEvaluation, ContactSensitivity]
    ] = [(current, current_contact, current_sensitivity)]
    current_key = current.path.x_mm.tobytes() + current.path.y_mm.tobytes()
    legal_best_key = legal_best.path.x_mm.tobytes() + legal_best.path.y_mm.tobytes()
    mandatory_detail_keys = {legal_best_key}
    if robust_warm_key is not None:
        mandatory_detail_keys.add(robust_warm_key)
    for evaluated, _ in ranked_archive[:12]:
        evaluated_key = evaluated.path.x_mm.tobytes() + evaluated.path.y_mm.tobytes()
        if evaluated_key == current_key:
            continue
        if perf_counter() >= deadline and evaluated_key not in mandatory_detail_keys:
            break
        else:
            detailed_archive.append(
                _detailed_contact(
                    course,
                    evaluated,
                    evaluator,
                    config,
                    comparison.original.metrics.length_m,
                )
            )
    if not detailed_archive:
        detailed_archive.append((current, current_contact, current_sensitivity))
    detailed_archive.sort(key=lambda item: item[0].metrics.predicted_time_s)
    legal_best, legal_contact, legal_sensitivity = detailed_archive[0]
    robust_candidates = [
        item
        for item in detailed_archive
        if item[1].robust and item[2].robust_2mm_1deg
    ]
    robust_best, robust_contact, robust_sensitivity = (
        robust_candidates[0]
        if robust_candidates
        else (current, current_contact, current_sensitivity)
    )
    strong_candidates = [item for item in detailed_archive if _strong_robust(item[2])]
    if strong_candidates:
        strong_best, strong_contact, strong_sensitivity = strong_candidates[0]
    else:
        strong_best = strong_contact = strong_sensitivity = None
    lower_key = lower_best.path.x_mm.tobytes() + lower_best.path.y_mm.tobytes()
    if lower_key == current_key:
        lower_best, lower_contact = current, current_contact
    else:
        lower_contact = evaluator.evaluate(
            lower_best.path.x_mm,
            lower_best.path.y_mm,
            lower_best.path.yaw_rad,
            start_segment=0,
            end_segment=course.point_count - 2,
            compute_metrics=True,
        )
        lower_contacted = apply_contact_progress_to_path(
            lower_best.path, lower_contact, course
        )
        lower_best = evaluate_global_path(
            course,
            lower_contacted,
            config,
            comparison.original.metrics.length_m,
            None,
            enforce_line_rule=False,
            enforce_geometry=False,
            detailed_interactions=True,
            enforce_board=False,
        )

    most_effective = None
    saving = 0.0
    if legal_best.path.selected_edges:
        edge_times: list[tuple[float, tuple[int, int, str]]] = []
        current_source_time = np.interp(
            course.distance_mm * 0.001,
            current.path.source_progress_distance_mm * 0.001,
            current.cumulative_time_s,
        )
        best_source_time = np.interp(
            course.distance_mm * 0.001,
            legal_best.path.source_progress_distance_mm * 0.001,
            legal_best.cumulative_time_s,
        )
        for _, start_index, end_index, kind in legal_best.path.selected_edges:
            delta = float(
                (current_source_time[end_index] - current_source_time[start_index])
                - (best_source_time[end_index] - best_source_time[start_index])
            )
            edge_times.append((delta, (start_index, end_index, kind)))
        saving, most_effective = max(edge_times, key=lambda item: item[0])

    total_s = perf_counter() - total_start
    _save_checkpoint(checkpoint, legal_best, stage_records, total_s)
    _save_seed_path(robust_cache, course, robust_best.path)
    return DeepSearchResult(
        current,
        current_contact,
        current_sensitivity,
        legal_best,
        legal_contact,
        legal_sensitivity,
        robust_best,
        robust_contact,
        robust_sensitivity,
        strong_best,
        strong_contact,
        strong_sensitivity,
        lower_best,
        lower_contact,
        tuple(stage_records),
        np.asarray(convergence_time, dtype=np.float32),
        np.asarray(convergence_best, dtype=np.float32),
        total_evaluated,
        total_s,
        most_effective,
        saving,
    )


def assemble_deep_path(
    course: Course,
    baseline_x: np.ndarray,
    baseline_y: np.ndarray,
    edges: tuple[_ShortcutEdge, ...],
    config: PlannerConfig,
    label: str,
):
    """任意の非重複辺を基準経路へ挿入して10mm再サンプリングする。"""

    ordered = tuple(sorted(edges, key=lambda edge: (edge.start_index, edge.end_index)))
    if any(
        left.end_index >= right.start_index
        for left, right in zip(ordered, ordered[1:])
    ):
        raise ValueError("deep辺が重複しています")
    raw_x: list[np.ndarray] = []
    raw_y: list[np.ndarray] = []
    raw_id: list[np.ndarray] = []
    cursor = 0
    selected: list[tuple[int, int, int, str]] = []
    for number, edge in enumerate(ordered, start=1):
        raw_x.append(baseline_x[cursor : edge.start_index + 1])
        raw_y.append(baseline_y[cursor : edge.start_index + 1])
        raw_id.append(np.full(edge.start_index + 1 - cursor, -1, dtype=np.int32))
        raw_x.append(edge.x_mm[1:])
        raw_y.append(edge.y_mm[1:])
        raw_id.append(np.full(edge.x_mm.size - 1, number, dtype=np.int32))
        selected.append((number, edge.start_index, edge.end_index, edge.kind))
        cursor = edge.end_index + 1
    raw_x.append(baseline_x[cursor:])
    raw_y.append(baseline_y[cursor:])
    raw_id.append(np.full(baseline_x.size - cursor, -1, dtype=np.int32))
    x0 = np.concatenate(raw_x).astype(np.float64)
    y0 = np.concatenate(raw_y).astype(np.float64)
    edge_id0 = np.concatenate(raw_id)
    segment = np.hypot(np.diff(x0), np.diff(y0))
    distance0 = np.concatenate((np.zeros(1), np.cumsum(segment)))
    keep = np.concatenate(([True], np.diff(distance0) > 1.0e-6))
    x0, y0, edge_id0, distance0 = x0[keep], y0[keep], edge_id0[keep], distance0[keep]
    target = np.arange(
        0.0, distance0[-1], config.global_resample_interval_mm, dtype=np.float64
    )
    if target.size == 0 or target[-1] < distance0[-1] - 1.0e-6:
        target = np.append(target, distance0[-1])
    x = np.interp(target, distance0, x0).astype(np.float32)
    y = np.interp(target, distance0, y0).astype(np.float32)
    nearest = np.clip(
        np.searchsorted(distance0, target, side="right") - 1,
        0,
        edge_id0.size - 1,
    )
    edge_ids = edge_id0[nearest].astype(np.int32)
    x[0], y[0] = course.x_mm[0], course.y_mm[0]
    x[-1], y[-1] = course.x_mm[-1], course.y_mm[-1]
    yaw, curvature, slew = _geometry_arrays(x, y)
    from .model import GlobalPath

    return GlobalPath(
        label,
        x,
        y,
        target.astype(np.float32),
        np.full(x.size, -1.0, dtype=np.float32),
        np.full(x.size, -1.0, dtype=np.float32),
        edge_ids,
        np.zeros(x.size, dtype=np.bool_),
        yaw,
        curvature,
        slew,
        np.zeros(x.size, dtype=np.float32),
        0.0,
        tuple(selected),
    )
