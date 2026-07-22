from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from .course import load_board_boundary
from .geometry import cumulative_distance_m
from .global_planner import (
    _ShortcutEdge,
    _base_edge,
    _build_global_path,
    _geometry_arrays,
    _k_best_edge_paths,
    _make_edges,
    _run_geometric_lower_bound,
    _run_mode,
    evaluate_global_path,
    extract_anchor_indices,
    global_path_from_local,
)
from .legality import (
    WhiteLineContactEvaluator,
    apply_contact_progress_to_path,
    load_vehicle_footprint,
)
from .model import (
    BoardBoundary,
    Comparison,
    ContactEvaluation,
    Course,
    EvaluatedGlobalPath,
    GlobalComparison,
    GlobalSearchResult,
    GlobalSearchStats,
    PlannerConfig,
    VehicleFootprint,
)
from .portable import plan_speed, run_comparison


def _zero_stats(mode: str, elapsed_s: float = 0.0) -> GlobalSearchStats:
    return GlobalSearchStats(
        mode,
        0,
        0,
        0,
        0.0,
        0.0,
        0,
        0.0,
        0.0,
        elapsed_s,
        0,
        0,
    )


def _fallback_evaluated(
    course: Course,
    comparison: Comparison,
    config: PlannerConfig,
    label: str = "現在4.471秒Frenetフォールバック",
) -> EvaluatedGlobalPath:
    path = global_path_from_local(course, comparison, config, label=label)
    return evaluate_global_path(
        course,
        path,
        config,
        comparison.original.metrics.length_m,
        None,
        enforce_line_rule=False,
        enforce_board=False,
    )


def _unconfirmed_result(
    mode: str,
    fallback: EvaluatedGlobalPath,
    status: str,
) -> GlobalSearchResult:
    gated_path = replace(
        fallback.path,
        label=f"{mode}: LN5実車外形未設定のため未探索",
    )
    gated = replace(fallback, path=gated_path)
    return GlobalSearchResult(
        mode,
        np.array([0, gated.path.x_mm.size - 1], dtype=np.int32),
        gated,
        gated,
        _zero_stats(mode),
        True,
        None,
        False,
        False,
        status,
    )


def _nearest_contact_progress(
    target_distance_mm: np.ndarray,
    contact: ContactEvaluation,
) -> np.ndarray:
    nearest = np.searchsorted(contact.pose_distance_mm, target_distance_mm, side="left")
    nearest = np.clip(nearest, 0, contact.pose_distance_mm.size - 1)
    previous = np.maximum(nearest - 1, 0)
    choose_previous = (
        np.abs(target_distance_mm - contact.pose_distance_mm[previous])
        <= np.abs(target_distance_mm - contact.pose_distance_mm[nearest])
    )
    nearest[choose_previous] = previous[choose_previous]
    return contact.source_progress_index[nearest].astype(np.float32)


def _legalize_edge(
    edge: _ShortcutEdge,
    evaluator: WhiteLineContactEvaluator,
    course: Course,
    config: PlannerConfig,
) -> _ShortcutEdge:
    yaw, _, _ = _geometry_arrays(edge.x_mm, edge.y_mm)
    contact = evaluator.evaluate(
        edge.x_mm,
        edge.y_mm,
        yaw,
        start_segment=min(edge.start_index, course.point_count - 2),
        end_segment=min(edge.end_index, course.point_count - 2),
    )
    if not contact.legal:
        return replace(
            edge,
            reference_valid=False,
            reference_violation=contact.violation,
        )
    edge_distance_mm = cumulative_distance_m(edge.x_mm, edge.y_mm) * 1000.0
    progress = _nearest_contact_progress(edge_distance_mm, contact)
    risk_penalty = (
        edge.crossing_count * config.legal_crossing_risk_penalty_s
        + edge.shallow_crossing_count * config.legal_shallow_crossing_risk_penalty_s
        + edge.parallel_distance_mm
        * 0.001
        * config.legal_parallel_risk_penalty_s_per_m
    )
    return replace(
        edge,
        source_progress_index=progress,
        approximate_time_s=edge.approximate_time_s + risk_penalty,
        reference_valid=True,
        reference_violation="",
    )


def _evaluate_legal_path(
    course: Course,
    path,
    evaluator: WhiteLineContactEvaluator,
    config: PlannerConfig,
    original_length_m: float,
) -> tuple[EvaluatedGlobalPath | None, ContactEvaluation]:
    contact = evaluator.evaluate(
        path.x_mm,
        path.y_mm,
        path.yaw_rad,
        start_segment=0,
        end_segment=course.point_count - 2,
    )
    if not contact.legal:
        return None, contact
    contacted_path = apply_contact_progress_to_path(path, contact, course)
    evaluated = evaluate_global_path(
        course,
        contacted_path,
        config,
        original_length_m,
        None,
        enforce_line_rule=False,
        enforce_board=False,
    )
    if not evaluated.metrics.valid:
        return None, contact
    warning = "、".join(
        item for item in (evaluated.metrics.warning, contact.warning) if item
    )
    evaluated = replace(
        evaluated,
        metrics=replace(evaluated.metrics, warning=warning),
    )
    return evaluated, contact


def _legal_rank(
    evaluated: EvaluatedGlobalPath,
    contact: ContactEvaluation,
) -> tuple[float, ...]:
    metrics = evaluated.metrics
    shallow = metrics.shallow_line_crossing_count > 0
    sensor_robust = contact.robust and not shallow and metrics.parallel_line_distance_m < 0.05
    risk = (
        metrics.line_crossing_count
        + 10.0 * metrics.shallow_line_crossing_count
        + metrics.parallel_line_distance_m
    )
    return (
        0.0 if sensor_robust else 1.0,
        metrics.predicted_time_s,
        metrics.length_m,
        metrics.max_curvature_slew_per_m2,
        risk,
    )


def run_legal_global_mode(
    course: Course,
    mode: str,
    config: PlannerConfig | None = None,
    *,
    local_comparison: Comparison | None = None,
    footprint: VehicleFootprint | None = None,
    boundary: BoardBoundary | None = None,
) -> tuple[Comparison, GlobalSearchResult, str]:
    """実車外形確認をゲートにし、合法primitiveだけでDAGを作る。"""

    if mode not in {"reference", "embedded-lite"}:
        raise ValueError(f"未対応モードです: {mode}")
    config = config or PlannerConfig()
    comparison = local_comparison or run_comparison(course, config)
    boundary = load_board_boundary(course) if boundary is None else boundary
    footprint = footprint or load_vehicle_footprint()
    fallback = _fallback_evaluated(course, comparison, config)
    board_status = (
        f"CAD板境界確認済み（{boundary.source}）"
        if boundary is not None and boundary.confirmed
        else "板境界未確認"
    )
    if not footprint.confirmed:
        result = _unconfirmed_result(
            mode,
            fallback,
            "LN5実車外形未設定。大域経路は採用せず4.471秒へフォールバック",
        )
        return comparison, result, board_status

    total_start = perf_counter()
    evaluator = WhiteLineContactEvaluator(course, footprint, config, boundary)
    fallback_path, fallback_contact = _evaluate_legal_path(
        course,
        fallback.path,
        evaluator,
        config,
        comparison.original.metrics.length_m,
    )
    if fallback_path is None:
        fallback_path = fallback
    anchors = extract_anchor_indices(
        course,
        comparison.best.path.x_mm,
        comparison.best.path.y_mm,
        config,
        mode,
    )
    raw_edges, generation_s, considered = _make_edges(
        course,
        comparison,
        anchors,
        config,
        mode,
        boundary,
        apply_theoretical_line_tube=False,
    )
    legality_start = perf_counter()
    legal_edges = [
        legalized
        for edge in raw_edges
        if (legalized := _legalize_edge(edge, evaluator, course, config)).reference_valid
    ]
    baseline_plan = plan_speed(
        comparison.best.path.x_mm, comparison.best.path.y_mm, config
    )
    segment_time = 2.0 * np.diff(baseline_plan.distance_m) / np.maximum(
        baseline_plan.speed_mps[:-1] + baseline_plan.speed_mps[1:], 1.0e-6
    )
    baseline_time = np.concatenate((np.zeros(1), np.cumsum(segment_time)))
    legal_base_edges: list[_ShortcutEdge] = []
    for node in range(anchors.size - 1):
        base = _base_edge(
            node,
            anchors,
            comparison.best.path.x_mm,
            comparison.best.path.y_mm,
            baseline_plan.distance_m,
            baseline_time,
        )
        legalized = _legalize_edge(base, evaluator, course, config)
        if legalized.reference_valid:
            legal_base_edges.append(legalized)
    legality_s = perf_counter() - legality_start
    top_k = config.reference_top_k if mode == "reference" else config.embedded_top_k
    sequences, graph_s = _k_best_edge_paths(
        anchors,
        legal_edges,
        legal_base_edges,
        top_k,
        reference_only=True,
    )
    edge_map = {edge.edge_id: edge for edge in legal_base_edges + legal_edges}
    evaluation_start = perf_counter()
    best = fallback_path
    best_contact = fallback_contact if fallback_path is not fallback else None
    evaluated_count = 0
    for sequence in sequences:
        path = _build_global_path(
            course,
            sequence,
            edge_map,
            config,
            f"LN5実車合法{mode}経路",
            perf_counter() - total_start,
            allow_synthetic_progress=False,
        )
        evaluated, contact = _evaluate_legal_path(
            course,
            path,
            evaluator,
            config,
            comparison.original.metrics.length_m,
        )
        evaluated_count += 1
        if evaluated is None:
            continue
        if best_contact is None or _legal_rank(evaluated, contact) < _legal_rank(best, best_contact):
            best = evaluated
            best_contact = contact
    evaluation_s = perf_counter() - evaluation_start
    legal = best_contact is not None and best_contact.legal
    robust = best_contact is not None and best_contact.robust
    fallback_used = best.path.selected_edges == ()
    max_points = max((edge.x_mm.size for edge in raw_edges), default=0)
    memory = int(
        anchors.nbytes
        + sum(
            edge.x_mm.nbytes + edge.y_mm.nbytes + edge.source_progress_index.nbytes
            for edge in raw_edges
        )
        + max_points * 8 * 6
    )
    stats = GlobalSearchStats(
        mode,
        int(anchors.size),
        len(raw_edges),
        len(legal_edges),
        generation_s + legality_s,
        graph_s,
        evaluated_count,
        evaluation_s,
        0.0,
        perf_counter() - total_start,
        considered * 4 + evaluated_count * (config.speed_scan_iterations * 2 + 8),
        memory,
    )
    status = (
        "LN5実車外形で全swept姿勢が白線接触"
        if legal
        else "LN5実車外形で合法な完整経路なし"
    )
    result = GlobalSearchResult(
        mode,
        anchors,
        best,
        best,
        stats,
        fallback_used,
        best_contact,
        legal,
        robust,
        status,
    )
    return comparison, result, board_status


def run_legal_comparison(
    course: Course,
    config: PlannerConfig | None = None,
    *,
    local_comparison: Comparison | None = None,
) -> GlobalComparison:
    """理論下限とLN5実車合法経路を明確に分離する。"""

    config = config or PlannerConfig()
    comparison = local_comparison or run_comparison(course, config)
    boundary = load_board_boundary(course)
    footprint = load_vehicle_footprint()
    fallback = _fallback_evaluated(course, comparison, config)
    lower = _run_geometric_lower_bound(
        course, comparison, config, boundary, fallback
    )
    theoretical = _run_mode(
        course, comparison, config, "reference", boundary, fallback
    )
    theoretical_path = replace(
        theoretical.adopted.path,
        label="規定最大車体による非実車・非合法性確認の幾何下限",
    )
    theoretical_eval = replace(theoretical.adopted, path=theoretical_path)
    theoretical = replace(
        theoretical,
        best_global=theoretical_eval,
        adopted=theoretical_eval,
        legal=False,
        robust=False,
        legality_status="競技無効: 規定最大直径250mm仮想円だけで判定",
    )
    if footprint.confirmed:
        theoretical_evaluator = WhiteLineContactEvaluator(
            course, footprint, config, boundary
        )
        theoretical_contact = theoretical_evaluator.evaluate(
            theoretical_eval.path.x_mm,
            theoretical_eval.path.y_mm,
            theoretical_eval.path.yaw_rad,
            start_segment=0,
            end_segment=course.point_count - 2,
        )
        theoretical = replace(theoretical, contact=theoretical_contact)
    _, reference, board_status = run_legal_global_mode(
        course,
        "reference",
        config,
        local_comparison=comparison,
        footprint=footprint,
        boundary=boundary,
    )
    _, embedded, _ = run_legal_global_mode(
        course,
        "embedded-lite",
        config,
        local_comparison=comparison,
        footprint=footprint,
        boundary=boundary,
    )
    legal_candidates = [
        result.adopted
        for result in (reference, embedded)
        if result.legal
    ]
    final = (
        min(legal_candidates, key=lambda item: item.metrics.predicted_time_s)
        if legal_candidates
        else fallback
    )
    footprint_status = (
        f"LN5実車外形確認済み: {footprint.source}"
        if footprint.confirmed
        else "LN5実車外形未設定。大域経路は競技有効値へ採用しない"
    )
    return GlobalComparison(
        comparison,
        fallback,
        lower,
        theoretical,
        reference,
        embedded,
        final,
        boundary,
        board_status,
        footprint,
        footprint_status,
    )
