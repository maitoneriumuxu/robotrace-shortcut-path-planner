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
    evaluate_contact_sensitivity,
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
    candidate=None,
) -> EvaluatedGlobalPath:
    selected_comparison = (
        comparison if candidate is None else replace(comparison, best=candidate)
    )
    path = global_path_from_local(course, selected_comparison, config, label=label)
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
        label=f"{mode}: 接触証明部品が未確認のため未探索",
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
        None,
        None,
        None,
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
    unreachable = evaluator.count_radially_unreachable_segments(
        edge.x_mm,
        edge.y_mm,
        min(edge.start_index, course.point_count - 2),
        min(edge.end_index, course.point_count - 2),
    )
    if unreachable:
        return replace(
            edge,
            reference_valid=False,
            reference_violation=f"未通過LINE segment {unreachable}区間（外接円でも到達不能）",
        )
    yaw, _, _ = _geometry_arrays(edge.x_mm, edge.y_mm)
    contact = evaluator.evaluate(
        edge.x_mm,
        edge.y_mm,
        yaw,
        start_segment=min(edge.start_index, course.point_count - 2),
        end_segment=min(edge.end_index, course.point_count - 2),
        endpoint_tolerance_segments=28,
        compute_metrics=False,
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
    *,
    compute_contact_metrics: bool = True,
) -> tuple[EvaluatedGlobalPath | None, ContactEvaluation]:
    contact = evaluator.evaluate(
        path.x_mm,
        path.y_mm,
        path.yaw_rad,
        start_segment=0,
        end_segment=course.point_count - 2,
        compute_metrics=compute_contact_metrics,
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


def run_legal_global_mode(
    course: Course,
    mode: str,
    config: PlannerConfig | None = None,
    *,
    local_comparison: Comparison | None = None,
    footprint: VehicleFootprint | None = None,
    boundary: BoardBoundary | None = None,
    evaluate_robust: bool = True,
) -> tuple[Comparison, GlobalSearchResult, str]:
    """設計確認済み接触証明部品をゲートにし、合法primitiveだけでDAGを作る。"""

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
    if not footprint.design_confirmed:
        result = _unconfirmed_result(
            mode,
            fallback,
            "接触証明部品が未確認。大域経路は採用せず4.471秒へフォールバック",
        )
        return comparison, result, board_status

    total_start = perf_counter()
    evaluator = WhiteLineContactEvaluator(course, footprint, config, boundary)
    fallback_path = None
    fallback_contact = None
    selected_local_fallback = None
    local_fallbacks = sorted(
        (
            comparison.best,
            comparison.legacy_time,
            comparison.elastic,
            comparison.original,
        ),
        key=lambda item: (item.metrics.predicted_time_s, item.path.label),
    )
    seen_paths: set[bytes] = set()
    for local_candidate in local_fallbacks:
        fingerprint = local_candidate.path.x_mm.tobytes() + local_candidate.path.y_mm.tobytes()
        if fingerprint in seen_paths:
            continue
        seen_paths.add(fingerprint)
        candidate_fallback = _fallback_evaluated(
            course,
            comparison,
            config,
            label=f"合法fallback: {local_candidate.path.label}",
            candidate=local_candidate,
        )
        evaluated_fallback, contact_fallback = _evaluate_legal_path(
            course,
            candidate_fallback.path,
            evaluator,
            config,
            comparison.original.metrics.length_m,
            compute_contact_metrics=False,
        )
        if evaluated_fallback is None:
            continue
        fallback = candidate_fallback
        fallback_path = evaluated_fallback
        fallback_contact = contact_fallback
        selected_local_fallback = local_candidate
        break
    if selected_local_fallback is None:
        selected_local_fallback = comparison.best
    search_comparison = replace(comparison, best=selected_local_fallback)
    anchors = extract_anchor_indices(
        course,
        selected_local_fallback.path.x_mm,
        selected_local_fallback.path.y_mm,
        config,
        mode,
    )
    raw_edges, generation_s, considered = _make_edges(
        course,
        search_comparison,
        anchors,
        config,
        mode,
        boundary,
        apply_theoretical_line_tube=False,
        prefilter_line_radius_mm=(
            max(
                float(np.max(np.hypot(component[:, 0], component[:, 1])))
                for component in footprint.contact_witness_components_mm
            )
            + config.white_line_half_width_mm
        ),
    )
    baseline_plan = plan_speed(
        selected_local_fallback.path.x_mm, selected_local_fallback.path.y_mm, config
    )
    segment_time = 2.0 * np.diff(baseline_plan.distance_m) / np.maximum(
        baseline_plan.speed_mps[:-1] + baseline_plan.speed_mps[1:], 1.0e-6
    )
    baseline_time = np.concatenate((np.zeros(1), np.cumsum(segment_time)))
    edge_check_candidates = [edge for edge in raw_edges if edge.reference_valid]
    edge_check_candidates.sort(
        key=lambda edge: (
            -(
                float(baseline_time[edge.end_index] - baseline_time[edge.start_index])
                - edge.approximate_time_s
            ),
            edge.start_index,
            edge.end_index,
            edge.edge_id,
        )
    )
    edge_check_limit = (
        config.legal_reference_edge_check_limit
        if mode == "reference"
        else config.legal_embedded_edge_check_limit
    )
    edge_check_candidates = edge_check_candidates[:edge_check_limit]
    legality_start = perf_counter()
    legal_edges = [
        legalized
        for edge in edge_check_candidates
        if (legalized := _legalize_edge(edge, evaluator, course, config)).reference_valid
    ]
    legal_base_edges: list[_ShortcutEdge] = []
    for node in range(anchors.size - 1):
        base = _base_edge(
            node,
            anchors,
            selected_local_fallback.path.x_mm,
            selected_local_fallback.path.y_mm,
            baseline_plan.distance_m,
            baseline_time,
        )
        # 基準辺は、直前に完全経路として同じ合法性ゲートを通したfallbackの
        # 部分列。候補ごとの重い再検査を省き、結合後の完全経路で再確認する。
        legal_base_edges.append(base)
    legality_s = perf_counter() - legality_start
    top_k = (
        config.legal_reference_top_k
        if mode == "reference"
        else config.embedded_top_k
    )
    sequences, graph_s = _k_best_edge_paths(
        anchors,
        legal_edges,
        legal_base_edges,
        top_k,
        reference_only=True,
    )
    edge_map = {edge.edge_id: edge for edge in legal_base_edges + legal_edges}
    evaluation_start = perf_counter()
    previews: list[tuple[EvaluatedGlobalPath, ContactEvaluation | None]] = []
    if fallback_path is not None and fallback_contact is not None and fallback_contact.legal:
        previews.append((fallback_path, fallback_contact))
    for sequence in sequences:
        if all(edge_id < 0 for edge_id in sequence):
            continue
        path = _build_global_path(
            course,
            sequence,
            edge_map,
            config,
            f"LN5実車合法{mode}経路",
            perf_counter() - total_start,
            allow_synthetic_progress=False,
        )
        preview = evaluate_global_path(
            course,
            path,
            config,
            comparison.original.metrics.length_m,
            None,
            enforce_line_rule=False,
            enforce_board=False,
        )
        if preview.metrics.valid:
            previews.append((preview, None))
    previews.sort(
        key=lambda item: (
            item[0].metrics.predicted_time_s,
            item[0].metrics.length_m,
            item[0].path.selected_edges,
        )
    )
    best, best_contact = fallback, None
    best_sensitivity = None
    robust_best = None
    robust_best_contact = None
    robust_best_sensitivity = None
    evaluated_count = 0
    for preview, cached_contact in previews:
        if cached_contact is None:
            evaluated, preliminary_contact = _evaluate_legal_path(
                course,
                preview.path,
                evaluator,
                config,
                comparison.original.metrics.length_m,
                compute_contact_metrics=False,
            )
            evaluated_count += 1
            if evaluated is None:
                continue
        else:
            evaluated, preliminary_contact = preview, cached_contact
        if not evaluate_robust:
            best, best_contact = evaluated, preliminary_contact
            break
        detailed_evaluated, detailed_contact = _evaluate_legal_path(
            course,
            evaluated.path,
            evaluator,
            config,
            comparison.original.metrics.length_m,
            compute_contact_metrics=True,
        )
        if detailed_evaluated is None:
            continue
        sensitivity = evaluate_contact_sensitivity(
            evaluator,
            evaluated.path.x_mm,
            evaluated.path.y_mm,
            evaluated.path.yaw_rad,
            config,
        )
        if best_contact is None:
            best, best_contact = detailed_evaluated, detailed_contact
            best_sensitivity = sensitivity
        if (
            detailed_contact.robust
            and sensitivity.robust_2mm_1deg
        ):
            robust_best = detailed_evaluated
            robust_best_contact = detailed_contact
            robust_best_sensitivity = sensitivity
            break
    evaluation_s = perf_counter() - evaluation_start
    legal = best_contact is not None and best_contact.legal
    robust = bool(
        best_contact is not None
        and best_contact.robust
        and best_sensitivity is not None
        and best_sensitivity.robust_2mm_1deg
    )
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
        len(sequences),
        evaluation_s,
        0.0,
        perf_counter() - total_start,
        considered * 4 + len(sequences) * (config.speed_scan_iterations * 2 + 8),
        memory,
    )
    status = (
        "200mm横バーが全swept姿勢で接触し、全LINEを順番に通過（設計上合法）"
        if legal
        else "200mm横バー接触＋全LINE通過を満たす完全経路なし"
    )
    result = GlobalSearchResult(
        mode,
        anchors,
        best,
        best,
        stats,
        fallback_used,
        best_contact,
        best_sensitivity,
        robust_best,
        robust_best_contact,
        robust_best_sensitivity,
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
    if footprint.design_confirmed:
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
    if embedded.legal and (
        not reference.legal
        or embedded.adopted.metrics.predicted_time_s
        < reference.adopted.metrics.predicted_time_s
    ):
        reference = replace(
            reference,
            best_global=embedded.adopted,
            adopted=embedded.adopted,
            fallback_used=embedded.fallback_used,
            contact=embedded.contact,
            sensitivity=embedded.sensitivity,
            robust_best=(
                embedded.robust_best
                if reference.robust_best is None
                or (
                    embedded.robust_best is not None
                    and embedded.robust_best.metrics.predicted_time_s
                    < reference.robust_best.metrics.predicted_time_s
                )
                else reference.robust_best
            ),
            robust_best_contact=(
                embedded.robust_best_contact
                if embedded.robust_best is not None
                and (
                    reference.robust_best is None
                    or embedded.robust_best.metrics.predicted_time_s
                    < reference.robust_best.metrics.predicted_time_s
                )
                else reference.robust_best_contact
            ),
            robust_best_sensitivity=(
                embedded.robust_best_sensitivity
                if embedded.robust_best is not None
                and (
                    reference.robust_best is None
                    or embedded.robust_best.metrics.predicted_time_s
                    < reference.robust_best.metrics.predicted_time_s
                )
                else reference.robust_best_sensitivity
            ),
            legal=True,
            robust=embedded.robust,
            legality_status=(
                "reference候補集合へembedded-lite最良を統合。"
                + embedded.legality_status
            ),
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
    robust_candidates = []
    for result in (reference, embedded):
        if (
            result.robust_best is None
            or result.robust_best_contact is None
            or result.robust_best_sensitivity is None
        ):
            continue
        robust_candidates.append(
            (
                result.robust_best,
                result.robust_best_contact,
                result.robust_best_sensitivity,
            )
        )
    robust_candidates.sort(key=lambda item: item[0].metrics.predicted_time_s)
    if robust_candidates:
        robust_final, robust_final_contact, robust_final_sensitivity = robust_candidates[0]
    else:
        robust_final = robust_final_contact = robust_final_sensitivity = None
    footprint_status = (
        "200mm横バー接触＋全LINE通過で設計上合法を評価、実車製作後確認は未完了"
        if footprint.design_confirmed and not footprint.as_built_confirmed
        else "接触証明部品が未設定"
    )
    return GlobalComparison(
        comparison,
        fallback,
        lower,
        theoretical,
        reference,
        embedded,
        final,
        robust_final,
        robust_final_contact,
        robust_final_sensitivity,
        boundary,
        board_status,
        footprint,
        footprint_status,
    )
