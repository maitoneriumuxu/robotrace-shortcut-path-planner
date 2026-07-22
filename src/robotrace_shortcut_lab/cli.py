from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from .course import load_course
from .deep_planner import run_deep_reference
from .global_planner import run_global_comparison
from .legal_planner import run_legal_global_mode
from .model import BatchCourseResult, COURSE_FILE, PlannerConfig
from .report import write_all_courses_png, write_deep_result_png, write_global_result_png


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="保存済みロボトレース点列の大域最短時間経路を比較します"
    )
    parser.add_argument(
        "--course",
        type=Path,
        default=COURSE_FILE,
        help="data/courses/normalized配下の大会TSV",
    )
    parser.add_argument(
        "--mode",
        choices=("reference", "embedded-lite", "deep-reference"),
        default="reference",
        help="探索規模。単一コース画像では差を示すため両方を計算します",
    )
    parser.add_argument(
        "--deep-budget-seconds",
        type=float,
        default=1_500.0,
        help="deep-referenceの総計算時間上限（既定1500秒）",
    )
    parser.add_argument(
        "--all-courses",
        action="store_true",
        help="normalized配下の全大会を指定モードで回帰確認します",
    )
    return parser


def _run_all_courses(mode: str, config: PlannerConfig) -> int:
    results: list[BatchCourseResult] = []
    for course_path in sorted(Path("data/courses/normalized").glob("*.tsv")):
        start = perf_counter()
        try:
            course = load_course(course_path)
            local, global_result, board_status = run_legal_global_mode(
                course, mode, config, evaluate_robust=False
            )
            baseline_time = local.best.metrics.predicted_time_s
            selected = global_result.adopted
            selected_time = selected.metrics.predicted_time_s
            fallback = global_result.fallback_used
            status = "フォールバック" if fallback else "改善"
            if not global_result.legal:
                status += "・設計legal経路なし"
            elif global_result.sensitivity is None:
                status += "・全LINE通過legal（robust未評価）"
            elif not global_result.robust:
                status += "・設計legal（robust警告）"
            else:
                status += "・設計legal/robust"
            if board_status == "板境界未確認":
                status += "・板境界未確認"
            if not selected.metrics.valid:
                status = f"不合格: {selected.metrics.violation}"
            result = BatchCourseResult(
                course.course_id,
                baseline_time,
                selected_time,
                selected_time - baseline_time,
                mode,
                global_result.stats.anchor_count,
                global_result.stats.candidate_edge_count,
                perf_counter() - start,
                fallback,
                selected.metrics.valid,
                status,
            )
        except Exception as error:  # 全大会回帰では失敗コースも画像へ残す。
            result = BatchCourseResult(
                course_path.stem,
                0.0,
                0.0,
                0.0,
                mode,
                0,
                0,
                perf_counter() - start,
                True,
                False,
                f"異常終了: {type(error).__name__}",
            )
        results.append(result)
        print(
            f"{result.course_id}: {result.baseline_time_s:.3f}s -> "
            f"{result.selected_time_s:.3f}s ({result.improvement_s:+.3f}s), "
            f"{result.status}, {result.total_s:.2f}s"
        )
    output = write_all_courses_png(Path("outputs/all_courses.png"), results)
    improved = sum(
        item.improvement_s < -1.0e-6
        and item.valid
        and not item.fallback_used
        for item in results
    )
    fallback = sum(item.fallback_used and item.valid for item in results)
    invalid = sum(not item.valid for item in results)
    print(f"改善={improved}, フォールバック={fallback}, 不合格={invalid}")
    print(output.resolve())
    return 1 if invalid else 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = PlannerConfig()
    if args.all_courses:
        return _run_all_courses(args.mode, config)

    course = load_course(args.course)
    if args.mode == "deep-reference":
        result = run_deep_reference(
            course,
            config,
            budget_s=args.deep_budget_seconds,
            progress=print,
        )
        output = write_deep_result_png(
            Path("outputs/deep_result.png"), course, result, config
        )
        print(f"deep初期合法: {result.current.metrics.predicted_time_s:.6f}s")
        for stage in result.stage_records:
            print(
                f"{stage.name}: anchors={stage.anchor_count}, "
                f"edges={stage.generated_edge_count}, checked={stage.screened_edge_count}, "
                f"legal={stage.legal_edge_count}, K={stage.top_k_count}, "
                f"full={stage.full_path_count}, best={stage.best_time_s:.6f}s, "
                f"elapsed={stage.elapsed_s:.1f}s"
            )
        print(
            f"legal幾何下限: {result.geometric_lower_bound.metrics.predicted_time_s:.6f}s"
        )
        print(f"deep legal最速: {result.legal_best.metrics.predicted_time_s:.6f}s")
        print(f"deep robust最速: {result.robust_best.metrics.predicted_time_s:.6f}s")
        if result.strong_robust_best is not None:
            print(
                "±5mm/±2deg最速: "
                f"{result.strong_robust_best.metrics.predicted_time_s:.6f}s"
            )
        print(
            f"評価候補={result.evaluated_candidate_count}, 総時間={result.total_s:.1f}s"
        )
        print(output.resolve())
        return 0
    comparison = run_global_comparison(course, config)
    output = write_global_result_png(
        Path("outputs/result.png"), comparison, config
    )
    print(
        f"現在基準: {comparison.current_baseline.metrics.predicted_time_s:.3f}s, "
        f"{comparison.current_baseline.metrics.length_m:.3f}m"
    )
    print(
        f"幾何下限: {comparison.geometric_lower_bound.metrics.predicted_time_s:.3f}s, "
        f"{comparison.geometric_lower_bound.metrics.length_m:.3f}m"
    )
    theoretical = comparison.maximum_vehicle_lower_bound.adopted.metrics
    print(
        f"規定最大車体の非合法理論下限: "
        f"{theoretical.predicted_time_s:.3f}s, {theoretical.length_m:.3f}m, "
        "競技有効経路には不採用"
    )
    for result in (comparison.reference, comparison.embedded_lite):
        metrics = result.adopted.metrics
        stats = result.stats
        print(
            f"{result.mode}: {metrics.predicted_time_s:.3f}s, {metrics.length_m:.3f}m, "
            f"shortcut={metrics.shortcut_edge_count}, anchors={stats.anchor_count}, "
            f"candidates={stats.candidate_edge_count}, valid={stats.valid_edge_count}, "
            f"topK={stats.top_k_count}, total={stats.total_s:.2f}s, "
            f"fallback={result.fallback_used}, legal={result.legal}, "
            f"status={result.legality_status}"
        )
    final = comparison.final.metrics
    print(
        f"最終採用: {final.predicted_time_s:.3f}s "
        f"(4.000秒差 {final.predicted_time_s - 4.0:+.3f}s), "
        f"{final.length_m:.3f}m, {final.shortcut_edge_count}辺"
    )
    print(comparison.board_status)
    print(comparison.footprint_status)
    if comparison.robust_final is not None:
        print(
            f"robust最良: {comparison.robust_final.metrics.predicted_time_s:.3f}s"
        )
    print(
        "contact witness: 10x200mm横バー, design_confirmed="
        f"{comparison.vehicle_footprint.design_confirmed}, as_built_confirmed="
        f"{comparison.vehicle_footprint.as_built_confirmed}"
    )
    final_contact = (
        comparison.reference.contact
        if comparison.final.path.selected_edges
        == comparison.reference.adopted.path.selected_edges
        else comparison.embedded_lite.contact
    )
    if final_contact is not None:
        print(
            "全LINE通過: "
            f"{final_contact.all_line_segments_covered}, "
            f"未通過segment={final_contact.unvisited_segment_count}"
        )
    print(output.resolve())
    return 0
