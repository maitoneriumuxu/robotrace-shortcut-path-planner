from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
import numpy as np

from .model import (
    BatchCourseResult,
    Comparison,
    EvaluatedGlobalPath,
    GlobalComparison,
    PlannerConfig,
)


COLORS = ("#6B7280", "#2563EB", "#D97706", "#059669")
LIMIT_COLORS = ("#7C3AED", "#DC2626", "#F59E0B", "#2563EB", "#0891B2")


def _set_japanese_font() -> None:
    candidates = (
        Path(r"C:\Windows\Fonts\meiryo.ttc"),
        Path(r"C:\Windows\Fonts\YuGothM.ttc"),
    )
    for candidate in candidates:
        if candidate.exists():
            name = font_manager.FontProperties(fname=str(candidate)).get_name()
            plt.rcParams["font.family"] = name
            return


def _format_rows(comparison: Comparison) -> tuple[list[str], list[list[str]]]:
    paths = (
        comparison.original,
        comparison.elastic,
        comparison.legacy_time,
        comparison.best,
    )
    elastic_time = comparison.elastic.metrics.predicted_time_s
    labels = [
        "予測走行時間 [s]",
        "Elastic差 [s]",
        "経路長 [m]",
        "経路短縮率 [%]",
        "最大横オフセット [mm]",
        "最小半径 [mm]",
        "最大 |dκ/ds| [1/m²]",
        "最大速度 [m/s]",
        "候補生成時間 [ms]",
        "制約判定",
        "採用方式",
    ]
    rows: list[list[str]] = []
    for item in paths:
        metrics = item.metrics
        rows.append(
            [
                f"{metrics.predicted_time_s:.3f}",
                f"{metrics.predicted_time_s - elastic_time:+.3f}",
                f"{metrics.length_m:.3f}",
                f"{metrics.shortening_percent:.2f}",
                f"{metrics.max_offset_mm:.1f}",
                f"{metrics.min_radius_mm:.1f}",
                f"{metrics.max_curvature_slew_per_m2:.1f}",
                f"{metrics.max_speed_mps:.1f}",
                f"{item.path.generation_s * 1000.0:.1f}",
                "合格" if metrics.valid else f"違反: {metrics.violation}",
                "採用" if item is comparison.best else "比較候補",
            ]
        )
    return labels, rows


def write_result_png(
    destination: str | Path,
    comparison: Comparison,
    config: PlannerConfig,
) -> Path:
    """全体・ヘアピン拡大・比較表・速度プロファイルを日本語PNGへまとめる。"""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    _set_japanese_font()
    plt.rcParams["axes.unicode_minus"] = False

    figure = plt.figure(figsize=(18, 10), dpi=150, facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.18, 1.0),
        height_ratios=(1.08, 0.92),
        wspace=0.18,
        hspace=0.24,
    )
    map_axis = figure.add_subplot(grid[0, 0])
    zoom_axis = figure.add_subplot(grid[1, 0])
    table_axis = figure.add_subplot(grid[0, 1])
    speed_axis = figure.add_subplot(grid[1, 1])

    paths = (
        comparison.original,
        comparison.elastic,
        comparison.legacy_time,
        comparison.best,
    )
    styles = (("-", 1.25), ("-", 1.25), ("--", 1.25), ("-", 1.35))
    short_labels = ("原コース", "Elastic Band", "旧時間選択型#7", "改善後最良")
    for item, short_label, color, (line_style, width) in zip(
        paths, short_labels, COLORS, styles, strict=True
    ):
        for axis in (map_axis, zoom_axis):
            axis.plot(
                item.path.x_mm,
                item.path.y_mm,
                color=color,
                linestyle=line_style,
                linewidth=width,
                label=short_label,
                alpha=0.94,
            )
    map_axis.set_title("コース全体 ― 4経路比較", fontsize=13)
    map_axis.set_xlabel("X [mm]")
    map_axis.set_ylabel("Y [mm]")
    map_axis.set_aspect("equal", adjustable="box")
    map_axis.grid(True, color="#D1D5DB", linewidth=0.55, alpha=0.7)
    map_axis.legend(loc="upper right", framealpha=0.92, fontsize=7.3)

    if comparison.window_center_indices:
        start = max(0, min(comparison.window_center_indices) - 220)
        finish = min(
            comparison.original.path.x_mm.size,
            max(comparison.window_center_indices) + 91,
        )
        zoom_x = np.concatenate([item.path.x_mm[start:finish] for item in paths])
        zoom_y = np.concatenate([item.path.y_mm[start:finish] for item in paths])
        padding_x = max(40.0, 0.08 * float(np.ptp(zoom_x)))
        padding_y = max(40.0, 0.08 * float(np.ptp(zoom_y)))
        zoom_axis.set_xlim(float(np.min(zoom_x)) - padding_x, float(np.max(zoom_x)) + padding_x)
        zoom_axis.set_ylim(float(np.min(zoom_y)) - padding_y, float(np.max(zoom_y)) + padding_y)
    zoom_axis.set_title("自動検出した中央の串状ヘアピン拡大", fontsize=12)
    zoom_axis.set_xlabel("X [mm]")
    zoom_axis.set_ylabel("Y [mm]")
    zoom_axis.set_aspect("equal", adjustable="box")
    zoom_axis.grid(True, color="#D1D5DB", linewidth=0.55, alpha=0.7)
    zoom_axis.legend(loc="upper right", framealpha=0.92, fontsize=7.2)

    labels, rows = _format_rows(comparison)
    table_axis.axis("off")
    table_axis.set_title("ATTACK速度モデルによる比較", fontsize=13, pad=10)
    cell_text = [
        [labels[index], rows[0][index], rows[1][index], rows[2][index], rows[3][index]]
        for index in range(len(labels))
    ]
    table = table_axis.table(
        cellText=cell_text,
        colLabels=("評価項目", "原コース", "Elastic", "旧#7", "改善後"),
        cellLoc="right",
        colLoc="center",
        bbox=(0.0, 0.10, 1.0, 0.84),
        colWidths=(0.34, 0.155, 0.16, 0.155, 0.19),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.18)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_facecolor("#E5E7EB")
        elif column == 0:
            cell.set_facecolor("#F3F4F6")
            cell.set_text_props(ha="left")
    table_axis.text(
        0.0,
        0.0,
        f"採用: #{comparison.selected_candidate_id} {comparison.selected_candidate_name}\n"
        f"候補評価 {comparison.candidate_evaluation_count}本 / "
        f"自動検出中心 {comparison.window_center_indices}",
        transform=table_axis.transAxes,
        fontsize=8.5,
        color="#374151",
        va="bottom",
    )

    for item, short_label, color in zip(paths, short_labels, COLORS, strict=True):
        speed_axis.plot(
            item.distance_m,
            item.speed_mps,
            color=color,
            linewidth=1.35,
            label=short_label,
        )
    reason_y = config.min_speed_mps - 0.22
    speed_axis.scatter(
        comparison.best.distance_m,
        np.full_like(comparison.best.distance_m, reason_y),
        c=np.asarray(LIMIT_COLORS)[comparison.best.speed_limit_reason],
        s=3.0,
        linewidths=0.0,
        alpha=0.9,
    )
    speed_axis.set_title("速度プロファイル（下端色帯: 改善後の支配要因）", fontsize=11.5)
    speed_axis.set_xlabel("経路距離 [m]")
    speed_axis.set_ylabel("速度 [m/s]")
    speed_axis.set_ylim(config.min_speed_mps - 0.35, config.max_speed_mps + 0.45)
    speed_axis.grid(True, color="#D1D5DB", linewidth=0.55, alpha=0.7)
    path_legend = speed_axis.legend(loc="upper right", fontsize=7.2)
    speed_axis.add_artist(path_legend)
    limit_handles = [
        Line2D([0], [0], color=color, linewidth=4, label=label)
        for color, label in zip(
            LIMIT_COLORS,
            ("最高速度", "GFCP", "AALP", "加速", "減速"),
            strict=True,
        )
    ]
    speed_axis.legend(
        handles=limit_handles,
        loc="lower center",
        ncol=5,
        fontsize=7.0,
        framealpha=0.9,
    )

    figure.text(
        0.5,
        0.012,
        "注意: 過去ATTACKパラメータ3基準を調整（3.6～13.0 m/s、GFCP R10 3.6 m/s、指数0.33、加速20～55 m/s²、減速55 m/s²）。"
        "Python予測は実走可能性・実機性能・競技上の安全を保証しません。",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4B5563",
    )
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def _global_status(item: EvaluatedGlobalPath) -> str:
    if not item.metrics.valid:
        return "不合格"
    if item.metrics.warning:
        return "警告"
    return "合格"


def _time_on_source_progress(
    item: EvaluatedGlobalPath, source_distance_mm: np.ndarray
) -> np.ndarray:
    progress = item.path.source_progress_distance_mm.astype(np.float64)
    keep = np.concatenate(([True], np.diff(progress) > 1.0e-6))
    return np.interp(
        source_distance_mm,
        progress[keep],
        item.cumulative_time_s.astype(np.float64)[keep],
    )


def _write_unconstrained_global_result_png(
    destination: str | Path,
    comparison: GlobalComparison,
    config: PlannerConfig,
) -> Path:
    """大域経路、最大短縮区間、速度・累積時間、探索統計を1枚にする。"""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    _set_japanese_font()
    plt.rcParams["axes.unicode_minus"] = False

    local = comparison.local
    current = comparison.current_baseline
    reference = comparison.reference.adopted
    embedded = comparison.embedded_lite.adopted
    final = comparison.final
    items = (current, reference, embedded, final)
    labels = ("現在4.471秒経路", "reference大域", "embedded-lite", "最終採用")
    colors = ("#D97706", "#059669", "#2563EB", "#BE185D")

    figure = plt.figure(figsize=(20, 12), dpi=150, facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.18, 1.0),
        height_ratios=(1.02, 0.98),
        wspace=0.16,
        hspace=0.20,
    )
    map_axis = figure.add_subplot(grid[0, 0])
    zoom_axis = figure.add_subplot(grid[1, 0])
    table_axis = figure.add_subplot(grid[0, 1])
    profile_grid = grid[1, 1].subgridspec(2, 1, hspace=0.28)
    speed_axis = figure.add_subplot(profile_grid[0, 0])
    time_axis = figure.add_subplot(profile_grid[1, 0])

    map_axis.plot(
        local.original.path.x_mm,
        local.original.path.y_mm,
        color="#6B7280",
        linewidth=1.0,
        label="原コース",
        alpha=0.85,
    )
    styles = ("--", "-", "-.", "-")
    for item, label, color, style in zip(items, labels, colors, styles, strict=True):
        map_axis.plot(
            item.path.x_mm,
            item.path.y_mm,
            color=color,
            linestyle=style,
            linewidth=1.25 if item is not final else 1.55,
            label=f"{label} {item.metrics.predicted_time_s:.3f}s",
            alpha=0.92,
        )
    anchors = comparison.reference.anchor_indices
    map_axis.scatter(
        local.original.path.x_mm[anchors],
        local.original.path.y_mm[anchors],
        s=7,
        facecolors="none",
        edgecolors="#111827",
        linewidths=0.45,
        label="referenceアンカー",
        zorder=5,
    )
    shortcut_mask = reference.path.shortcut_edge_id > 0
    if np.any(shortcut_mask):
        map_axis.scatter(
            reference.path.x_mm[shortcut_mask],
            reference.path.y_mm[shortcut_mask],
            s=2.0,
            color="#10B981",
            label="採用ショートカット辺",
            zorder=6,
        )
    map_axis.set_title("2025年全日本 ― 大域最短時間経路探索", fontsize=13)
    map_axis.set_xlabel("X [mm]")
    map_axis.set_ylabel("Y [mm]")
    map_axis.set_aspect("equal", adjustable="box")
    map_axis.grid(True, color="#D1D5DB", linewidth=0.5, alpha=0.65)
    map_axis.legend(loc="upper right", fontsize=6.7, framealpha=0.92)

    zoom_source = final if final.path.selected_edges else reference
    if zoom_source.path.selected_edges:
        selected = max(
            zoom_source.path.selected_edges,
            key=lambda edge: float(
                local.original.distance_m[edge[2]] - local.original.distance_m[edge[1]]
            ),
        )
        _, start_index, end_index, kind = selected
    else:
        start_index, end_index, kind = 0, local.original.path.x_mm.size - 1, "なし"
    zoom_axis.plot(
        local.original.path.x_mm[start_index : end_index + 1],
        local.original.path.y_mm[start_index : end_index + 1],
        color="#6B7280",
        linewidth=1.0,
        linestyle=":",
        label="スキップした原コース",
    )
    for item, label, color, style in zip(
        (current, zoom_source),
        ("現在経路", "新経路"),
        (colors[0], "#059669"),
        ("--", "-"),
        strict=True,
    ):
        mask = (
            (item.path.source_progress_index >= start_index)
            & (item.path.source_progress_index <= end_index)
        )
        zoom_axis.plot(
            item.path.x_mm[mask],
            item.path.y_mm[mask],
            color=color,
            linestyle=style,
            linewidth=1.5,
            label=label,
        )
    entry = np.array(
        [local.original.path.x_mm[start_index], local.original.path.y_mm[start_index]]
    )
    exit_point = np.array(
        [local.original.path.x_mm[end_index], local.original.path.y_mm[end_index]]
    )
    zoom_axis.scatter(
        [entry[0], exit_point[0]],
        [entry[1], exit_point[1]],
        marker="o",
        s=28,
        color=("#111827", "#BE185D"),
        label="入口・出口",
        zorder=8,
    )
    crossing = zoom_source.path.deliberate_line_crossing
    if np.any(crossing):
        zoom_axis.scatter(
            zoom_source.path.x_mm[crossing],
            zoom_source.path.y_mm[crossing],
            marker="x",
            s=26,
            color="#DC2626",
            label="意図的白線交差",
            zorder=9,
        )
    zoom_values_x = np.concatenate(
        (
            local.original.path.x_mm[start_index : end_index + 1],
            zoom_source.path.x_mm[
                (zoom_source.path.source_progress_index >= start_index)
                & (zoom_source.path.source_progress_index <= end_index)
            ],
        )
    )
    zoom_values_y = np.concatenate(
        (
            local.original.path.y_mm[start_index : end_index + 1],
            zoom_source.path.y_mm[
                (zoom_source.path.source_progress_index >= start_index)
                & (zoom_source.path.source_progress_index <= end_index)
            ],
        )
    )
    pad_x = max(80.0, float(np.ptp(zoom_values_x)) * 0.08)
    pad_y = max(80.0, float(np.ptp(zoom_values_y)) * 0.08)
    zoom_axis.set_xlim(float(np.min(zoom_values_x)) - pad_x, float(np.max(zoom_values_x)) + pad_x)
    zoom_axis.set_ylim(float(np.min(zoom_values_y)) - pad_y, float(np.max(zoom_values_y)) + pad_y)
    zoom_axis.set_title(
        f"最大短縮区間: index {start_index}→{end_index}（{kind}）",
        fontsize=11.5,
    )
    zoom_axis.set_xlabel("X [mm]")
    zoom_axis.set_ylabel("Y [mm]")
    zoom_axis.set_aspect("equal", adjustable="box")
    zoom_axis.grid(True, color="#D1D5DB", linewidth=0.5, alpha=0.65)
    zoom_axis.legend(loc="best", fontsize=7.0, framealpha=0.92)

    ref_stats = comparison.reference.stats
    emb_stats = comparison.embedded_lite.stats
    final_stats = (
        ref_stats
        if final.path.selected_edges == reference.path.selected_edges
        else emb_stats
        if final.path.selected_edges == embedded.path.selected_edges
        else None
    )
    stats = (None, ref_stats, emb_stats, final_stats)
    row_labels = (
        "予測時間 [s]",
        "4.000秒との差 [s]",
        "現在経路との差 [s]",
        "経路長 [m]",
        "短縮率 [%]",
        "最大速度 [m/s]",
        "最大 |omega| [deg/s]",
        "最小半径 [mm]",
        "最大 |dκ/ds| [1/m²]",
        "ショートカット辺数",
        "スキップ元距離 [m]",
        "他ライン交差数",
        "最小交差角 [deg]",
        "計算時間 [s]",
        "アンカー数",
        "候補辺数",
        "有効候補辺数",
        "上位完全再評価数",
        "判定",
    )
    columns: list[list[str]] = []
    for item, stat in zip(items, stats, strict=True):
        metrics = item.metrics
        columns.append(
            [
                f"{metrics.predicted_time_s:.3f}",
                f"{metrics.predicted_time_s - 4.0:+.3f}",
                f"{metrics.predicted_time_s - current.metrics.predicted_time_s:+.3f}",
                f"{metrics.length_m:.3f}",
                f"{metrics.shortening_percent:.2f}",
                f"{metrics.max_speed_mps:.1f}",
                f"{metrics.max_omega_deg_s:.0f}",
                f"{metrics.min_radius_mm:.1f}",
                f"{metrics.max_curvature_slew_per_m2:.1f}",
                str(metrics.shortcut_edge_count),
                f"{metrics.skipped_source_distance_m:.2f}",
                str(metrics.line_crossing_count),
                (
                    "―"
                    if not np.isfinite(metrics.min_line_crossing_angle_deg)
                    else f"{metrics.min_line_crossing_angle_deg:.1f}"
                ),
                "―" if stat is None else f"{stat.total_s:.2f}",
                "―" if stat is None else str(stat.anchor_count),
                "―" if stat is None else str(stat.candidate_edge_count),
                "―" if stat is None else str(stat.valid_edge_count),
                "―" if stat is None else str(stat.top_k_count),
                _global_status(item)[:44],
            ]
        )
    table_axis.axis("off")
    table_axis.set_title(
        f"固定ATTACKモデル比較（幾何下限 {comparison.geometric_lower_bound.metrics.predicted_time_s:.3f}s）",
        fontsize=12.5,
        pad=8,
    )
    cell_text = [
        [row_labels[index], *(column[index] for column in columns)]
        for index in range(len(row_labels))
    ]
    table = table_axis.table(
        cellText=cell_text,
        colLabels=("評価項目", "現在", "reference", "embedded", "最終"),
        cellLoc="right",
        colLoc="center",
        bbox=(0.0, 0.08, 1.0, 0.88),
        colWidths=(0.34, 0.145, 0.17, 0.17, 0.175),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.25)
    table.scale(1.0, 1.06)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#E5E7EB")
        elif column == 0:
            cell.set_facecolor("#F3F4F6")
            cell.set_text_props(ha="left")
    table_axis.text(
        0.0,
        0.0,
        f"板境界: {comparison.board_status}\n"
        "規定上限車体での白線到達可能性を検査。実車外形未登録のため実走適合は未確認。",
        transform=table_axis.transAxes,
        fontsize=7.4,
        color="#374151",
        va="bottom",
    )

    for item, label, color, style in zip(items, labels, colors, styles, strict=True):
        speed_axis.plot(
            item.path.source_progress_distance_mm * 0.001,
            item.speed_mps,
            color=color,
            linestyle=style,
            linewidth=1.1,
            label=label,
        )
    reason_y = config.min_speed_mps - 0.22
    speed_axis.scatter(
        final.path.source_progress_distance_mm * 0.001,
        np.full_like(final.speed_mps, reason_y),
        c=np.asarray(LIMIT_COLORS)[final.speed_limit_reason],
        s=2.4,
        linewidths=0.0,
    )
    speed_axis.set_title("速度プロファイル（下端色帯: 最終経路の支配要因）", fontsize=10.5)
    speed_axis.set_ylabel("速度 [m/s]")
    speed_axis.set_ylim(config.min_speed_mps - 0.35, config.max_speed_mps + 0.45)
    speed_axis.grid(True, color="#D1D5DB", linewidth=0.45, alpha=0.65)
    speed_axis.legend(loc="upper right", fontsize=6.4, ncol=2)
    limit_handles = [
        Line2D([0], [0], color=color, linewidth=4, label=label)
        for color, label in zip(
            LIMIT_COLORS,
            ("最高速度", "GFCP", "AALP", "加速", "減速"),
            strict=True,
        )
    ]
    speed_axis.legend(
        handles=limit_handles,
        loc="lower center",
        ncol=5,
        fontsize=6.1,
        framealpha=0.88,
    )

    source_distance = local.original.distance_m * 1000.0
    current_time = _time_on_source_progress(current, source_distance)
    for item, label, color, style in zip(items, labels, colors, styles, strict=True):
        values = _time_on_source_progress(item, source_distance)
        time_axis.plot(
            source_distance * 0.001,
            values,
            color=color,
            linestyle=style,
            linewidth=1.05,
            label=label,
        )
    delta_axis = time_axis.twinx()
    final_time = _time_on_source_progress(final, source_distance)
    delta_axis.plot(
        source_distance * 0.001,
        final_time - current_time,
        color="#111827",
        linewidth=0.9,
        alpha=0.72,
        label="最終－現在",
    )
    delta_axis.axhline(0.0, color="#9CA3AF", linewidth=0.6)
    delta_axis.set_ylabel("累積時間差 [s]", color="#111827")
    time_axis.set_title("累積予測時間と現在経路との差", fontsize=10.5)
    time_axis.set_xlabel("原コース進行距離 [m]")
    time_axis.set_ylabel("累積時間 [s]")
    time_axis.grid(True, color="#D1D5DB", linewidth=0.45, alpha=0.65)
    time_axis.legend(loc="upper left", fontsize=6.1, ncol=2)
    delta_axis.legend(loc="lower right", fontsize=6.4)

    figure.text(
        0.5,
        0.012,
        "固定速度モデル: R10=min=始端=終端3.6、max13.0 m/s、指数0.33、加速20～55、減速55 m/s²、omega 300～1500 deg/s、AALP 100、前後4反復。"
        "予測値は実走性能・競技上の適合・安全を保証しません。",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#4B5563",
    )
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def _draw_vehicle_outline(
    axis,
    components: tuple[np.ndarray, ...],
    x_mm: float,
    y_mm: float,
    yaw_rad: float,
    *,
    color: str,
    alpha: float = 0.20,
) -> None:
    cosine = float(np.cos(yaw_rad))
    sine = float(np.sin(yaw_rad))
    for component in components:
        vertices = np.asarray(component, dtype=np.float64)
        world_x = x_mm + cosine * vertices[:, 0] - sine * vertices[:, 1]
        world_y = y_mm + sine * vertices[:, 0] + cosine * vertices[:, 1]
        axis.add_patch(
            MplPolygon(
                np.column_stack((world_x, world_y)),
                closed=True,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=alpha,
            )
        )


def write_global_result_png(
    destination: str | Path,
    comparison: GlobalComparison,
    config: PlannerConfig,
) -> Path:
    """実車合法性を理論下限から分離した監査画像を作る。"""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    _set_japanese_font()
    plt.rcParams["axes.unicode_minus"] = False

    local = comparison.local
    current = comparison.current_baseline
    theoretical_result = comparison.maximum_vehicle_lower_bound
    theoretical = theoretical_result.adopted
    reference_result = comparison.reference
    embedded_result = comparison.embedded_lite
    reference = reference_result.adopted
    embedded = embedded_result.adopted
    final = comparison.final

    figure = plt.figure(figsize=(20, 12), dpi=150, facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.05, 1.25),
        height_ratios=(1.0, 1.0),
        wspace=0.13,
        hspace=0.18,
    )
    map_axis = figure.add_subplot(grid[0, 0])
    zoom_axis = figure.add_subplot(grid[1, 0])
    table_axis = figure.add_subplot(grid[0, 1])
    contact_axis = figure.add_subplot(grid[1, 1])

    map_axis.plot(
        local.original.path.x_mm,
        local.original.path.y_mm,
        color="#9CA3AF",
        linewidth=1.1,
        label="原コース白線中心",
        zorder=1,
    )
    map_axis.plot(
        current.path.x_mm,
        current.path.y_mm,
        color="#D97706",
        linewidth=1.25,
        linestyle="--",
        label=f"現在Frenet {current.metrics.predicted_time_s:.3f}s",
        zorder=3,
    )
    map_axis.plot(
        theoretical.path.x_mm,
        theoretical.path.y_mm,
        color="#4B5563",
        linewidth=1.35,
        linestyle=(0, (2, 2)),
        label=f"規定最大車体理論下限 {theoretical.metrics.predicted_time_s:.3f}s（競技無効）",
        zorder=4,
    )
    map_axis.plot(
        reference.path.x_mm,
        reference.path.y_mm,
        color="#059669",
        linewidth=1.0,
        linestyle=":",
        label=f"LN5合法reference {reference.metrics.predicted_time_s:.3f}s",
        zorder=2,
    )
    map_axis.plot(
        embedded.path.x_mm,
        embedded.path.y_mm,
        color="#2563EB",
        linewidth=1.0,
        linestyle="-.",
        label=f"embedded-lite {embedded.metrics.predicted_time_s:.3f}s",
        zorder=2,
    )
    map_axis.plot(
        final.path.x_mm,
        final.path.y_mm,
        color="#BE185D",
        linewidth=1.55,
        label=f"最終採用 {final.metrics.predicted_time_s:.3f}s",
        zorder=5,
    )
    theoretical_shortcut = theoretical.path.shortcut_edge_id > 0
    if np.any(theoretical_shortcut):
        map_axis.scatter(
            theoretical.path.x_mm[theoretical_shortcut],
            theoretical.path.y_mm[theoretical_shortcut],
            s=2.4,
            color="#DC2626",
            label="競技無効ショートカット",
            zorder=6,
        )
    map_axis.set_title("2025年全日本 ― LN5実車合法性ゲート", fontsize=13)
    map_axis.set_xlabel("X [mm]")
    map_axis.set_ylabel("Y [mm]")
    map_axis.set_aspect("equal", adjustable="box")
    map_axis.grid(True, color="#D1D5DB", linewidth=0.5, alpha=0.65)
    map_axis.legend(loc="upper right", fontsize=6.6, framealpha=0.94)

    if theoretical.path.selected_edges:
        _, start_index, end_index, kind = max(
            theoretical.path.selected_edges,
            key=lambda edge: edge[2] - edge[1],
        )
    else:
        start_index, end_index, kind = 0, local.original.path.x_mm.size - 1, "なし"
    source_x = local.original.path.x_mm[start_index : end_index + 1]
    source_y = local.original.path.y_mm[start_index : end_index + 1]
    zoom_axis.plot(
        source_x,
        source_y,
        color="#E5E7EB",
        linewidth=6.0,
        solid_capstyle="round",
        label="幅19mm白線領域（表示用）",
        zorder=1,
    )
    zoom_axis.plot(
        source_x,
        source_y,
        color="#6B7280",
        linewidth=0.8,
        linestyle=":",
        label="飛ばした元コース",
        zorder=2,
    )
    zoom_axis.plot(
        current.path.x_mm,
        current.path.y_mm,
        color="#D97706",
        linewidth=1.1,
        linestyle="--",
        label="現在4.471秒",
        zorder=3,
    )
    if np.any(theoretical_shortcut):
        zoom_axis.plot(
            theoretical.path.x_mm[theoretical_shortcut],
            theoretical.path.y_mm[theoretical_shortcut],
            color="#DC2626",
            linewidth=2.0,
            linestyle=(0, (3, 2)),
            label="2.980秒理論下限（競技無効）",
            zorder=6,
        )
    entry_x = float(local.original.path.x_mm[start_index])
    entry_y = float(local.original.path.y_mm[start_index])
    exit_x = float(local.original.path.x_mm[end_index])
    exit_y = float(local.original.path.y_mm[end_index])
    zoom_axis.scatter(
        [entry_x, exit_x],
        [entry_y, exit_y],
        s=30,
        color=("#111827", "#BE185D"),
        label="理論下限の入口・出口",
        zorder=8,
    )
    if comparison.vehicle_footprint.confirmed:
        final_contact = (
            reference_result.contact
            if reference_result.legal
            and final.path.selected_edges == reference.path.selected_edges
            else embedded_result.contact
            if embedded_result.legal
            and final.path.selected_edges == embedded.path.selected_edges
            else None
        )
        if final_contact is not None:
            sample_distance = np.arange(
                0.0,
                float(final_contact.pose_distance_mm[-1]) + 1.0,
                500.0,
            )
            sample_indices = np.searchsorted(final_contact.pose_distance_mm, sample_distance)
            special = np.array(
                [
                    final_contact.min_margin_pose_index,
                    int(np.argmax(np.abs(final_contact.pose_yaw_rad))),
                    int(
                        np.searchsorted(
                            final_contact.pose_distance_mm,
                            final.path.cumulative_distance_mm[
                                int(np.argmax(np.abs(final.path.curvature_per_m)))
                            ],
                        )
                    ),
                ],
                dtype=np.int32,
            )
            switch_indices = np.flatnonzero(
                np.diff(final_contact.source_progress_index) > 1.0
            ) + 1
            crossing_indices = np.searchsorted(
                final_contact.pose_distance_mm,
                final.path.cumulative_distance_mm[final.path.deliberate_line_crossing],
            )
            sample_indices = np.unique(
                np.clip(
                    np.concatenate(
                        (sample_indices, special, switch_indices, crossing_indices)
                    ),
                    0,
                    final_contact.pose_x_mm.size - 1,
                )
            )
            for pose_index in sample_indices:
                _draw_vehicle_outline(
                    zoom_axis,
                    comparison.vehicle_footprint.polygon_components_mm,
                    float(final_contact.pose_x_mm[pose_index]),
                    float(final_contact.pose_y_mm[pose_index]),
                    float(final_contact.pose_yaw_rad[pose_index]),
                    color="#059669",
                )
        theoretical_contact = theoretical_result.contact
        if theoretical_contact is not None:
            detached = np.asarray(
                [not segments for segments in theoretical_contact.contact_segments],
                dtype=np.bool_,
            )
            detached_starts = np.flatnonzero(
                detached & np.concatenate(([True], ~detached[:-1]))
            )
            for pose_index in detached_starts[:30]:
                _draw_vehicle_outline(
                    zoom_axis,
                    comparison.vehicle_footprint.polygon_components_mm,
                    float(theoretical_contact.pose_x_mm[pose_index]),
                    float(theoretical_contact.pose_y_mm[pose_index]),
                    float(theoretical_contact.pose_yaw_rad[pose_index]),
                    color="#DC2626",
                    alpha=0.35,
                )
    else:
        zoom_axis.text(
            0.03,
            0.04,
            "LN5実車床面投影が未設定のため\n"
            "外形・重なり・最小接触余裕は描画しない",
            transform=zoom_axis.transAxes,
            fontsize=9.0,
            color="#B91C1C",
            bbox={"facecolor": "#FEF2F2", "edgecolor": "#FCA5A5", "alpha": 0.95},
        )
    zoom_x = np.concatenate((source_x, theoretical.path.x_mm[theoretical_shortcut]))
    zoom_y = np.concatenate((source_y, theoretical.path.y_mm[theoretical_shortcut]))
    pad_x = max(100.0, float(np.ptp(zoom_x)) * 0.08)
    pad_y = max(100.0, float(np.ptp(zoom_y)) * 0.08)
    zoom_axis.set_xlim(float(np.min(zoom_x)) - pad_x, float(np.max(zoom_x)) + pad_x)
    zoom_axis.set_ylim(float(np.min(zoom_y)) - pad_y, float(np.max(zoom_y)) + pad_y)
    zoom_axis.set_title(
        f"無効化した最大ジャンプ: index {start_index}→{end_index}（{kind}）",
        fontsize=11.5,
    )
    zoom_axis.set_xlabel("X [mm]")
    zoom_axis.set_ylabel("Y [mm]")
    zoom_axis.set_aspect("equal", adjustable="box")
    zoom_axis.grid(True, color="#D1D5DB", linewidth=0.5, alpha=0.65)
    zoom_axis.legend(loc="best", fontsize=6.7, framealpha=0.93)

    results = (None, theoretical_result, reference_result, embedded_result, None)
    items = (current, theoretical, reference, embedded, final)
    column_names = ("現在", "2.980理論", "legal ref", "embedded", "最終")
    contacts = (
        None,
        theoretical_result.contact,
        reference_result.contact,
        embedded_result.contact,
        reference_result.contact
        if reference_result.legal and final.path.selected_edges == reference.path.selected_edges
        else embedded_result.contact
        if embedded_result.legal and final.path.selected_edges == embedded.path.selected_edges
        else None,
    )
    legal_labels = (
        "既存fallback",
        "不合法",
        "合法" if reference_result.legal else "未評価",
        "合法" if embedded_result.legal else "未評価",
        "合法"
        if contacts[-1] is not None and contacts[-1].legal
        else "正式fallback（未評価）",
    )
    row_labels = (
        "予測時間 [s]",
        "4.000秒との差 [s]",
        "合法／不合法",
        "robust／警告",
        "白線完全離脱数",
        "最小重なり面積 [mm²]",
        "最小接触余裕 [mm]",
        "ライン乗り換え数",
        "浅角交差数",
        "平行近接距離 [m]",
        "板境界判定",
        "実車外形ソース",
        "ショートカット辺数",
        "計算時間 [s]",
    )
    columns: list[list[str]] = []
    for position, (item, result, contact) in enumerate(
        zip(items, results, contacts, strict=True)
    ):
        metrics = item.metrics
        robust_text = (
            "robust" if contact is not None and contact.robust else "警告／未評価"
        )
        detachment = "―" if contact is None else str(contact.detachment_count)
        min_area = "―" if contact is None else f"{contact.min_overlap_area_mm2:.1f}"
        min_margin = "―" if contact is None else f"{contact.min_contact_margin_mm:.2f}"
        switches = "―" if contact is None else str(contact.line_switch_count)
        shallow = metrics.shallow_line_crossing_count
        board_text = (
            "実車外形で確認"
            if contact is not None and contact.legal and comparison.board_boundary is not None
            else "実車未確認"
        )
        source_text = (
            "規定最大250mm仮想円"
            if position == 1
            else "LN5未設定"
            if not comparison.vehicle_footprint.confirmed
            else "LN5確認済み外形"
        )
        columns.append(
            [
                f"{metrics.predicted_time_s:.3f}",
                f"{metrics.predicted_time_s - 4.0:+.3f}",
                legal_labels[position],
                robust_text,
                detachment,
                min_area,
                min_margin,
                switches,
                str(shallow),
                f"{metrics.parallel_line_distance_m:.3f}",
                board_text,
                source_text,
                str(metrics.shortcut_edge_count),
                "―" if result is None else f"{result.stats.total_s:.2f}",
            ]
        )
    table_axis.axis("off")
    table_axis.set_title("競技合法性ハードゲート比較", fontsize=12.5, pad=8)
    cell_text = [
        [row_labels[index], *(column[index] for column in columns)]
        for index in range(len(row_labels))
    ]
    table = table_axis.table(
        cellText=cell_text,
        colLabels=("評価項目", *column_names),
        cellLoc="right",
        colLoc="center",
        bbox=(0.0, 0.12, 1.0, 0.82),
        colWidths=(0.31, 0.13, 0.14, 0.14, 0.14, 0.14),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.1)
    table.scale(1.0, 1.12)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#E5E7EB")
        elif column == 0:
            cell.set_facecolor("#F3F4F6")
            cell.set_text_props(ha="left")
        if row > 0 and column == 2:
            cell.set_facecolor("#FEF2F2")
    table_axis.text(
        0.0,
        0.02,
        f"{comparison.footprint_status}\n"
        "2.980秒と2.004秒は理論下限であり、競技有効経路として採用しない。",
        transform=table_axis.transAxes,
        fontsize=8.0,
        color="#991B1B",
        va="bottom",
    )

    final_contact = contacts[-1]
    if final_contact is None:
        contact_axis.axis("off")
        contact_axis.text(
            0.5,
            0.62,
            "LN5実車外形未設定",
            ha="center",
            va="center",
            fontsize=18,
            color="#B91C1C",
        )
        contact_axis.text(
            0.5,
            0.47,
            "実接触segment・重なり面積・接触余裕・同時接触数は\n"
            "外形確認まで計算しない。最終は4.471秒へフォールバック。",
            ha="center",
            va="center",
            fontsize=11,
            color="#374151",
        )
        contact_axis.set_title("実接触状態グラフ（未評価）", fontsize=12)
    else:
        distance_m = final_contact.pose_distance_mm * 0.001
        contact_min = np.asarray(
            [min(segments) if segments else np.nan for segments in final_contact.contact_segments]
        )
        contact_max = np.asarray(
            [max(segments) if segments else np.nan for segments in final_contact.contact_segments]
        )
        contact_axis.fill_between(
            distance_m,
            contact_min,
            contact_max,
            color="#BFDBFE",
            alpha=0.55,
            label="contact segment候補範囲",
        )
        contact_axis.plot(
            distance_m,
            final_contact.source_progress_index,
            color="#2563EB",
            linewidth=1.0,
            label="DP選択source progress",
        )
        metric_axis = contact_axis.twinx()
        metric_axis.plot(
            distance_m,
            final_contact.overlap_area_mm2,
            color="#059669",
            linewidth=0.9,
            label="重なり面積 [mm²]",
        )
        metric_axis.plot(
            distance_m,
            final_contact.contact_margin_mm,
            color="#D97706",
            linewidth=0.9,
            label="接触余裕 [mm]",
        )
        metric_axis.scatter(
            distance_m,
            final_contact.simultaneous_line_count,
            s=2.0,
            color="#BE185D",
            label="同時接触ライン数",
        )
        contact_axis.set_xlabel("合法経路距離 [m]")
        contact_axis.set_ylabel("実接触segment index")
        metric_axis.set_ylabel("重なり・余裕・同時接触")
        contact_axis.grid(True, color="#D1D5DB", linewidth=0.45, alpha=0.65)
        contact_axis.legend(loc="upper left", fontsize=7.0)
        metric_axis.legend(loc="upper right", fontsize=7.0)
        contact_axis.set_title("実接触segment DPと接触余裕", fontsize=12)

    figure.text(
        0.5,
        0.012,
        "固定ATTACKモデル: R10=min=始端=終端=3.6m/s、max=13.0m/s、加速20～55、減速55m/s²、omega 300～1500deg/s、AALP 100、前後4反復。"
        "実車外形未設定時は大域経路を採用しない。",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#4B5563",
    )
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def write_all_courses_png(
    destination: str | Path,
    results: list[BatchCourseResult],
) -> Path:
    """31コースの改善量、フォールバック、異常終了有無を一覧化する。"""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    _set_japanese_font()
    ordered = sorted(results, key=lambda item: (item.improvement_s, item.course_id))
    labels = [item.course_id for item in ordered]
    improvements = np.asarray([item.improvement_s for item in ordered])
    colors = [
        "#DC2626" if not item.valid else "#9CA3AF" if item.fallback_used else "#059669"
        for item in ordered
    ]
    figure, (bar_axis, table_axis) = plt.subplots(
        1,
        2,
        figsize=(18, 12),
        dpi=150,
        gridspec_kw={"width_ratios": (0.9, 1.45), "wspace": 0.12},
        facecolor="white",
    )
    y = np.arange(len(ordered))
    bar_axis.barh(y, improvements, color=colors, height=0.72)
    bar_axis.axvline(0.0, color="#111827", linewidth=0.7)
    bar_axis.set_yticks(y, labels, fontsize=7.1)
    bar_axis.invert_yaxis()
    bar_axis.set_xlabel("既存最良に対する時間差 [s]（負が改善）")
    bar_axis.set_title("全31大会コース embedded-lite回帰", fontsize=13)
    bar_axis.grid(True, axis="x", color="#D1D5DB", linewidth=0.5, alpha=0.65)
    if improvements.size and float(np.max(np.abs(improvements))) < 1.0e-4:
        bar_axis.set_xlim(-0.01, 0.01)
    for position, value in enumerate(improvements):
        bar_axis.text(
            value,
            position,
            f" {value:+.3f}",
            va="center",
            ha="left" if value >= 0.0 else "right",
            fontsize=6.5,
        )

    table_axis.axis("off")
    def short_status(item: BatchCourseResult) -> str:
        if not item.valid:
            return "不合格"
        if "LN5実車外形未設定" in item.status:
            return "fallback（外形未設定）"
        return "fallback" if item.fallback_used else "改善"

    table_rows = [
        [
            item.course_id,
            f"{item.baseline_time_s:.3f}",
            f"{item.selected_time_s:.3f}",
            f"{item.improvement_s:+.3f}",
            str(item.anchor_count),
            str(item.candidate_edge_count),
            f"{item.total_s:.2f}",
            short_status(item),
        ]
        for item in sorted(results, key=lambda value: value.course_id)
    ]
    table = table_axis.table(
        cellText=table_rows,
        colLabels=(
            "コース",
            "既存 [s]",
            "採用 [s]",
            "差 [s]",
            "アンカー",
            "候補辺",
            "時間 [s]",
            "判定",
        ),
        cellLoc="right",
        colLoc="center",
        bbox=(0.0, 0.04, 1.0, 0.92),
        colWidths=(0.19, 0.10, 0.10, 0.09, 0.09, 0.10, 0.09, 0.24),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor("#E5E7EB")
        elif column == 0:
            cell.set_facecolor("#F3F4F6")
            cell.set_text_props(ha="left")
    improved = sum(item.improvement_s < -1.0e-6 and item.valid for item in results)
    fallback = sum(item.fallback_used and item.valid for item in results)
    invalid = sum(not item.valid for item in results)
    table_axis.set_title(
        f"改善 {improved} / フォールバック {fallback} / 不合格 {invalid}",
        fontsize=12.5,
    )
    figure.text(
        0.5,
        0.012,
        "LN5実車外形未設定のため全大会で大域探索を停止し既存経路へfallback。板境界未収録大会は板内も未確認。実走保証ではありません。",
        ha="center",
        fontsize=8.2,
        color="#4B5563",
    )
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output
