from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .model import Comparison, PlannerConfig


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
        "注意: 過去ATTACKパラメータ3基準（3.6～13.0 m/s、GFCP 3.0 m/s、指数0.33、加速20～55 m/s²、減速55 m/s²）。"
        "Python予測は実走可能性・実機性能・競技上の安全を保証しません。",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4B5563",
    )
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output
