from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np

from .model import Comparison, PlannerConfig


COLORS = ("#6B7280", "#2563EB", "#059669")


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
    paths = (comparison.original, comparison.elastic, comparison.best)
    base_time = comparison.original.metrics.predicted_time_s
    labels = [
        "予測走行時間 [s]",
        "原コース比 [s]",
        "経路長 [m]",
        "経路短縮率 [%]",
        "最大横オフセット [mm]",
        "最小半径 [mm]",
        "最大 |dκ/ds| [1/m²]",
        "候補生成時間 [ms]",
        "制約判定",
    ]
    rows: list[list[str]] = []
    for item in paths:
        metrics = item.metrics
        rows.append(
            [
                f"{metrics.predicted_time_s:.3f}",
                f"{metrics.predicted_time_s - base_time:+.3f}",
                f"{metrics.length_m:.3f}",
                f"{metrics.shortening_percent:.2f}",
                f"{metrics.max_offset_mm:.1f}",
                f"{metrics.min_radius_mm:.1f}",
                f"{metrics.max_curvature_slew_per_m2:.1f}",
                f"{item.path.generation_s * 1000.0:.1f}",
                "合格" if metrics.valid else f"違反: {metrics.violation}",
            ]
        )
    return labels, rows


def write_result_png(
    destination: str | Path,
    comparison: Comparison,
    config: PlannerConfig,
) -> Path:
    """全体経路、比較表、速度プロファイルを日本語PNGへまとめる。"""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    _set_japanese_font()
    plt.rcParams["axes.unicode_minus"] = False

    figure = plt.figure(figsize=(16, 9), dpi=150, facecolor="white")
    grid = figure.add_gridspec(
        2, 2, width_ratios=(1.25, 1.0), height_ratios=(1.08, 0.92), wspace=0.19, hspace=0.27
    )
    map_axis = figure.add_subplot(grid[:, 0])
    table_axis = figure.add_subplot(grid[0, 1])
    speed_axis = figure.add_subplot(grid[1, 1])

    paths = (comparison.original, comparison.elastic, comparison.best)
    styles = (("-", 1.55), ("-", 1.45), ("-", 1.55))
    for item, color, (line_style, width) in zip(paths, COLORS, styles, strict=True):
        map_axis.plot(
            item.path.x_mm,
            item.path.y_mm,
            color=color,
            linestyle=line_style,
            linewidth=width,
            label=item.path.label,
            alpha=0.94,
        )
    map_axis.set_title("2025年 全日本ロボトレース ― ショートカット経路比較", fontsize=14)
    map_axis.set_xlabel("X [mm]")
    map_axis.set_ylabel("Y [mm]")
    map_axis.set_aspect("equal", adjustable="box")
    map_axis.grid(True, color="#D1D5DB", linewidth=0.55, alpha=0.7)
    map_axis.legend(loc="best", framealpha=0.92, fontsize=9)

    labels, rows = _format_rows(comparison)
    table_axis.axis("off")
    table_axis.set_title("3方式の比較（簡易速度計画）", fontsize=13, pad=10)
    cell_text = [[labels[index], rows[0][index], rows[1][index], rows[2][index]] for index in range(len(labels))]
    table = table_axis.table(
        cellText=cell_text,
        colLabels=("評価項目", "原コース", "Elastic Band", "時間選択型"),
        cellLoc="right",
        colLoc="center",
        bbox=(0.0, 0.08, 1.0, 0.86),
        colWidths=(0.39, 0.19, 0.21, 0.21),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1.0, 1.28)
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
        f"採用候補: #{comparison.selected_candidate_id} {comparison.selected_candidate_name} / "
        "時間選択型の生成時間は8候補評価＋最良候補再生成の合計",
        transform=table_axis.transAxes,
        fontsize=8.5,
        color="#374151",
        va="bottom",
    )

    for item, color in zip(paths, COLORS, strict=True):
        speed_axis.plot(
            item.distance_m,
            item.speed_mps,
            color=color,
            linewidth=1.35,
            label=item.path.label,
        )
    speed_axis.set_title("経路距離に対する速度プロファイル", fontsize=12)
    speed_axis.set_xlabel("経路距離 [m]")
    speed_axis.set_ylabel("速度 [m/s]")
    speed_axis.set_ylim(0.8, config.max_speed_mps + 0.4)
    speed_axis.grid(True, color="#D1D5DB", linewidth=0.55, alpha=0.7)
    speed_axis.legend(loc="upper right", fontsize=8)

    figure.text(
        0.5,
        0.012,
        "注意: 最大8.0 m/s・最低1.0 m/s・加速10.0 m/s²・減速30.0 m/s²・最大角速度1500 deg/s・"
        "GFCP基準3.0 m/s・指数0.33・オフセット75 mm・最小半径100 mmは比較用の仮設定で、実走を保証しません。",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4B5563",
    )
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output
