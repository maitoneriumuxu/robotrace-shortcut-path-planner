from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import BoardBoundary, Course


def load_course(path: str | Path) -> Course:
    """保存済みTSVから距離とXYだけを読み込む。"""

    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()
    course_id = source.stem
    header_index: int | None = None
    for index, line in enumerate(lines):
        if line.startswith("# course_id\t"):
            course_id = line.split("\t", 1)[1]
        if line == "distance_mm\tline_x_mm\tline_y_mm":
            header_index = index
            break
    if header_index is None:
        raise ValueError(f"コース列が見つかりません: {source}")

    values = np.loadtxt(lines[header_index + 1 :], delimiter="\t", dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"コース列数が不正です: {source}")
    if values.shape[0] > 6100:
        raise ValueError(f"6100点を超えるコースです: {values.shape[0]}")
    if not np.isfinite(values).all():
        raise ValueError("コースにNaNまたは無限値があります")

    return Course(course_id, values[:, 0], values[:, 1], values[:, 2])


def load_board_boundary(
    course: Course,
    metadata_path: str | Path = "data/courses/board_boundaries.json",
) -> BoardBoundary | None:
    """CADから確認できた競技板セル境界を読む。未収録コースはNone。"""

    source = Path(metadata_path)
    if not source.exists():
        return None
    data = json.loads(source.read_text(encoding="utf-8"))
    entry = data.get("courses", {}).get(course.course_id)
    if not isinstance(entry, dict):
        return None
    raw_rectangles = entry.get("rectangles_mm", [])
    rectangles = tuple(
        tuple(float(value) for value in rectangle)
        for rectangle in raw_rectangles
        if isinstance(rectangle, list) and len(rectangle) == 4
    )
    if not rectangles:
        return None
    return BoardBoundary(
        rectangles,
        str(entry.get("source_file", source)),
        bool(entry.get("confirmed", False)),
    )
