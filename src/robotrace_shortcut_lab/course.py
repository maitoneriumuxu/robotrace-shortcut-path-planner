from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import CoursePath


def load_course(path: str | Path) -> CoursePath:
    """保存済み10 mm間隔TSVからコース座標だけを読む。"""

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
    return CoursePath(course_id, values[:, 0], values[:, 1], values[:, 2])
