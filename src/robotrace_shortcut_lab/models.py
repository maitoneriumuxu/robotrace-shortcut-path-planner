from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class CoursePath:
    course_id: str
    distance_mm: FloatArray
    x_mm: FloatArray
    y_mm: FloatArray

    def __post_init__(self) -> None:
        for name in ("distance_mm", "x_mm", "y_mm"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.float32))
        if not self.distance_mm.size or not (
            self.distance_mm.size == self.x_mm.size == self.y_mm.size
        ):
            raise ValueError("コース点列が空か、列数が一致しません")
        if np.any(np.diff(self.distance_mm) <= 0.0):
            raise ValueError("distance_mmは単調増加である必要があります")

    @property
    def point_count(self) -> int:
        return int(self.distance_mm.size)


@dataclass(frozen=True)
class Path:
    label: str
    x_mm: FloatArray
    y_mm: FloatArray
    straight_cores: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_mm", np.asarray(self.x_mm, dtype=np.float32))
        object.__setattr__(self, "y_mm", np.asarray(self.y_mm, dtype=np.float32))
        if not self.x_mm.size or self.x_mm.size != self.y_mm.size:
            raise ValueError("経路XYの点数が一致しません")


@dataclass(frozen=True)
class Settings:
    offset_limit_mm: float = 100.0
    min_radius_mm: float = 60.0
    radius_window_segments: int = 20
    elastic_iterations: int = 600
    source_weight: float = 0.0005
    smooth_weight: float = 0.36
    radius_relax_target_mm: float = 100.0
    radius_relax_every: int = 10
    radius_relax_blend: float = 0.12
    rdp_tolerance_mm: float = 60.0
    straight_max_deviation_mm: float = 90.0
    straight_min_length_mm: float = 2000.0
    straight_min_chord_ratio: float = 0.96
    # 長い直線を優先しつつ、曲率連続接続は450 mm相当の遷移を確保する。
    transition_segments: int = 45
    max_anchor_count: int = 128


@dataclass(frozen=True)
class Metrics:
    path_length_mm: float
    shortening_percent: float
    max_offset_mm: float
    min_radius_mm: float
    max_curvature_slew_per_m2: float
    join_curvature_slew_per_m2: float
    valid: bool
