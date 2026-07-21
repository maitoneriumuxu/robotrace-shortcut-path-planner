from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


COURSE_FILE = Path("data/courses/normalized/2025alljapan.tsv")


@dataclass(frozen=True)
class PlannerConfig:
    """経路比較用の仮設定。実機保証値ではない。"""

    max_speed_mps: float = 8.0
    min_speed_mps: float = 1.0
    acceleration_mps2: float = 10.0
    deceleration_mps2: float = 30.0
    max_omega_deg_s: float = 1500.0
    gfcp_reference_speed_mps: float = 3.0
    gfcp_exponent: float = 0.33
    max_angular_accel_rad_s2: float = 300.0
    offset_limit_mm: float = 75.0
    min_radius_mm: float = 100.0
    radius_window: int = 20
    edge_keep_points: int = 30
    edge_blend_points: int = 30


@dataclass(frozen=True)
class CandidateWeights:
    candidate_id: int
    name: str
    iterations: int
    length_weight: float
    curvature_weight: float
    slew_weight: float
    source_weight: float
    step_limit_mm: float = 1.0


CANDIDATES: tuple[CandidateWeights, ...] = (
    CandidateWeights(0, "曲率重視", 220, 0.10, 0.30, 0.12, 0.0030),
    CandidateWeights(1, "曲率変化重視", 240, 0.08, 0.16, 0.34, 0.0040),
    CandidateWeights(2, "安定バランス", 280, 0.20, 0.22, 0.16, 0.0020),
    CandidateWeights(3, "標準バランス", 400, 0.34, 0.16, 0.12, 0.0010, 1.1),
    CandidateWeights(4, "短縮バランス", 500, 0.42, 0.14, 0.12, 0.0006, 1.2),
    CandidateWeights(5, "経路長重視1", 600, 0.50, 0.12, 0.10, 0.0003, 1.3),
    CandidateWeights(6, "経路長重視2", 700, 0.58, 0.10, 0.08, 0.0002, 1.4),
    CandidateWeights(7, "最大短縮", 700, 0.65, 0.08, 0.08, 0.0001, 1.5),
)


@dataclass(frozen=True)
class Course:
    course_id: str
    distance_mm: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray

    @property
    def point_count(self) -> int:
        return int(self.x_mm.size)


@dataclass(frozen=True)
class GeneratedPath:
    label: str
    x_mm: np.ndarray
    y_mm: np.ndarray
    offset_mm: np.ndarray
    generation_s: float
    candidate_id: int | None = None
    candidate_name: str | None = None


@dataclass(frozen=True)
class Metrics:
    predicted_time_s: float
    length_m: float
    shortening_percent: float
    max_offset_mm: float
    min_radius_mm: float
    max_curvature_slew_per_m2: float
    valid: bool
    violation: str


@dataclass(frozen=True)
class EvaluatedPath:
    path: GeneratedPath
    metrics: Metrics
    distance_m: np.ndarray
    speed_mps: np.ndarray


@dataclass(frozen=True)
class Comparison:
    original: EvaluatedPath
    elastic: EvaluatedPath
    best: EvaluatedPath
    selected_candidate_id: int
    selected_candidate_name: str
    candidate_search_s: float
    candidate_generation_s: tuple[float, ...]
