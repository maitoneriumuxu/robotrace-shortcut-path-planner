from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


COURSE_FILE = Path("data/courses/normalized/2025alljapan.tsv")


@dataclass(frozen=True)
class PlannerConfig:
    """全日本ATTACKパラメータ3を基準にした比較設定。"""

    max_speed_mps: float = 13.0
    min_speed_mps: float = 3.6
    gfcp_reference_speed_mps: float = 3.6
    gfcp_exponent: float = 0.33
    min_acceleration_mps2: float = 20.0
    max_acceleration_mps2: float = 55.0
    deceleration_mps2: float = 55.0
    break_kp: float = 1.00
    max_acceleration_omega_deg_s: float = 300.0
    min_acceleration_omega_deg_s: float = 1500.0
    jerk_mps3: float = 3.0
    speed_scan_iterations: int = 4
    # 計時前はゴールライン付近からスタートラインまでの直線を助走する。
    # 距離は経路終点（ゴール）と始点（スタート）の実座標差から求める。
    gate_runup_initial_speed_mps: float = 0.0
    finish_speed_is_free: bool = True
    search_run_speed_mps: float = 3.6
    max_aalp_deg_s_per_ms: float = 100.0
    firmware_sample_s: float = 0.001
    offset_limit_mm: float = 75.0
    legacy_min_radius_mm: float = 100.0
    radius_window: int = 20
    edge_keep_points: int = 30
    edge_blend_points: int = 30
    max_segment_mm: float = 20.0
    max_offset_step_mm: float = 10.0
    max_curvature_slew_per_m2: float = 180.0
    long_window_coarse: tuple[tuple[float, float], ...] = (
        (200.0, -5.0),
        (200.0, 5.0),
        (500.0, -10.0),
        (500.0, 10.0),
        (800.0, -20.0),
        (800.0, 20.0),
    )
    long_window_refine: tuple[tuple[float, float], ...] = (
        (200.0, -2.5),
        (200.0, 2.5),
        (500.0, -5.0),
        (500.0, 5.0),
        (800.0, -5.0),
        (800.0, 5.0),
    )
    global_resample_interval_mm: float = 10.0
    reference_anchor_limit: int = 256
    embedded_anchor_limit: int = 96
    reference_edge_limit: int = 20_000
    embedded_edge_limit: int = 1_200
    reference_top_k: int = 64
    embedded_top_k: int = 8
    legal_reference_edge_check_limit: int = 1_200
    legal_embedded_edge_check_limit: int = 300
    legal_reference_top_k: int = 16
    reference_max_skip_mm: float = 12_000.0
    embedded_max_skip_mm: float = 6_000.0
    shortcut_min_skip_mm: float = 250.0
    shortcut_min_saving_mm: float = 35.0
    connector_max_angle_deg: float = 55.0
    # 旧2.980秒の非実車・非合法性確認の理論下限だけに使う。
    # reference/embedded-lite/最終採用の合法判定には使用禁止。
    rule_max_robot_radius_mm: float = 125.0
    rule_line_half_width_mm: float = 9.5
    line_parallel_warning_distance_mm: float = 45.0
    line_parallel_warning_angle_deg: float = 15.0
    # 以下の零係数も旧理論下限の再現専用。合法候補には後段の
    # legal_*非零係数を用い、白線完全離脱は係数でなくハード棄却する。
    line_crossing_penalty_s: float = 0.0
    line_shallow_crossing_penalty_s: float = 0.0
    line_parallel_penalty_s_per_m: float = 0.0
    board_robot_margin_mm: float = 125.0  # 旧規定最大車体理論下限専用
    white_line_half_width_mm: float = 9.5
    legal_pose_step_mm: float = 2.0
    legal_yaw_step_deg: float = 1.0
    # 競技では全LINE区間の通過が必要。通常の実接触progressは同一segmentに
    # 留まるか次へ進む。同一姿勢で中間segment全てへ実接触した場合だけ、
    # 点列密度に応じた複数segment前進を許す（未接触segmentの飛越しは禁止）。
    contact_dp_max_progress_step_segments: int = 1
    minimum_overlap_area_mm2: float = 20.0
    minimum_penetration_mm: float = 1.0
    minimum_contact_margin_mm: float = 1.0
    robust_thresholds_confirmed: bool = False
    robust_position_errors_mm: tuple[float, ...] = (1.0, 2.0, 5.0)
    robust_yaw_errors_deg: tuple[float, ...] = (0.5, 1.0, 2.0)
    # 合法性と分離したセンサ誤認リスク。合法候補間の同順位比較にだけ使う。
    legal_crossing_risk_penalty_s: float = 0.01
    legal_shallow_crossing_risk_penalty_s: float = 0.10
    legal_parallel_risk_penalty_s_per_m: float = 0.05


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
    frenet_locked: bool = True


@dataclass(frozen=True)
class Metrics:
    predicted_time_s: float
    length_m: float
    shortening_percent: float
    max_offset_mm: float
    min_radius_mm: float
    max_curvature_slew_per_m2: float
    max_speed_mps: float
    gfcp_only_time_s: float
    valid: bool
    violation: str


@dataclass(frozen=True)
class EvaluatedPath:
    path: GeneratedPath
    metrics: Metrics
    distance_m: np.ndarray
    speed_mps: np.ndarray
    gfcp_only_speed_mps: np.ndarray
    speed_limit_reason: np.ndarray
    speed_plan_s: float


@dataclass(frozen=True)
class SpeedPlan:
    distance_m: np.ndarray
    speed_mps: np.ndarray
    gfcp_limit_mps: np.ndarray
    aalp_limit_mps: np.ndarray
    acceleration_limit_mps2: np.ndarray
    limit_reason: np.ndarray
    elapsed_s: float


@dataclass(frozen=True)
class Comparison:
    original: EvaluatedPath
    elastic: EvaluatedPath
    legacy_time: EvaluatedPath
    best: EvaluatedPath
    selected_candidate_id: int
    selected_candidate_name: str
    candidate_search_s: float
    candidate_evaluation_count: int
    window_center_indices: tuple[int, ...]
    approximate_o_n_scans: int


@dataclass(frozen=True)
class BoardBoundary:
    """CADの競技板セルを正規化座標へ変換した走行面境界。"""

    rectangles_mm: tuple[tuple[float, float, float, float], ...]
    source: str
    confirmed: bool


@dataclass(frozen=True)
class VehicleFootprint:
    """全車体外形と白線接触を証明する物理部品を分離した定義。"""

    full_footprint_components_mm: tuple[np.ndarray, ...]
    contact_witness_components_mm: tuple[np.ndarray, ...]
    origin_definition: str
    board_clearance_radius_mm: float
    safety_margin_mm: float
    source: str
    full_footprint_source: str
    contact_witness_source: str
    design_confirmed: bool
    as_built_confirmed: bool


@dataclass(frozen=True)
class ContactSensitivity:
    """接触証明部品の取付位置・yaw誤差に対する合法性。"""

    position_error_mm: np.ndarray
    position_all_legal: np.ndarray
    yaw_error_deg: np.ndarray
    yaw_all_legal: np.ndarray
    robust_2mm_1deg: bool
    evaluated_variants: int


@dataclass(frozen=True)
class ContactEvaluation:
    """2 mm以下の中間姿勢を含む実車白線接触評価。"""

    pose_distance_mm: np.ndarray
    pose_x_mm: np.ndarray
    pose_y_mm: np.ndarray
    pose_yaw_rad: np.ndarray
    contact_segments: tuple[tuple[int, ...], ...]
    source_progress_index: np.ndarray
    overlap_area_mm2: np.ndarray
    penetration_mm: np.ndarray
    contact_boundary_length_mm: np.ndarray
    contact_margin_mm: np.ndarray
    simultaneous_line_count: np.ndarray
    past_contact_count: np.ndarray
    current_contact_count: np.ndarray
    future_contact_count: np.ndarray
    legal: bool
    robust: bool
    detachment_count: int
    line_switch_count: int
    all_line_segments_covered: bool
    unvisited_segment_count: int
    min_overlap_area_mm2: float
    min_penetration_mm: float
    min_contact_margin_mm: float
    near_point_contact_distance_mm: float
    simultaneous_contact_pose_count: int
    min_margin_pose_index: int
    violation: str
    warning: str


@dataclass(frozen=True)
class GlobalPath:
    """約10 mm間隔へ再サンプリングした大域経路。"""

    label: str
    x_mm: np.ndarray
    y_mm: np.ndarray
    cumulative_distance_mm: np.ndarray
    source_progress_index: np.ndarray
    source_progress_distance_mm: np.ndarray
    shortcut_edge_id: np.ndarray
    deliberate_line_crossing: np.ndarray
    yaw_rad: np.ndarray
    curvature_per_m: np.ndarray
    curvature_slew_per_m2: np.ndarray
    speed_mps: np.ndarray
    generation_s: float
    selected_edges: tuple[tuple[int, int, int, str], ...]


@dataclass(frozen=True)
class GlobalMetrics:
    predicted_time_s: float
    length_m: float
    shortening_percent: float
    max_speed_mps: float
    max_omega_deg_s: float
    min_radius_mm: float
    max_curvature_slew_per_m2: float
    shortcut_edge_count: int
    skipped_source_distance_m: float
    line_crossing_count: int
    shallow_line_crossing_count: int
    min_line_crossing_angle_deg: float
    past_line_crossing_count: int
    future_line_crossing_count: int
    parallel_line_distance_m: float
    valid: bool
    warning: str
    violation: str


@dataclass(frozen=True)
class EvaluatedGlobalPath:
    path: GlobalPath
    metrics: GlobalMetrics
    distance_m: np.ndarray
    speed_mps: np.ndarray
    speed_limit_reason: np.ndarray
    cumulative_time_s: np.ndarray
    speed_plan_s: float


@dataclass(frozen=True)
class GlobalSearchStats:
    mode: str
    anchor_count: int
    candidate_edge_count: int
    valid_edge_count: int
    geometry_check_s: float
    graph_search_s: float
    top_k_count: int
    full_evaluation_s: float
    local_finish_s: float
    total_s: float
    approximate_o_n_scans: int
    max_work_memory_bytes: int


@dataclass(frozen=True)
class GlobalSearchResult:
    mode: str
    anchor_indices: np.ndarray
    best_global: EvaluatedGlobalPath
    adopted: EvaluatedGlobalPath
    stats: GlobalSearchStats
    fallback_used: bool
    contact: ContactEvaluation | None = None
    sensitivity: ContactSensitivity | None = None
    robust_best: EvaluatedGlobalPath | None = None
    robust_best_contact: ContactEvaluation | None = None
    robust_best_sensitivity: ContactSensitivity | None = None
    legal: bool = False
    robust: bool = False
    legality_status: str = "競技合法性未評価"


@dataclass(frozen=True)
class GlobalComparison:
    local: Comparison
    current_baseline: EvaluatedGlobalPath
    geometric_lower_bound: EvaluatedGlobalPath
    maximum_vehicle_lower_bound: GlobalSearchResult
    reference: GlobalSearchResult
    embedded_lite: GlobalSearchResult
    final: EvaluatedGlobalPath
    robust_final: EvaluatedGlobalPath | None
    robust_final_contact: ContactEvaluation | None
    robust_final_sensitivity: ContactSensitivity | None
    board_boundary: BoardBoundary | None
    board_status: str
    vehicle_footprint: VehicleFootprint
    footprint_status: str


@dataclass(frozen=True)
class BatchCourseResult:
    course_id: str
    baseline_time_s: float
    selected_time_s: float
    improvement_s: float
    mode: str
    anchor_count: int
    candidate_edge_count: int
    total_s: float
    fallback_used: bool
    valid: bool
    status: str
