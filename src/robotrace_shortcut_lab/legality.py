from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .model import (
    BoardBoundary,
    ContactEvaluation,
    ContactSensitivity,
    Course,
    GlobalPath,
    PlannerConfig,
    VehicleFootprint,
)


FOOTPRINT_FILE = Path("data/vehicle/LN5_footprint.json")


def load_vehicle_footprint(
    path: str | Path = FOOTPRINT_FILE,
) -> VehicleFootprint:
    """全車体外形と接触証明部品を混同せずに読む。"""

    source_path = Path(path)
    if not source_path.exists():
        return VehicleFootprint(
            full_footprint_components_mm=(),
            contact_witness_components_mm=(),
            origin_definition="未設定",
            board_clearance_radius_mm=125.0,
            safety_margin_mm=0.0,
            source=f"{source_path}が存在しない",
            full_footprint_source="未設定",
            contact_witness_source="未設定",
            design_confirmed=False,
            as_built_confirmed=False,
        )
    data = json.loads(source_path.read_text(encoding="utf-8"))

    def valid_components(name: str) -> tuple[np.ndarray, ...]:
        components: list[np.ndarray] = []
        for raw in data.get(name, ()):
            vertices = np.asarray(raw, dtype=np.float64)
            if (
                vertices.ndim != 2
                or vertices.shape[0] < 3
                or vertices.shape[1] != 2
                or not np.isfinite(vertices).all()
            ):
                continue
            polygon = Polygon(vertices)
            if polygon.is_valid and polygon.area > 1.0e-6:
                components.append(vertices.astype(np.float32))
        return tuple(components)

    full = valid_components("full_footprint_components_mm")
    witness = valid_components("contact_witness_components_mm")
    design_confirmed = bool(data.get("design_confirmed", False)) and bool(witness)
    as_built_confirmed = (
        bool(data.get("as_built_confirmed", False)) and bool(full) and bool(witness)
    )
    return VehicleFootprint(
        full_footprint_components_mm=full,
        contact_witness_components_mm=witness,
        origin_definition=str(data.get("origin_definition", "未設定")),
        board_clearance_radius_mm=float(data.get("board_clearance_radius_mm", 125.0)),
        safety_margin_mm=float(data.get("safety_margin_mm", 0.0)),
        source=str(data.get("source", source_path)),
        full_footprint_source=str(data.get("full_footprint_source", "未設定")),
        contact_witness_source=str(data.get("contact_witness_source", "未設定")),
        design_confirmed=design_confirmed,
        as_built_confirmed=as_built_confirmed,
    )


def resample_swept_poses(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    yaw_rad: np.ndarray,
    config: PlannerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """並進2 mm、yaw 1 degの両方を上限に中間姿勢を固定補間する。"""

    x = np.asarray(x_mm, dtype=np.float64)
    y = np.asarray(y_mm, dtype=np.float64)
    yaw = np.unwrap(np.asarray(yaw_rad, dtype=np.float64))
    if not (x.size == y.size == yaw.size) or x.size < 2:
        raise ValueError("経路姿勢点列の長さが不正です")
    pose_x: list[float] = [float(x[0])]
    pose_y: list[float] = [float(y[0])]
    pose_yaw: list[float] = [float(yaw[0])]
    pose_distance: list[float] = [0.0]
    total = 0.0
    max_yaw_step = np.deg2rad(config.legal_yaw_step_deg)
    for index in range(x.size - 1):
        dx = float(x[index + 1] - x[index])
        dy = float(y[index + 1] - y[index])
        distance = float(np.hypot(dx, dy))
        yaw_delta = float(yaw[index + 1] - yaw[index])
        subdivisions = max(
            1,
            int(np.ceil(distance / config.legal_pose_step_mm)),
            int(np.ceil(abs(yaw_delta) / max(max_yaw_step, 1.0e-9))),
        )
        for step in range(1, subdivisions + 1):
            ratio = step / subdivisions
            pose_x.append(float(x[index] + ratio * dx))
            pose_y.append(float(y[index] + ratio * dy))
            pose_yaw.append(float(yaw[index] + ratio * yaw_delta))
            pose_distance.append(total + ratio * distance)
        total += distance
    return (
        np.asarray(pose_distance, dtype=np.float32),
        np.asarray(pose_x, dtype=np.float32),
        np.asarray(pose_y, dtype=np.float32),
        np.asarray(pose_yaw, dtype=np.float32),
    )


def solve_contact_progress(
    contact_segments: tuple[tuple[int, ...], ...],
    config: PlannerConfig,
    *,
    start_segment: int | None = None,
    end_segment: int | None = None,
    endpoint_tolerance_segments: int = 0,
) -> np.ndarray | None:
    """実際に接触したsegmentだけで単調な進行列をDP選択する。"""

    if not contact_segments or any(not segments for segments in contact_segments):
        return None
    first = tuple(sorted(set(contact_segments[0])))
    if start_segment is not None:
        first = tuple(
            segment
            for segment in first
            if abs(segment - start_segment) <= endpoint_tolerance_segments
        )
    if not first:
        return None
    states = {segment: 0.0 for segment in first}
    parents: list[dict[int, int]] = [{segment: -1 for segment in first}]
    max_step = config.contact_dp_max_progress_step_segments
    for pose_index in range(1, len(contact_segments)):
        current = tuple(sorted(set(contact_segments[pose_index])))
        current_set = set(current)
        transition_contact_set = current_set | set(contact_segments[pose_index - 1])
        next_states: dict[int, float] = {}
        next_parent: dict[int, int] = {}
        for segment in current:
            best_cost = float("inf")
            best_previous = -1
            for previous, previous_cost in states.items():
                if segment < previous:
                    continue
                jump = segment - previous
                if jump > max_step:
                    # 点列間隔が姿勢間隔より細かいコースでは、同一姿勢で
                    # 複数の連続segmentへ接触する。その全てが実接触集合に
                    # ある場合だけまとめて進め、未接触の中間は飛ばさない。
                    if not all(
                        required in transition_contact_set
                        for required in range(previous + 1, segment + 1)
                    ):
                        continue
                cost = previous_cost + abs(jump - 0.2)
                if cost < best_cost or (
                    abs(cost - best_cost) <= 1.0e-12 and previous > best_previous
                ):
                    best_cost = cost
                    best_previous = previous
            if best_previous >= 0:
                next_states[segment] = best_cost
                next_parent[segment] = best_previous
        if not next_states:
            return None
        states = next_states
        parents.append(next_parent)
    if end_segment is not None:
        end_candidates = [
            segment
            for segment in states
            if abs(segment - end_segment) <= endpoint_tolerance_segments
        ]
        if not end_candidates:
            return None
        selected = min(end_candidates, key=lambda segment: (states[segment], -segment))
    else:
        selected = min(states, key=lambda segment: (states[segment], -segment))
    progress = np.empty(len(contact_segments), dtype=np.float32)
    for pose_index in range(len(contact_segments) - 1, -1, -1):
        progress[pose_index] = float(selected)
        selected = parents[pose_index][selected]
    return progress


class WhiteLineContactEvaluator:
    """接触証明部品と白線、保守125mm円と板を別々に調べる。"""

    def __init__(
        self,
        course: Course,
        footprint: VehicleFootprint,
        config: PlannerConfig,
        boundary: BoardBoundary | None = None,
    ) -> None:
        if not footprint.design_confirmed or not footprint.contact_witness_components_mm:
            raise ValueError("白線接触証明部品が設計確認されていません")
        self.course = course
        self.footprint = footprint
        self.config = config
        self.local_components = tuple(
            np.asarray(component, dtype=np.float64)
            for component in footprint.contact_witness_components_mm
        )
        self.witness_max_radius_mm = max(
            float(np.max(np.hypot(component[:, 0], component[:, 1])))
            for component in self.local_components
        )
        coordinates = np.column_stack((course.x_mm, course.y_mm)).astype(np.float64)
        self.centerline = LineString(coordinates)
        self.white_region = self.centerline.buffer(
            config.white_line_half_width_mm,
            quad_segs=8,
            cap_style="round",
            join_style="round",
        )
        self.segment_lines = tuple(
            LineString((coordinates[index], coordinates[index + 1]))
            for index in range(coordinates.shape[0] - 1)
        )
        self.segment_regions = tuple(
            segment.buffer(
                config.white_line_half_width_mm,
                quad_segs=6,
                cap_style="round",
                join_style="round",
            )
            for segment in self.segment_lines
        )
        self.segment_tree = STRtree(self.segment_regions)
        self.board_region = (
            unary_union(
                [box(min_x, min_y, max_x, max_y) for min_x, max_x, min_y, max_y in boundary.rectangles_mm]
            )
            if boundary is not None
            else None
        )
        self.board_center_region = (
            self.board_region.buffer(
                -footprint.board_clearance_radius_mm,
                quad_segs=16,
                join_style="round",
            )
            if self.board_region is not None
            else None
        )

    def count_radially_unreachable_segments(
        self,
        x_mm: np.ndarray,
        y_mm: np.ndarray,
        start_segment: int,
        end_segment: int,
    ) -> int:
        """外接円でも届かない必須LINEを詳細姿勢判定前に数える。"""

        start = max(0, start_segment)
        end = min(len(self.segment_lines) - 1, end_segment)
        if end < start:
            return 0
        path_line = LineString(
            np.column_stack((x_mm.astype(np.float64), y_mm.astype(np.float64)))
        )
        required = np.asarray(self.segment_lines[start : end + 1], dtype=object)
        distances = np.asarray(shapely.distance(required, path_line), dtype=np.float64)
        reachable_radius = self.witness_max_radius_mm + self.config.white_line_half_width_mm
        return int(np.count_nonzero(distances > reachable_radius + 1.0e-9))

    def witness_geometry(
        self,
        x_mm: float,
        y_mm: float,
        yaw_rad: float,
        witness_offset_x_mm: float = 0.0,
        witness_offset_y_mm: float = 0.0,
        witness_yaw_error_rad: float = 0.0,
    ):
        cosine = float(np.cos(yaw_rad))
        sine = float(np.sin(yaw_rad))
        local_cosine = float(np.cos(witness_yaw_error_rad))
        local_sine = float(np.sin(witness_yaw_error_rad))
        polygons = []
        for component in self.local_components:
            local_x = (
                local_cosine * component[:, 0]
                - local_sine * component[:, 1]
                + witness_offset_x_mm
            )
            local_y = (
                local_sine * component[:, 0]
                + local_cosine * component[:, 1]
                + witness_offset_y_mm
            )
            # 外形JSONは+X=前方、+Y=左方。経路yawは世界+X基準。
            world_x = x_mm + cosine * local_x - sine * local_y
            world_y = y_mm + sine * local_x + cosine * local_y
            polygons.append(Polygon(np.column_stack((world_x, world_y))))
        return unary_union(polygons)

    def witness_geometries(
        self,
        x_mm: np.ndarray,
        y_mm: np.ndarray,
        yaw_rad: np.ndarray,
        witness_offset_x_mm: float = 0.0,
        witness_offset_y_mm: float = 0.0,
        witness_yaw_error_rad: float = 0.0,
    ):
        """全姿勢の接触証明部品をShapely 2配列演算で一括生成する。"""

        cosine = np.cos(yaw_rad.astype(np.float64))[:, None]
        sine = np.sin(yaw_rad.astype(np.float64))[:, None]
        local_cosine = float(np.cos(witness_yaw_error_rad))
        local_sine = float(np.sin(witness_yaw_error_rad))
        geometries = None
        for component in self.local_components:
            local_x = (
                local_cosine * component[:, 0]
                - local_sine * component[:, 1]
                + witness_offset_x_mm
            )
            local_y = (
                local_sine * component[:, 0]
                + local_cosine * component[:, 1]
                + witness_offset_y_mm
            )
            world_x = x_mm[:, None] + cosine * local_x[None, :] - sine * local_y[None, :]
            world_y = y_mm[:, None] + sine * local_x[None, :] + cosine * local_y[None, :]
            polygons = shapely.polygons(np.stack((world_x, world_y), axis=2))
            geometries = polygons if geometries is None else shapely.union(geometries, polygons)
        return geometries

    @staticmethod
    def _contact_group_count(segments: tuple[int, ...]) -> int:
        if not segments:
            return 0
        return 1 + sum(
            current > previous + 1
            for previous, current in zip(segments, segments[1:])
        )

    def evaluate(
        self,
        x_mm: np.ndarray,
        y_mm: np.ndarray,
        yaw_rad: np.ndarray,
        *,
        start_segment: int | None = None,
        end_segment: int | None = None,
        endpoint_tolerance_segments: int = 0,
        witness_offset_x_mm: float = 0.0,
        witness_offset_y_mm: float = 0.0,
        witness_yaw_error_rad: float = 0.0,
        check_board: bool = True,
        compute_metrics: bool = True,
    ) -> ContactEvaluation:
        pose_distance, pose_x, pose_y, pose_yaw = resample_swept_poses(
            x_mm, y_mm, yaw_rad, self.config
        )
        vehicles = self.witness_geometries(
            pose_x,
            pose_y,
            pose_yaw,
            witness_offset_x_mm,
            witness_offset_y_mm,
            witness_yaw_error_rad,
        )
        contact_lists: list[list[int]] = [[] for _ in range(pose_x.size)]
        pairs = self.segment_tree.query(vehicles, predicate="intersects")
        if pairs.size:
            for pose_index, segment_index in zip(pairs[0], pairs[1], strict=True):
                contact_lists[int(pose_index)].append(int(segment_index))
        contacts = [tuple(sorted(set(segments))) for segments in contact_lists]
        simultaneous = np.asarray(
            [self._contact_group_count(segments) for segments in contacts],
            dtype=np.int16,
        )
        has_contact = np.asarray([bool(segments) for segments in contacts], dtype=np.bool_)
        if compute_metrics:
            intersections = shapely.intersection(vehicles, self.white_region)
            overlap_area = np.asarray(shapely.area(intersections), dtype=np.float32)
            boundary_length = np.asarray(
                shapely.length(
                    shapely.intersection(shapely.boundary(vehicles), self.white_region)
                ),
                dtype=np.float32,
            )
            centerline_distance = np.asarray(
                shapely.distance(vehicles, self.centerline), dtype=np.float32
            )
            contact_margin = (
                self.config.white_line_half_width_mm - centerline_distance
            ).astype(np.float32)
            contact_margin[~has_contact] = -np.inf
            penetration = np.maximum(contact_margin, 0.0).astype(np.float32)
        else:
            overlap_area = has_contact.astype(np.float32)
            boundary_length = np.zeros(pose_x.size, dtype=np.float32)
            contact_margin = np.where(has_contact, 1.0, -np.inf).astype(np.float32)
            penetration = has_contact.astype(np.float32)
        if check_board and self.board_center_region is not None:
            board_inside = np.asarray(
                shapely.covers(
                    self.board_center_region,
                    shapely.points(pose_x.astype(np.float64), pose_y.astype(np.float64)),
                ),
                dtype=np.bool_,
            )
            board_outside = int(np.count_nonzero(~board_inside))
        else:
            board_outside = 0
        contact_tuple = tuple(contacts)
        empty = np.asarray([not item for item in contact_tuple], dtype=np.bool_)
        detachment_count = int(
            np.count_nonzero(empty & np.concatenate(([True], ~empty[:-1])))
        )
        progress = solve_contact_progress(
            contact_tuple,
            self.config,
            start_segment=start_segment,
            end_segment=end_segment,
            endpoint_tolerance_segments=endpoint_tolerance_segments,
        )
        violations: list[str] = []
        if start_segment is not None and end_segment is not None:
            required_start = max(0, start_segment)
            required_end = min(len(self.segment_lines) - 1, end_segment)
            contacted = {
                segment
                for segments in contact_tuple
                for segment in segments
                if required_start <= segment <= required_end
            }
            required_count = max(required_end - required_start + 1, 0)
            unvisited_segment_count = required_count - len(contacted)
        else:
            unvisited_segment_count = 0
        all_line_segments_covered = unvisited_segment_count == 0
        if detachment_count:
            violations.append(f"白線完全離脱{detachment_count}区間")
        if not all_line_segments_covered:
            violations.append(f"未通過LINE segment {unvisited_segment_count}区間")
        if progress is None and not detachment_count:
            violations.append("全LINEを順番に通る実接触segment列が不成立")
        if board_outside:
            violations.append(f"規定最大半径125mm円が板外{board_outside}姿勢")
        legal = not violations
        if progress is None:
            progress = np.full(pose_x.size, -1.0, dtype=np.float32)
        past_count = np.zeros(pose_x.size, dtype=np.int16)
        current_count = np.zeros(pose_x.size, dtype=np.int16)
        future_count = np.zeros(pose_x.size, dtype=np.int16)
        for index, segments in enumerate(contact_tuple):
            selected = int(progress[index])
            if selected < 0:
                continue
            past_count[index] = sum(segment < selected for segment in segments)
            current_count[index] = sum(segment == selected for segment in segments)
            future_count[index] = sum(segment > selected for segment in segments)
        min_area = float(np.min(overlap_area)) if overlap_area.size else 0.0
        min_penetration = float(np.min(penetration)) if penetration.size else 0.0
        min_margin = float(np.min(contact_margin)) if contact_margin.size else float("-inf")
        min_margin_index = int(np.argmin(contact_margin)) if contact_margin.size else 0
        near_point = (
            (overlap_area < self.config.minimum_overlap_area_mm2)
            | (contact_margin < self.config.minimum_contact_margin_mm)
        )
        pose_step = np.diff(pose_distance, prepend=pose_distance[0])
        near_point_distance = float(np.sum(pose_step[near_point]))
        simultaneous_pose_count = int(np.count_nonzero(simultaneous >= 2))
        robust = bool(
            legal
            and min_area >= self.config.minimum_overlap_area_mm2
            and min_penetration >= self.config.minimum_penetration_mm
            and min_margin >= self.config.minimum_contact_margin_mm
        )
        warnings: list[str] = []
        if legal and not robust:
            warnings.append("接触余裕がrobust仮閾値未満")
        if not self.config.robust_thresholds_confirmed:
            warnings.append("robust閾値の実機根拠未確定")
        switches = sum(
            int(current - previous > 1)
            for previous, current, segments in zip(
                progress[:-1], progress[1:], contact_tuple[1:], strict=True
            )
            if not all(
                required in segments
                for required in range(int(previous) + 1, int(current) + 1)
            )
        )
        return ContactEvaluation(
            pose_distance,
            pose_x,
            pose_y,
            pose_yaw,
            contact_tuple,
            progress,
            overlap_area,
            penetration,
            boundary_length,
            contact_margin,
            simultaneous,
            past_count,
            current_count,
            future_count,
            legal,
            robust,
            detachment_count,
            switches,
            all_line_segments_covered,
            unvisited_segment_count,
            min_area,
            min_penetration,
            min_margin,
            near_point_distance,
            simultaneous_pose_count,
            min_margin_index,
            "、".join(violations),
            "、".join(warnings),
        )


def evaluate_contact_sensitivity(
    evaluator: WhiteLineContactEvaluator,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    yaw_rad: np.ndarray,
    config: PlannerConfig,
    *,
    start_segment: int = 0,
    end_segment: int | None = None,
) -> ContactSensitivity:
    """取付誤差は±X/±Y、yaw誤差は±角度の全組で合法性を調べる。"""

    if end_segment is None:
        end_segment = evaluator.course.point_count - 2
    tasks: list[tuple[str, int, float, float, float]] = []
    for error in config.robust_position_errors_mm:
        for offset_x, offset_y in (
            (error, 0.0),
            (-error, 0.0),
            (0.0, error),
            (0.0, -error),
        ):
            tasks.append(("position", len(tasks), offset_x, offset_y, 0.0))
    position_task_count = len(tasks)
    for error_index, error in enumerate(config.robust_yaw_errors_deg):
        for sign in (-1.0, 1.0):
            tasks.append(
                (
                    "yaw",
                    error_index,
                    0.0,
                    0.0,
                    float(np.deg2rad(sign * error)),
                )
            )

    def evaluate_variant(task: tuple[str, int, float, float, float]) -> bool:
        _, _, offset_x, offset_y, yaw_error = task
        return evaluator.evaluate(
                x_mm,
                y_mm,
                yaw_rad,
                start_segment=start_segment,
                end_segment=end_segment,
                witness_offset_x_mm=offset_x,
                witness_offset_y_mm=offset_y,
                witness_yaw_error_rad=yaw_error,
                check_board=False,
                compute_metrics=False,
            ).legal

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(evaluate_variant, tasks))
    position_results = [
        all(results[index * 4 : index * 4 + 4])
        for index in range(len(config.robust_position_errors_mm))
    ]
    yaw_results = [
        all(results[position_task_count + index * 2 : position_task_count + index * 2 + 2])
        for index in range(len(config.robust_yaw_errors_deg))
    ]
    position_errors = np.asarray(config.robust_position_errors_mm, dtype=np.float32)
    yaw_errors = np.asarray(config.robust_yaw_errors_deg, dtype=np.float32)
    position_legal = np.asarray(position_results, dtype=np.bool_)
    yaw_legal = np.asarray(yaw_results, dtype=np.bool_)
    position_2mm = bool(position_legal[np.where(position_errors == 2.0)[0][0]])
    yaw_1deg = bool(yaw_legal[np.where(yaw_errors == 1.0)[0][0]])
    return ContactSensitivity(
        position_errors,
        position_legal,
        yaw_errors,
        yaw_legal,
        position_2mm and yaw_1deg,
        len(tasks),
    )


def apply_contact_progress_to_path(
    path: GlobalPath,
    contact: ContactEvaluation,
    course: Course,
) -> GlobalPath:
    """10 mm経路点に最密の実接触姿勢のsegmentを割り当てる。"""

    nearest = np.searchsorted(
        contact.pose_distance_mm,
        path.cumulative_distance_mm,
        side="left",
    )
    nearest = np.clip(nearest, 0, contact.pose_distance_mm.size - 1)
    previous = np.maximum(nearest - 1, 0)
    use_previous = (
        np.abs(path.cumulative_distance_mm - contact.pose_distance_mm[previous])
        <= np.abs(path.cumulative_distance_mm - contact.pose_distance_mm[nearest])
    )
    nearest[use_previous] = previous[use_previous]
    progress = contact.source_progress_index[nearest].astype(np.float32)
    progress_int = np.clip(progress.astype(np.int32), 0, course.point_count - 1)
    source_distance = course.distance_mm[progress_int].astype(np.float32)
    return replace(
        path,
        source_progress_index=progress,
        source_progress_distance_mm=source_distance,
    )
