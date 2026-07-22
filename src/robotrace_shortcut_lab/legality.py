from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .model import (
    BoardBoundary,
    ContactEvaluation,
    Course,
    GlobalPath,
    PlannerConfig,
    VehicleFootprint,
)


FOOTPRINT_FILE = Path("data/vehicle/LN5_footprint.json")


def load_vehicle_footprint(
    path: str | Path = FOOTPRINT_FILE,
) -> VehicleFootprint:
    """実車外形を読む。confirmedでも頂点が不正な場合は未確認へ降格する。"""

    source_path = Path(path)
    if not source_path.exists():
        return VehicleFootprint(
            (),
            "未設定",
            None,
            None,
            None,
            None,
            0.0,
            f"{source_path}が存在しない",
            False,
        )
    data = json.loads(source_path.read_text(encoding="utf-8"))
    raw_components = data.get("polygon_components_mm") or []
    if not raw_components and data.get("polygon_vertices_mm"):
        raw_components = [data["polygon_vertices_mm"]]
    components: list[np.ndarray] = []
    for raw in raw_components:
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
    requested_confirmed = bool(data.get("confirmed", False))
    confirmed = requested_confirmed and bool(components)

    def optional_float(name: str) -> float | None:
        value = data.get(name)
        return None if value is None else float(value)

    source = str(data.get("source", source_path))
    if requested_confirmed and not confirmed:
        source += "（confirmed=trueだが有効な外形頂点がない）"
    return VehicleFootprint(
        tuple(components),
        str(data.get("origin_definition", "未設定")),
        optional_float("front_mm"),
        optional_float("rear_mm"),
        optional_float("left_mm"),
        optional_float("right_mm"),
        float(data.get("safety_margin_mm", 0.0)),
        source,
        confirmed,
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
) -> np.ndarray | None:
    """実際に接触したsegmentだけで単調な進行列をDP選択する。"""

    if not contact_segments or any(not segments for segments in contact_segments):
        return None
    first = tuple(sorted(set(contact_segments[0])))
    if start_segment is not None:
        first = tuple(segment for segment in first if segment == start_segment)
    if not first:
        return None
    states = {segment: 0.0 for segment in first}
    parents: list[dict[int, int]] = [{segment: -1 for segment in first}]
    previous_set = set(first)
    local_jump = config.contact_dp_local_jump_segments
    for pose_index in range(1, len(contact_segments)):
        current = tuple(sorted(set(contact_segments[pose_index])))
        current_set = set(current)
        next_states: dict[int, float] = {}
        next_parent: dict[int, int] = {}
        for segment in current:
            best_cost = float("inf")
            best_previous = -1
            for previous, previous_cost in states.items():
                if segment < previous:
                    continue
                jump = segment - previous
                simultaneous = (
                    previous in current_set or segment in previous_set
                )
                if jump > local_jump and not simultaneous:
                    continue
                cost = previous_cost + abs(jump - 0.2) + (20.0 if jump > local_jump else 0.0)
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
        previous_set = current_set
    if end_segment is not None:
        if end_segment not in states:
            return None
        selected = end_segment
    else:
        selected = min(states, key=lambda segment: (states[segment], -segment))
    progress = np.empty(len(contact_segments), dtype=np.float32)
    for pose_index in range(len(contact_segments) - 1, -1, -1):
        progress[pose_index] = float(selected)
        selected = parents[pose_index][selected]
    return progress


class WhiteLineContactEvaluator:
    """Shapelyのカプセル和集合で実車外形と白線の連続接触を調べる。"""

    def __init__(
        self,
        course: Course,
        footprint: VehicleFootprint,
        config: PlannerConfig,
        boundary: BoardBoundary | None = None,
    ) -> None:
        if not footprint.confirmed or not footprint.polygon_components_mm:
            raise ValueError("LN5実車外形が未確認です")
        self.course = course
        self.footprint = footprint
        self.config = config
        self.local_components = tuple(
            np.asarray(component, dtype=np.float64)
            for component in footprint.polygon_components_mm
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

    def vehicle_geometry(self, x_mm: float, y_mm: float, yaw_rad: float):
        cosine = float(np.cos(yaw_rad))
        sine = float(np.sin(yaw_rad))
        polygons = []
        for component in self.local_components:
            # 外形JSONは+X=前方、+Y=左方。経路yawは世界+X基準。
            world_x = x_mm + cosine * component[:, 0] - sine * component[:, 1]
            world_y = y_mm + sine * component[:, 0] + cosine * component[:, 1]
            polygons.append(Polygon(np.column_stack((world_x, world_y))))
        return unary_union(polygons)

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
    ) -> ContactEvaluation:
        pose_distance, pose_x, pose_y, pose_yaw = resample_swept_poses(
            x_mm, y_mm, yaw_rad, self.config
        )
        contacts: list[tuple[int, ...]] = []
        overlap_area = np.zeros(pose_x.size, dtype=np.float32)
        penetration = np.zeros(pose_x.size, dtype=np.float32)
        boundary_length = np.zeros(pose_x.size, dtype=np.float32)
        contact_margin = np.full(pose_x.size, -np.inf, dtype=np.float32)
        simultaneous = np.zeros(pose_x.size, dtype=np.int16)
        board_outside = 0
        for index, (x, y, yaw) in enumerate(
            zip(pose_x, pose_y, pose_yaw, strict=True)
        ):
            vehicle = self.vehicle_geometry(float(x), float(y), float(yaw))
            if self.board_region is not None and not self.board_region.covers(vehicle):
                board_outside += 1
            candidate_indices = self.segment_tree.query(vehicle, predicate="intersects")
            segments = tuple(sorted(int(value) for value in candidate_indices))
            contacts.append(segments)
            simultaneous[index] = self._contact_group_count(segments)
            if not segments:
                continue
            intersection = vehicle.intersection(self.white_region)
            overlap_area[index] = float(intersection.area)
            boundary_length[index] = float(vehicle.boundary.intersection(self.white_region).length)
            centerline_distance = min(
                float(vehicle.distance(self.segment_lines[segment]))
                for segment in segments
            )
            margin = self.config.white_line_half_width_mm - centerline_distance
            penetration[index] = max(0.0, margin)
            contact_margin[index] = margin
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
        )
        violations: list[str] = []
        if detachment_count:
            violations.append(f"白線完全離脱{detachment_count}区間")
        if progress is None and not detachment_count:
            violations.append("単調な実接触segment列が不成立")
        if board_outside:
            violations.append(f"実車外形が板外{board_outside}姿勢")
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
        switches = int(np.count_nonzero(np.diff(progress) > 1.0))
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
            min_area,
            min_penetration,
            min_margin,
            min_margin_index,
            "、".join(violations),
            "、".join(warnings),
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
