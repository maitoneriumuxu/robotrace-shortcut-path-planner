from __future__ import annotations

import numpy as np


STRAIGHT_RADIUS_MM = np.float32(999_999.0)


def cumulative_distance_m(x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
    step_m = np.hypot(np.diff(x_mm), np.diff(y_mm)).astype(np.float64) * 0.001
    return np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(step_m)))


def path_length_m(x_mm: np.ndarray, y_mm: np.ndarray) -> float:
    return float(cumulative_distance_m(x_mm, y_mm)[-1])


def frenet_normals(x_mm: np.ndarray, y_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """原ラインの同一indexに固定した左法線を求める。"""

    dx = np.empty_like(x_mm, dtype=np.float32)
    dy = np.empty_like(y_mm, dtype=np.float32)
    dx[1:-1] = x_mm[2:] - x_mm[:-2]
    dy[1:-1] = y_mm[2:] - y_mm[:-2]
    dx[0], dy[0] = x_mm[1] - x_mm[0], y_mm[1] - y_mm[0]
    dx[-1], dy[-1] = x_mm[-1] - x_mm[-2], y_mm[-1] - y_mm[-2]
    length = np.hypot(dx, dy)
    length[length < 1.0e-6] = 1.0
    return (-dy / length).astype(np.float32), (dx / length).astype(np.float32)


def radius_mm(x_mm: np.ndarray, y_mm: np.ndarray, window: int = 20) -> np.ndarray:
    """現行ファーム相当の前後window点3点円半径を求める。"""

    x = np.asarray(x_mm, dtype=np.float32)
    y = np.asarray(y_mm, dtype=np.float32)
    result = np.full(x.size, STRAIGHT_RADIUS_MM, dtype=np.float32)
    if x.size < window * 2 + 1:
        return result

    x0, x1, x2 = x[:-2 * window], x[window:-window], x[2 * window :]
    y0, y1, y2 = y[:-2 * window], y[window:-window], y[2 * window :]
    cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    a = np.hypot(x1 - x0, y1 - y0)
    b = np.hypot(x2 - x1, y2 - y1)
    c = np.hypot(x0 - x2, y0 - y2)
    valid = (np.abs(cross) > 0.001) & (a > 0.001) & (b > 0.001) & (c > 0.001)
    middle = np.full(x1.size, STRAIGHT_RADIUS_MM, dtype=np.float32)
    middle[valid] = (a[valid] * b[valid] * c[valid] / (2.0 * np.abs(cross[valid]))).astype(
        np.float32
    )
    result[window:-window] = middle
    return result


def signed_curvature_per_m(
    x_mm: np.ndarray, y_mm: np.ndarray, window: int = 20
) -> np.ndarray:
    """固定幅3点円から速度評価用の符号付き曲率を求める。"""

    radius = radius_mm(x_mm, y_mm, window).astype(np.float64) * 0.001
    curvature = np.zeros(x_mm.size, dtype=np.float64)
    if x_mm.size < window * 2 + 1:
        return curvature
    x0, x1, x2 = x_mm[:-2 * window], x_mm[window:-window], x_mm[2 * window :]
    y0, y1, y2 = y_mm[:-2 * window], y_mm[window:-window], y_mm[2 * window :]
    cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    local_radius = radius[window:-window]
    curved = local_radius < 900.0
    middle = np.zeros_like(local_radius)
    middle[curved] = np.sign(cross[curved]) / local_radius[curved]
    curvature[window:-window] = middle
    return curvature


def curvature_slew_per_m2(x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
    distance = cumulative_distance_m(x_mm, y_mm)
    curvature = signed_curvature_per_m(x_mm, y_mm)
    # 10 mm点列の量子化ノイズを速度ペナルティへ直接入れないよう、
    # Cへ移植可能な固定21点移動平均に相当する平滑化を行う。
    curvature = np.convolve(curvature, np.ones(21) / 21.0, mode="same")
    slew = np.gradient(curvature, distance, edge_order=1)
    slew[~np.isfinite(slew)] = 0.0
    return slew


def expanded_mask(mask: np.ndarray, half_width: int) -> np.ndarray:
    kernel = np.ones(half_width * 2 + 1, dtype=np.int16)
    return np.convolve(mask.astype(np.int16), kernel, mode="same") > 0
