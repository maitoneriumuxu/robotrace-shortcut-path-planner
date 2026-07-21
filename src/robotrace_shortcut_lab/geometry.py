from __future__ import annotations

import numpy as np


STRAIGHT_RADIUS_MM = np.float32(999_999.0)


def cumulative_distance_mm(x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
    step = np.hypot(np.diff(x_mm), np.diff(y_mm)).astype(np.float32)
    return np.concatenate((np.zeros(1, dtype=np.float32), np.cumsum(step, dtype=np.float32)))


def path_length_mm(x_mm: np.ndarray, y_mm: np.ndarray) -> float:
    return float(cumulative_distance_mm(x_mm, y_mm)[-1])


def radius_mm(x_mm: np.ndarray, y_mm: np.ndarray, window: int = 20) -> np.ndarray:
    """ファームと同じ前後window点の3点円から半径を求める。"""

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


def signed_curvature_per_m(x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
    """経路の符号付き曲率κ[1/m]を求める。"""

    x = np.asarray(x_mm, dtype=np.float64) * 0.001
    y = np.asarray(y_mm, dtype=np.float64) * 0.001
    s = cumulative_distance_mm(x_mm, y_mm).astype(np.float64) * 0.001
    dx, dy = np.gradient(x, s), np.gradient(y, s)
    ddx, ddy = np.gradient(dx, s), np.gradient(dy, s)
    denominator = np.power(dx * dx + dy * dy, 1.5)
    return np.divide(
        dx * ddy - dy * ddx,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1.0e-9,
    ).astype(np.float32)


def curvature_slew_per_m2(x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
    """曲率変化率dκ/ds[1/m²]。一定速度では角加速度をv²倍で与える。"""

    s = cumulative_distance_mm(x_mm, y_mm).astype(np.float64) * 0.001
    curvature = signed_curvature_per_m(x_mm, y_mm).astype(np.float64)
    slew = np.gradient(curvature, s)
    slew[~np.isfinite(slew)] = 0.0
    return slew.astype(np.float32)


def quintic_hermite(
    p0: float,
    p1: float,
    v0: float,
    v1: float,
    a0: float,
    a1: float,
    u: np.ndarray,
) -> np.ndarray:
    """位置・一次微分・二次微分を両端で一致させる5次接続。"""

    c0, c1, c2 = p0, v0, a0 * 0.5
    delta_p = p1 - c0 - c1 - c2
    delta_v = v1 - c1 - 2.0 * c2
    delta_a = a1 - 2.0 * c2
    c3 = 10.0 * delta_p - 4.0 * delta_v + 0.5 * delta_a
    c4 = -15.0 * delta_p + 7.0 * delta_v - delta_a
    c5 = 6.0 * delta_p - 3.0 * delta_v + 0.5 * delta_a
    return c0 + c1 * u + c2 * u**2 + c3 * u**3 + c4 * u**4 + c5 * u**5
