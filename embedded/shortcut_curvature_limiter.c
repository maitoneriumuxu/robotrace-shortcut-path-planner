#include <math.h>
#include <stddef.h>

#include "shortcut_curvature_limiter.h"

#ifndef SCL_TEARDROP_CENTER
#define SCL_TEARDROP_CENTER (69)
#endif

#ifndef SCL_TEARDROP_SHIFT_MM
#define SCL_TEARDROP_SHIFT_MM (94.0f)
#endif

#ifndef SCL_TEARDROP_MAX_SLEW_RATIO
#define SCL_TEARDROP_MAX_SLEW_RATIO (1.0f)
#endif

#ifndef SCL_TEARDROP_PASS_LIMIT
#define SCL_TEARDROP_PASS_LIMIT (2)
#endif

#define SCL_EDGE_WEIGHT       100000.0f
#ifndef SCL_REGULARIZATION
#define SCL_REGULARIZATION    (10000.0f)
#endif
#define SCL_EPSILON           0.001f

typedef char scl_work_size_check[
    (sizeof(scl_work) <= SCL_WORK_BYTES) ? 1 : -1];

static float scl_abs(float value)
{
    return (value < 0.0f) ? -value : value;
}

static float scl_distance(
    const scl_path_access *path,
    int first,
    int second)
{
    float dx = path->path_x(path->context, second) - path->path_x(path->context, first);
    float dy = path->path_y(path->context, second) - path->path_y(path->context, first);

    return sqrtf(dx * dx + dy * dy);
}

static float scl_curvature_per_m(const scl_path_access *path, int index)
{
    float x0 = path->path_x(path->context, index - 1);
    float y0 = path->path_y(path->context, index - 1);
    float x1 = path->path_x(path->context, index);
    float y1 = path->path_y(path->context, index);
    float x2 = path->path_x(path->context, index + 1);
    float y2 = path->path_y(path->context, index + 1);
    float dx01 = x1 - x0;
    float dy01 = y1 - y0;
    float dx12 = x2 - x1;
    float dy12 = y2 - y1;
    float dx20 = x0 - x2;
    float dy20 = y0 - y2;
    float cross = dx01 * (y2 - y0) - dy01 * (x2 - x0);
    float a = sqrtf(dx01 * dx01 + dy01 * dy01);
    float b = sqrtf(dx12 * dx12 + dy12 * dy12);
    float c = sqrtf(dx20 * dx20 + dy20 * dy20);

    if ((a <= SCL_EPSILON) || (b <= SCL_EPSILON) || (c <= SCL_EPSILON)) {
        return 0.0f;
    }
    return 2000.0f * cross / (a * b * c);
}

static float scl_find_peak(
    const scl_path_access *path,
    int point_count,
    int *peak_index)
{
    int index;
    int start = 100;
    int end = point_count - 100;
    float previous;
    float maximum = 0.0f;

    *peak_index = start;
    previous = scl_curvature_per_m(path, start - 1);
    for (index = start; index < end; index++) {
        float curvature = scl_curvature_per_m(path, index);
        float step_m = scl_distance(path, index - 1, index) * 0.001f;
        float slew;

        if (step_m <= SCL_EPSILON * 0.001f) {
            previous = curvature;
            continue;
        }
        slew = scl_abs(curvature - previous) / step_m;
        if (maximum < slew) {
            maximum = slew;
            *peak_index = index;
        }
        previous = curvature;
    }
    return maximum;
}

static float scl_radius_mm(
    const scl_path_access *path,
    int index,
    int offset)
{
    float x0 = path->path_x(path->context, index - offset);
    float y0 = path->path_y(path->context, index - offset);
    float x1 = path->path_x(path->context, index);
    float y1 = path->path_y(path->context, index);
    float x2 = path->path_x(path->context, index + offset);
    float y2 = path->path_y(path->context, index + offset);
    float dx01 = x1 - x0;
    float dy01 = y1 - y0;
    float dx12 = x2 - x1;
    float dy12 = y2 - y1;
    float dx20 = x0 - x2;
    float dy20 = y0 - y2;
    float cross = dx01 * (y2 - y0) - dy01 * (x2 - x0);
    float a;
    float b;
    float c;

    if (scl_abs(cross) <= SCL_EPSILON) {
        return 999999.0f;
    }
    a = sqrtf(dx01 * dx01 + dy01 * dy01);
    b = sqrtf(dx12 * dx12 + dy12 * dy12);
    c = sqrtf(dx20 * dx20 + dy20 * dy20);
    if ((a <= SCL_EPSILON) || (b <= SCL_EPSILON) || (c <= SCL_EPSILON)) {
        return 0.0f;
    }
    return a * b * c / (2.0f * scl_abs(cross));
}

static int scl_path_is_valid(
    const scl_path_access *path,
    int point_count,
    const scl_config *config)
{
    int index;
    float offset_limit2 = config->max_offset_mm * config->max_offset_mm;

    for (index = 0; index < point_count; index++) {
        float dx = path->path_x(path->context, index) - path->source_x(path->context, index);
        float dy = path->path_y(path->context, index) - path->source_y(path->context, index);

        if (offset_limit2 + 0.01f < dx * dx + dy * dy) {
            return 0;
        }
    }
    for (index = config->radius_check_offset;
         index < point_count - config->radius_check_offset;
         index++) {
        float radius = scl_radius_mm(path, index, config->radius_check_offset);

        if ((0.0f < radius) && (radius < config->min_radius_mm)) {
            return 0;
        }
    }
    return 1;
}

static void scl_save_window(
    const scl_path_access *path,
    int start,
    int count,
    scl_work *work)
{
    int index;

    for (index = 0; index < count; index++) {
        work->original_x[index] = path->path_x(path->context, start + index);
        work->original_y[index] = path->path_y(path->context, start + index);
    }
}

static void scl_restore_window(
    const scl_path_access *path,
    int start,
    int count,
    const scl_work *work)
{
    int index;

    for (index = 0; index < count; index++) {
        path->write_path(
            path->context,
            start + index,
            work->original_x[index],
            work->original_y[index]);
    }
}

static float scl_matrix_value(int distance, int row, int count)
{
    if (distance == 0) {
        if ((row == 0) || (row == count - 1)) {
            return 1.0f;
        }
        if ((row == 1) || (row == count - 2)) {
            return 10.0f;
        }
        if ((row == 2) || (row == count - 3)) {
            return 19.0f;
        }
        return 20.0f;
    }
    if (distance == 1) {
        if ((row == 1) || (row == count - 1)) {
            return -3.0f;
        }
        if ((row == 2) || (row == count - 2)) {
            return -12.0f;
        }
        return -15.0f;
    }
    if (distance == 2) {
        if ((row == 2) || (row == count - 1)) {
            return 3.0f;
        }
        return 6.0f;
    }
    return -1.0f;
}

static void scl_build_factor(scl_work *work)
{
    int row;

    for (row = 0; row < SCL_SOLVE_POINT_COUNT; row++) {
        int distance;

        for (distance = 3; 0 <= distance; distance--) {
            int column = row - distance;
            float value;
            float product = 0.0f;
            int index;

            if (column < 0) {
                continue;
            }
            value = SCL_REGULARIZATION
                * scl_matrix_value(distance, row, SCL_SOLVE_POINT_COUNT);
            if (distance == 0) {
                value += ((row < 12) || (SCL_SOLVE_POINT_COUNT - 12 <= row))
                    ? SCL_EDGE_WEIGHT : 1.0f;
            }
            for (index = column - 3; index < column; index++) {
                if ((0 <= index) && (row - index <= 3)) {
                    product += work->factor[row - index][row]
                        * work->factor[column - index][column];
                }
            }
            if (distance == 0) {
                work->factor[0][row] = sqrtf(value - product);
            }
            else {
                work->factor[distance][row] =
                    (value - product) / work->factor[0][column];
            }
        }
    }
}

static void scl_solve(scl_work *work)
{
    int row;

    for (row = 0; row < SCL_SOLVE_POINT_COUNT; row++) {
        float product = 0.0f;
        int column;

        for (column = row - 3; column < row; column++) {
            if (0 <= column) {
                product += work->factor[row - column][row]
                    * work->intermediate[column];
            }
        }
        work->intermediate[row] =
            (work->vector[row] - product) / work->factor[0][row];
    }
    for (row = SCL_SOLVE_POINT_COUNT - 1; 0 <= row; row--) {
        float product = 0.0f;
        int column;

        for (column = row + 1;
             (column < SCL_SOLVE_POINT_COUNT) && (column <= row + 3);
             column++) {
            product += work->factor[column - row][column] * work->vector[column];
        }
        work->vector[row] =
            (work->intermediate[row] - product) / work->factor[0][row];
    }
}

static int scl_try_teardrop(
    const scl_path_access *path,
    int point_count,
    int peak,
    float before_peak,
    const scl_config *config,
    scl_work *work,
    float *after_peak,
    int *after_index)
{
    int start = peak - 100;
    int center = SCL_TEARDROP_CENTER;
    float tangent_x;
    float tangent_y;
    float tangent_length;
    float direction_x;
    float direction_y;
    int index;

    if ((start < 0) || (point_count < start + SCL_SOLVE_POINT_COUNT)) {
        return 0;
    }
    scl_save_window(path, start, SCL_SOLVE_POINT_COUNT, work);
    tangent_x = work->original_x[center + 1] - work->original_x[center - 1];
    tangent_y = work->original_y[center + 1] - work->original_y[center - 1];
    tangent_length = sqrtf(tangent_x * tangent_x + tangent_y * tangent_y);
    if (tangent_length <= SCL_EPSILON) {
        return 0;
    }
    direction_x = -tangent_x / tangent_length;
    direction_y = -tangent_y / tangent_length;

    for (index = 0; index < SCL_SOLVE_POINT_COUNT; index++) {
        float normalized = (float)(index - center) / 70.0f;
        float envelope = 0.0f;
        float weight = ((index < 12) || (SCL_SOLVE_POINT_COUNT - 12 <= index))
            ? SCL_EDGE_WEIGHT : 1.0f;

        if (scl_abs(normalized) < 1.0f) {
            envelope = 1.0f - normalized * normalized;
            envelope = envelope * envelope * envelope;
        }
        work->vector[index] = weight
            * (work->original_x[index]
                + envelope * SCL_TEARDROP_SHIFT_MM * direction_x);
    }
    scl_solve(work);
    for (index = 0; index < SCL_SOLVE_POINT_COUNT; index++) {
        path->write_path(
            path->context,
            start + index,
            work->vector[index],
            work->original_y[index]);
    }
    for (index = 0; index < SCL_SOLVE_POINT_COUNT; index++) {
        float normalized = (float)(index - center) / 70.0f;
        float envelope = 0.0f;
        float weight = ((index < 12) || (SCL_SOLVE_POINT_COUNT - 12 <= index))
            ? SCL_EDGE_WEIGHT : 1.0f;

        if (scl_abs(normalized) < 1.0f) {
            envelope = 1.0f - normalized * normalized;
            envelope = envelope * envelope * envelope;
        }
        work->vector[index] = weight
            * (work->original_y[index]
                + envelope * SCL_TEARDROP_SHIFT_MM * direction_y);
    }
    scl_solve(work);
    for (index = 0; index < SCL_SOLVE_POINT_COUNT; index++) {
        path->write_path(
            path->context,
            start + index,
            path->path_x(path->context, start + index),
            work->vector[index]);
    }
    *after_peak = scl_find_peak(path, point_count, after_index);
    if ((*after_peak < before_peak * SCL_TEARDROP_MAX_SLEW_RATIO)
        && scl_path_is_valid(path, point_count, config)) {
        return 1;
    }
    scl_restore_window(path, start, SCL_SOLVE_POINT_COUNT, work);
    return 0;
}

static int scl_try_regularization(
    const scl_path_access *path,
    int point_count,
    int peak,
    float before_peak,
    const scl_config *config,
    scl_work *work,
    float *after_peak,
    int *after_index)
{
    int start = peak - 80;
    int index;

    if ((start < 0) || (point_count < start + SCL_SOLVE_POINT_COUNT)) {
        return 0;
    }
    scl_save_window(path, start, SCL_SOLVE_POINT_COUNT, work);
    for (index = 0; index < SCL_SOLVE_POINT_COUNT; index++) {
        float weight = ((index < 12) || (SCL_SOLVE_POINT_COUNT - 12 <= index))
            ? SCL_EDGE_WEIGHT : 1.0f;
        work->vector[index] = weight * work->original_x[index];
    }
    scl_solve(work);
    for (index = 0; index < SCL_SOLVE_POINT_COUNT; index++) {
        path->write_path(
            path->context,
            start + index,
            work->vector[index],
            work->original_y[index]);
    }
    for (index = 0; index < SCL_SOLVE_POINT_COUNT; index++) {
        float weight = ((index < 12) || (SCL_SOLVE_POINT_COUNT - 12 <= index))
            ? SCL_EDGE_WEIGHT : 1.0f;
        work->vector[index] = weight * work->original_y[index];
    }
    scl_solve(work);
    for (index = 0; index < SCL_SOLVE_POINT_COUNT; index++) {
        path->write_path(
            path->context,
            start + index,
            path->path_x(path->context, start + index),
            work->vector[index]);
    }
    *after_peak = scl_find_peak(path, point_count, after_index);
    if ((*after_peak < before_peak) && scl_path_is_valid(path, point_count, config)) {
        return 1;
    }
    scl_restore_window(path, start, SCL_SOLVE_POINT_COUNT, work);
    return 0;
}

static int scl_try_source_relaxation(
    const scl_path_access *path,
    int point_count,
    int peak,
    float before_peak,
    const scl_config *config,
    scl_work *work,
    float *after_peak,
    int *after_index)
{
    int half_window = (300.0f < before_peak) ? 40 : 160;
    float blend = (300.0f < before_peak) ? 0.20f : 0.05f;
    int count = half_window * 2 + 1;
    int start = peak - half_window;
    int index;

    if ((start < 0) || (point_count < start + count)) {
        return 0;
    }
    if (count <= SCL_RESTORE_POINT_COUNT) {
        scl_save_window(path, start, count, work);
    }
    for (index = 0; index < count; index++) {
        float u = (float)(index - half_window) / (float)half_window;
        float envelope = 1.0f - u * u;
        float ratio;
        float original_x;
        float original_y;
        float x;
        float y;

        envelope = envelope * envelope * envelope;
        ratio = envelope * blend;
        original_x = path->path_x(path->context, start + index);
        original_y = path->path_y(path->context, start + index);
        x = original_x
            + (path->source_x(path->context, start + index) - original_x) * ratio;
        y = original_y
            + (path->source_y(path->context, start + index) - original_y) * ratio;
        path->write_path(path->context, start + index, x, y);
    }
    *after_peak = scl_find_peak(path, point_count, after_index);
    if ((*after_peak < before_peak) && scl_path_is_valid(path, point_count, config)) {
        return 1;
    }
    if (count <= SCL_RESTORE_POINT_COUNT) {
        scl_restore_window(path, start, count, work);
    }
    else {
        /* blend<1なので、追加配列を持たずに同じ式を逆算して元座標へ戻せる。 */
        for (index = 0; index < count; index++) {
            float u = (float)(index - half_window) / (float)half_window;
            float envelope = 1.0f - u * u;
            float ratio;
            float source_x;
            float source_y;
            float x;
            float y;

            envelope = envelope * envelope * envelope;
            ratio = envelope * blend;
            source_x = path->source_x(path->context, start + index);
            source_y = path->source_y(path->context, start + index);
            x = (path->path_x(path->context, start + index) - source_x * ratio)
                / (1.0f - ratio);
            y = (path->path_y(path->context, start + index) - source_y * ratio)
                / (1.0f - ratio);
            path->write_path(path->context, start + index, x, y);
        }
    }
    return 0;
}

//**********************************************************
// 最大曲率変化率の局所制限
//**********************************************************
// モード5の一括経路生成後に呼び、実走中や1ms割り込み内では呼ばない。
int scl_limit_curvature_slew(
    const scl_path_access *path,
    int point_count,
    const scl_config *config,
    scl_work *work,
    scl_result *result)
{
    int pass;
    int pass_limit;
    int peak_index;
    float peak;

    if ((path == NULL) || (config == NULL) || (work == NULL) || (result == NULL) ||
        (path->source_x == NULL) || (path->source_y == NULL) ||
        (path->path_x == NULL) || (path->path_y == NULL) ||
        (path->write_path == NULL) || (point_count < 201) ||
        (SCL_MAX_POINT_COUNT < point_count)) {
        return 0;
    }
    scl_build_factor(work);
    peak = scl_find_peak(path, point_count, &peak_index);
    result->before_peak_per_m2 = peak;
    result->before_peak_index = peak_index;
    result->accepted_count = 0;

    pass_limit = config->max_pass_count;
    if ((peak <= 300.0f) && (5 < pass_limit)) {
        pass_limit = 5;
    }
    else if ((peak <= 1000.0f) && (8 < pass_limit)) {
        pass_limit = 8;
    }

    for (pass = 0; pass < pass_limit; pass++) {
        float candidate_peak;
        int candidate_index;

        if (peak <= config->target_slew_per_m2) {
            break;
        }
        /* 涙滴補正は前半だけ。残りの反復はその形を崩さず平滑化する。 */
        if (((pass < SCL_TEARDROP_PASS_LIMIT) && scl_try_teardrop(
                path,
                point_count,
                peak_index,
                peak,
                config,
                work,
                &candidate_peak,
                &candidate_index)) ||
            scl_try_regularization(
                path,
                point_count,
                peak_index,
                peak,
                config,
                work,
                &candidate_peak,
                &candidate_index) ||
            scl_try_source_relaxation(
                path,
                point_count,
                peak_index,
                peak,
                config,
                work,
                &candidate_peak,
                &candidate_index)) {
            peak = candidate_peak;
            peak_index = candidate_index;
            result->accepted_count++;
        }
        else {
            break;
        }
    }
    result->after_peak_per_m2 = peak;
    result->after_peak_index = peak_index;
    return 1;
}
