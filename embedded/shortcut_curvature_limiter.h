#ifndef SHORTCUT_CURVATURE_LIMITER_H
#define SHORTCUT_CURVATURE_LIMITER_H

#define SCL_MAX_POINT_COUNT       6100
#define SCL_SOLVE_POINT_COUNT     161
#define SCL_RESTORE_POINT_COUNT   161
#define SCL_WORK_BYTES            5160

typedef float (*scl_read_point)(void *context, int index);
typedef void (*scl_write_point)(void *context, int index, float x_mm, float y_mm);

typedef struct {
    void *context;
    scl_read_point source_x;
    scl_read_point source_y;
    scl_read_point path_x;
    scl_read_point path_y;
    scl_write_point write_path;
} scl_path_access;

typedef struct {
    float max_offset_mm;
    float min_radius_mm;
    float target_slew_per_m2;
    int radius_check_offset;
    int max_pass_count;
} scl_config;

typedef struct {
    float factor[4][SCL_SOLVE_POINT_COUNT];
    float original_x[SCL_RESTORE_POINT_COUNT];
    float original_y[SCL_RESTORE_POINT_COUNT];
    float vector[SCL_SOLVE_POINT_COUNT];
    float intermediate[SCL_SOLVE_POINT_COUNT];
} scl_work;

typedef struct {
    float before_peak_per_m2;
    float after_peak_per_m2;
    int before_peak_index;
    int after_peak_index;
    int accepted_count;
} scl_result;

// 既存経路へ最大dκ/dsの局所後処理を行う。heapと全点作業配列は使わない。
int scl_limit_curvature_slew(
    const scl_path_access *path,
    int point_count,
    const scl_config *config,
    scl_work *work,
    scl_result *result);

#endif
