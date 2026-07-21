#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include "shortcut_curvature_limiter.h"

#ifndef SCL_HOST_MAX_PASS_COUNT
#define SCL_HOST_MAX_PASS_COUNT (12)
#endif

static uint64_t sqrt_count;

#ifdef __TINYC__
// Windows版TinyCCのCRTにはsqrtfエクスポートがないため、ホスト検証時だけ補う。
float sqrtf(float value)
{
    sqrt_count++;
    return (float)sqrt((double)value);
}
#endif

typedef struct {
    float source_x[SCL_MAX_POINT_COUNT];
    float source_y[SCL_MAX_POINT_COUNT];
    float path_x[SCL_MAX_POINT_COUNT];
    float path_y[SCL_MAX_POINT_COUNT];
} host_path;

static host_path points;
static scl_work work_area;
static uint64_t read_count;
static uint64_t write_count;

static float read_source_x(void *context, int index)
{
    read_count++;
    return ((host_path *)context)->source_x[index];
}

static float read_source_y(void *context, int index)
{
    read_count++;
    return ((host_path *)context)->source_y[index];
}

static float read_path_x(void *context, int index)
{
    read_count++;
    return ((host_path *)context)->path_x[index];
}

static float read_path_y(void *context, int index)
{
    read_count++;
    return ((host_path *)context)->path_y[index];
}

static void write_path(void *context, int index, float x_mm, float y_mm)
{
    host_path *path = (host_path *)context;

    write_count++;
    path->path_x[index] = x_mm;
    path->path_y[index] = y_mm;
}

static int read_array(FILE *file, float *values, int count)
{
    return fread(values, sizeof(float), (size_t)count, file) == (size_t)count;
}

static int write_array(FILE *file, const float *values, int count)
{
    return fwrite(values, sizeof(float), (size_t)count, file) == (size_t)count;
}

int main(int argc, char **argv)
{
    FILE *input;
    FILE *output;
    int32_t count;
    scl_path_access access;
    scl_config config;
    scl_result result;

    if (argc != 3) {
        return 2;
    }
    input = fopen(argv[1], "rb");
    if (input == NULL) {
        return 3;
    }
    if ((fread(&count, sizeof(count), 1, input) != 1) ||
        (count < 201) || (SCL_MAX_POINT_COUNT < count) ||
        !read_array(input, points.source_x, count) ||
        !read_array(input, points.source_y, count) ||
        !read_array(input, points.path_x, count) ||
        !read_array(input, points.path_y, count)) {
        fclose(input);
        return 4;
    }
    fclose(input);

    access.context = &points;
    access.source_x = read_source_x;
    access.source_y = read_source_y;
    access.path_x = read_path_x;
    access.path_y = read_path_y;
    access.write_path = write_path;
    config.max_offset_mm = 100.0f;
    config.min_radius_mm = 60.0f;
    config.target_slew_per_m2 = 150.0f;
    config.radius_check_offset = 20;
    config.max_pass_count = SCL_HOST_MAX_PASS_COUNT;
    if (!scl_limit_curvature_slew(&access, count, &config, &work_area, &result)) {
        return 5;
    }

    output = fopen(argv[2], "wb");
    if (output == NULL) {
        return 6;
    }
    if ((fwrite(&count, sizeof(count), 1, output) != 1) ||
        (fwrite(&result.before_peak_per_m2, sizeof(float), 1, output) != 1) ||
        (fwrite(&result.after_peak_per_m2, sizeof(float), 1, output) != 1) ||
        (fwrite(&result.before_peak_index, sizeof(int), 1, output) != 1) ||
        (fwrite(&result.after_peak_index, sizeof(int), 1, output) != 1) ||
        (fwrite(&result.accepted_count, sizeof(int), 1, output) != 1) ||
        (fwrite(&read_count, sizeof(read_count), 1, output) != 1) ||
        (fwrite(&write_count, sizeof(write_count), 1, output) != 1) ||
        (fwrite(&sqrt_count, sizeof(sqrt_count), 1, output) != 1) ||
        !write_array(output, points.path_x, count) ||
        !write_array(output, points.path_y, count)) {
        fclose(output);
        return 7;
    }
    fclose(output);
    return 0;
}
