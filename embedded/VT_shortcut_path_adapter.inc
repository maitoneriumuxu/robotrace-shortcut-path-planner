// VT_shortcut_path.c内でshortcut_curvature_limiter.cの後に読み込む移植用アダプタ。
static scl_work shortcut_slew_work;
static scl_result shortcut_slew_result;
static volatile unsigned long shortcut_slew_exec_us;

static float shortcut_slew_source_x(void *context, int index)
{
    (void)context;
    return VT_shortcut_path_buf[index].work.source.source_x;
}

static float shortcut_slew_source_y(void *context, int index)
{
    (void)context;
    return VT_shortcut_path_buf[index].work.source.source_y;
}

static float shortcut_slew_path_x(void *context, int index)
{
    (void)context;
    return VT_shortcut_path_buf[index].x;
}

static float shortcut_slew_path_y(void *context, int index)
{
    (void)context;
    return VT_shortcut_path_buf[index].y;
}

static void shortcut_slew_write(void *context, int index, float x_mm, float y_mm)
{
    (void)context;
    VT_shortcut_path_buf[index].x = x_mm;
    VT_shortcut_path_buf[index].y = y_mm;
}

// Elastic Band生成後、shortcut_finalize_path()より前に一度だけ呼ぶ。
static void shortcut_apply_slew_limit(int max_index)
{
    scl_path_access access;
    scl_config config;
    unsigned long start_time = us10_counter;

    access.context = 0;
    access.source_x = shortcut_slew_source_x;
    access.source_y = shortcut_slew_source_y;
    access.path_x = shortcut_slew_path_x;
    access.path_y = shortcut_slew_path_y;
    access.write_path = shortcut_slew_write;
    config.max_offset_mm = 100.0f;
    config.min_radius_mm = 60.0f;
    config.target_slew_per_m2 = 150.0f;
    config.radius_check_offset = VT_SHORTCUT_RADIUS_CHECK_OFFSET;
    config.max_pass_count = 12;
    usb_serial_task();
    scl_limit_curvature_slew(
        &access,
        max_index + 1,
        &config,
        &shortcut_slew_work,
        &shortcut_slew_result);
    shortcut_slew_exec_us = (us10_counter - start_time) * 10UL;
    usb_serial_task();
}
