#pragma once

#include <stdint.h>

namespace fdrtsp {

    struct video_render_interface {
        virtual int image_width() = 0;
        virtual int image_height() = 0;
        virtual int set_hwnd(void *hwnd) = 0;
        virtual int draw_yuv420(const uint8_t *const data[], int const linesize[], int w, int h) = 0;
        virtual int draw_rect(int x, int y, int w, int h) = 0;
        virtual int resize(int x, int y, int w, int h) = 0;
    };

}
