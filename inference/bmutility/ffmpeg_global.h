//
// Created by hsyuan on 2019-03-15.
//

#ifndef BM_UTILITY_FFMPEG_GLOBAL_H
#define BM_UTILITY_FFMPEG_GLOBAL_H

#ifdef __cplusplus
extern "C" {
#include "libavcodec/avcodec.h"
#include "libavutil/avutil.h"
#include "libavformat/avformat.h"
#include "libavutil/time.h"
}
#endif //!cplusplus

namespace bm {

    class FfmpegGlobal {
    public:
        FfmpegGlobal() {
            av_register_all();
            avformat_network_init();
            av_log_set_level(AV_LOG_DEBUG);
        }

        ~FfmpegGlobal() {
            std::cout << "~FfmpegGlobal() dtor.." << std::endl;
            avformat_network_deinit();
        }
    };
}

#endif //FACEDEMOSYSTEM_FFMPEG_GLOBAL_H
