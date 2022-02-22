/*==========================================================================
 * Copyright 2016-2022 by Sophgo Technologies Inc. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
============================================================================*/
//
// Created by hsyuan on 2019-03-15.
//

#ifndef BM_UTILITY_FFMPEG_GLOBAL_H
#define BM_UTILITY_FFMPEG_GLOBAL_H

#ifdef __cplusplus
extern "C" {
#include "libavdevice/avdevice.h"
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
            avdevice_register_all();
            av_log_set_level(AV_LOG_ERROR);
        }

        ~FfmpegGlobal() {
            std::cout << "~FfmpegGlobal() dtor.." << std::endl;
            avformat_network_deinit();
        }
    };
}

#endif //FACEDEMOSYSTEM_FFMPEG_GLOBAL_H
