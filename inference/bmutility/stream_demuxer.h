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
// Created by hsyuan on 2019-02-22.
//

#ifndef BM_UTILITY_STREAM_DEMUXER_H
#define BM_UTILITY_STREAM_DEMUXER_H

#include <iostream>
#include <thread>
#include <list>
#include <functional>
#include "ffmpeg_global.h"

namespace bm {

    struct StreamDemuxerEvents {
        virtual void on_avformat_opened(AVFormatContext *ifmt_ctx) {}

        virtual void on_avformat_closed() {}

        virtual int on_read_frame(AVPacket *pkt) = 0;

        virtual void on_read_eof(AVPacket *pkt) = 0;
    };

    class StreamDemuxer : FfmpegGlobal {
    public:
        enum State : int8_t {
            Initialize = 0,
            Service,
            Down
        };

        using OnAVFormatOpenedFunc = std::function<void(AVFormatContext*)>;
        using OnAVFormatClosedFunc = std::function<void()>;
        using OnReadFrameFunc = std::function<void(AVPacket *)>;
        using OnReadEofFunc = std::function<void(AVPacket *)>;
    private:

        AVFormatContext *m_ifmt_ctx;
        StreamDemuxerEvents *m_observer;

        State m_work_state;
        std::string m_inputUrl;
        std::thread *m_thread_reading;
        bool m_repeat;
        bool m_keep_running;
        int64_t m_last_frame_time{0};
        int64_t m_start_time;
        bool m_is_file_url{false};
        int m_id;

        OnAVFormatOpenedFunc m_pfnOnAVFormatOpened;
        OnAVFormatClosedFunc m_pfnOnAVFormatClosed;
        OnReadFrameFunc m_pfnOnReadFrame;
        OnReadEofFunc m_pfnOnReadEof;
    protected:
        int do_initialize();
        int do_service();
        int do_down();

    public:
        StreamDemuxer(int id=0);
        virtual ~StreamDemuxer();

        void set_avformat_opend_callback(OnAVFormatOpenedFunc func) {
            m_pfnOnAVFormatOpened = func;
        }

        void set_avformat_closed_callback(OnAVFormatClosedFunc func){
            m_pfnOnAVFormatClosed = func;
        }

        void set_read_Frame_callback(OnReadFrameFunc func){
            m_pfnOnReadFrame = func;
        }

        void set_read_eof_callback(OnReadEofFunc func){
            m_pfnOnReadEof = func;
        }

        int open_stream(std::string url, StreamDemuxerEvents *observer, bool repeat = true, bool isSyncOpen=false);
        int close_stream(bool is_waiting);

        //int get_codec_parameters(int stream_index, AVCodecParameters **p_codecpar);
        //int get_codec_type(int stream_index, int *p_codec_type);
    };
}


#endif //TESTUV_STREAM_DEMUXER_H
