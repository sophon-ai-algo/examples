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
// Created by hsyuan on 2019-02-27.
//

#ifndef BMUTILITY_STREAM_DECODE_H
#define BMUTILITY_STREAM_DECODE_H

#include "stream_demuxer.h"

namespace bm {

#if LIBAVCODEC_VERSION_MAJOR <= 56
    static AVPacket *av_packet_alloc() {
        AVPacket* pkt = new AVPacket;
        av_init_packet(pkt);
        return pkt;
    }

    static void av_packet_free(AVPacket** pkt){
        av_free_packet(*pkt);
        av_freep(pkt);
    }
#endif

    struct StreamDecoderEvents {
        virtual ~StreamDecoderEvents() {}

        virtual void on_decoded_avframe(const AVPacket *pkt, const AVFrame *pFrame) = 0;

        virtual void on_decoded_sei_info(const uint8_t *sei_data, int sei_data_len, uint64_t pts, int64_t pkt_pos){};
        virtual void on_stream_eof() {};
    };



    class StreamDecoder : public StreamDemuxerEvents {
        StreamDecoderEvents *m_observer;

        using OnDecodedFrameCallback = std::function<void(const AVPacket *pkt, const AVFrame *pFrame)>;
        using OnDecodedSEICallback =std::function<void(const uint8_t *sei_data, int sei_data_len, uint64_t pts, int64_t pkt_pos)>;
        using OnStreamEofCallback = std::function<void()>;
        OnDecodedFrameCallback m_OnDecodedFrameFunc;
        OnDecodedSEICallback m_OnDecodedSEIFunc;

        StreamDemuxer::OnAVFormatOpenedFunc m_pfnOnAVFormatOpened;
        StreamDemuxer::OnAVFormatClosedFunc m_pfnOnAVFormatClosed;
        StreamDemuxer::OnReadFrameFunc m_pfnOnReadFrame;
        StreamDemuxer::OnReadEofFunc m_pfnOnReadEof;

    protected:
        std::list<AVPacket *> m_list_packets;
        AVCodecContext *m_dec_ctx{nullptr};
        AVCodecContext *m_external_dec_ctx {nullptr};
        int m_video_stream_index{0};
        int m_frame_decoded_num{0};
        StreamDemuxer m_demuxer;
        AVDictionary *m_opts_decoder{nullptr};
        bool m_is_waiting_iframe{true};
        int m_id{0};
        AVRational m_timebase;
        //Functions
        int create_video_decoder(AVFormatContext *ifmt_ctx);

        int put_packet(AVPacket *pkt);

        AVPacket *get_packet();

        void clear_packets();

        int decode_frame(AVPacket *pkt, AVFrame *pFrame);

        int get_video_stream_index(AVFormatContext *ifmt_ctx);
        bool is_key_frame(AVPacket *pkt);

        //
        //Overload StreamDemuxerEvents Interface.
        //
        virtual void on_avformat_opened(AVFormatContext *ifmt_ctx) override;

        virtual void on_avformat_closed() override;

        virtual int on_read_frame(AVPacket *pkt) override;

        virtual void on_read_eof(AVPacket *pkt) override;

    public:
        StreamDecoder(int id, AVCodecContext *decoder=nullptr);
        virtual ~StreamDecoder();

        int set_observer(StreamDecoderEvents *observer);

        void set_decoded_frame_callback(OnDecodedFrameCallback func)
        {
            m_OnDecodedFrameFunc = func;
        }

        void set_decoded_sei_info_callback(OnDecodedSEICallback func)
        {
            m_OnDecodedSEIFunc = func;
        }

        void set_avformat_opend_callback(StreamDemuxer::OnAVFormatOpenedFunc func) {
            m_pfnOnAVFormatOpened = func;
        }

        void set_avformat_closed_callback(StreamDemuxer::OnAVFormatClosedFunc func){
            m_pfnOnAVFormatClosed = func;
        }

        void set_read_Frame_callback(StreamDemuxer::OnReadFrameFunc func){
            m_pfnOnReadFrame = func;
        }

        void set_read_eof_callback(StreamDemuxer::OnReadEofFunc func){
            m_pfnOnReadEof = func;
        }

        int open_stream(std::string url, bool repeat = true, AVDictionary *opts=nullptr);

        int close_stream(bool is_waiting = true);
        AVCodecID get_video_codec_id();

        //External utilities
        static AVPacket* ffmpeg_packet_alloc();
        static AVCodecContext* ffmpeg_create_decoder(enum AVCodecID id, AVDictionary **opts=nullptr);
    };
}

#endif //BM_FFMPEG_DECODE_TEST_STREAM_DECODE_H
