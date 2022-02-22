//
// Created by hsyuan on 2019-02-22.
//

#include "stream_demuxer.h"
//#include "otl_utils.h"

namespace bm {

    StreamDemuxer::StreamDemuxer(int id) : m_ifmt_ctx(nullptr), m_observer(nullptr),
                                     m_thread_reading(nullptr),m_id(id) {
        m_ifmt_ctx = avformat_alloc_context();
    }

    StreamDemuxer::~StreamDemuxer() {
        std::cout << "~StreamDemuxer() dtor..." << std::endl;
        close_stream(false);
        avformat_close_input(&m_ifmt_ctx);
        avformat_free_context(m_ifmt_ctx);
    }


/*int StreamDemuxer::get_codec_parameters(int stream_index, AVCodecParameters **p_codecpar)
{
    if (nullptr == m_ifmt_ctx || stream_index >= m_ifmt_ctx->nb_streams) {
        return -1;
    }

    *p_codecpar = m_ifmt_ctx->streams[stream_index]->codecpar;
    return 0;
}

int StreamDemuxer::get_codec_type(int stream_index, int *p_codec_type)
{
    if (nullptr == m_ifmt_ctx || stream_index >= m_ifmt_ctx->nb_streams) {
        return -1;
    }

    *p_codec_type = m_ifmt_ctx->streams[stream_index]->codecpar->codec_type;
    return 0;
}*/

    int StreamDemuxer::do_initialize() {

        std::string prefix = "rtsp://";
        AVDictionary *opts = NULL;
        if (m_inputUrl.compare(0, prefix.size(), prefix) == 0) {
            av_dict_set(&opts, "rtsp_transport", "tcp", 0);
            av_dict_set(&opts, "stimeout", "2000000", 0);
            av_dict_set(&opts, "probesize", "400", 0);
            av_dict_set(&opts, "analyzeduration", "100", 0);
        }else{
            m_is_file_url = true;
        }

        av_dict_set(&opts, "rw_timeout", "15000", 0);
        
        std::cout << "Open stream " << m_inputUrl << std::endl;

        int ret = avformat_open_input(&m_ifmt_ctx, m_inputUrl.c_str(), nullptr, &opts);
        av_dict_free(&opts);
        if (ret < 0) {
            std::cout << "Can't open file " << m_inputUrl << std::endl;
            return ret;
        }

        ret = avformat_find_stream_info(m_ifmt_ctx, NULL);
        if (ret < 0) {
            std::cout << "Unable to get stream info" << std::endl;
            return ret;
        }

        std::cout << "Init:total stream num:" << m_ifmt_ctx->nb_streams << std::endl;
        if (m_observer) {
            m_observer->on_avformat_opened(m_ifmt_ctx);
        }

        if (m_pfnOnAVFormatOpened != nullptr) {
            m_pfnOnAVFormatOpened(m_ifmt_ctx);
        }

        // Enter Working service
        m_work_state = Service;

        return 0;
    }

    int StreamDemuxer::do_down() {
        // Close avformat_input
        avformat_close_input(&m_ifmt_ctx);

        if (m_observer) {
            m_observer->on_avformat_closed();
        }

        if (m_pfnOnAVFormatClosed != nullptr) {
            m_pfnOnAVFormatClosed();
        }

        if (m_repeat) {
            m_work_state = Initialize;
        } else {
            m_keep_running = false;
        }

        return 0;
    }

    int StreamDemuxer::do_service() {
#if LIBAVCODEC_VERSION_MAJOR > 56
        AVPacket *pkt = av_packet_alloc();
#else
        AVPacket *pkt = (AVPacket*)av_malloc(sizeof(AVPacket));
        av_init_packet(pkt);
#endif

        m_start_time = av_gettime();
        int64_t frame_index = 0;
        while (Service == m_work_state) {
            int ret = av_read_frame(m_ifmt_ctx, pkt);
            if (ret < 0) {
                if (ret != AVERROR_EOF) continue;
                if (m_repeat && m_is_file_url) {
                    ret = av_seek_frame(m_ifmt_ctx, -1, m_ifmt_ctx->start_time, 0);
                    if (ret != 0) {
                        ret = av_seek_frame(m_ifmt_ctx,  -1, m_ifmt_ctx->start_time, AVSEEK_FLAG_BYTE);
                        if (ret < 0) {
                            std::cout << "av_seek_frame failed!" << std::endl;
                        }
                    }
                    frame_index = 0;
                    m_start_time = av_gettime();
                    printf("seek_to_start\n");
                    continue;
                }else{
                    printf("file[%d] end!\n", m_id);
                    if (m_observer) m_observer->on_read_eof(pkt);
                    if (m_pfnOnReadEof != nullptr) m_pfnOnReadEof(pkt);
                    m_work_state = Down;
                }

                break;
            }

            if (m_last_frame_time != 0) {
                if(pkt->pts==AV_NOPTS_VALUE){
                    AVRational time_base1=m_ifmt_ctx->streams[0]->time_base;
                    //Duration between 2 frames (us)
                    int64_t calc_duration=(double)AV_TIME_BASE/av_q2d(m_ifmt_ctx->streams[0]->r_frame_rate);
                    //Parameters
                    pkt->pts=(double)(frame_index*calc_duration)/(double)(av_q2d(time_base1)*AV_TIME_BASE);
                    pkt->dts=pkt->pts;
                    pkt->duration=(double)calc_duration/(double)(av_q2d(time_base1)*AV_TIME_BASE);
                }

                AVRational time_base = m_ifmt_ctx->streams[0]->time_base;
                AVRational time_base_q = {1, AV_TIME_BASE};
                int64_t pts_time = av_rescale_q(pkt->dts, time_base, time_base_q);
                int64_t now_time = av_gettime() - m_start_time;
                if (pts_time > now_time) {
                    int64_t delta = pts_time - now_time;
                    if (delta < 100000) {
                      av_usleep(delta);
                    }
                }
            }

            m_last_frame_time = av_gettime();
            if (pkt->stream_index == 0) frame_index++;

            if (m_observer) {
                m_observer->on_read_frame(pkt);
            }

            if (m_pfnOnReadFrame) {
                m_pfnOnReadFrame(pkt);
            }

            av_packet_unref(pkt);
        }

#if LIBAVCODEC_VERSION_MAJOR > 56
        av_packet_free(&pkt);
#else
        av_free_packet(pkt);
        av_freep(&pkt);
#endif

        return 0;
    }

    int StreamDemuxer::open_stream(std::string url, StreamDemuxerEvents *observer,
            bool repeat, bool is_sync_open) {

        //First stop previous
        close_stream(false);

        m_inputUrl = url;
        m_observer = observer;
        m_repeat = repeat;
        m_work_state = Initialize;
        if (is_sync_open) {
            int ret = do_initialize();
            if (ret < 0) {
                return ret;
            }
        }

        m_keep_running = true;
        m_thread_reading = new std::thread([&] {
            while (m_keep_running) {
                switch (m_work_state) {
                    case Initialize:
                        if (do_initialize() != 0) {
                           std::this_thread::sleep_for(std::chrono::seconds(1));
                        }
                        break;
                    case Service:
                        do_service();
                        break;
                    case Down:
                        do_down();
                        break;
                }
            }
        });

        return 0;
    }

    int StreamDemuxer::close_stream(bool is_waiting) {
        if (!is_waiting) {
            m_work_state = Down;
            m_repeat = false;
        }

        if (nullptr != m_thread_reading) {
            m_thread_reading->join();
            delete m_thread_reading;
            m_thread_reading = nullptr;
        }

        return 0;
    }

}
