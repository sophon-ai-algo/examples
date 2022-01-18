//
// Created by hsyuan on 2019-02-27.
//

#include "stream_decode.h"
#include "stream_sei.h"

namespace bm {
    StreamDecoder::StreamDecoder(int id, AVCodecContext *decoder):m_observer(nullptr),
    m_external_dec_ctx(decoder)
    {
        std::cout << "StreamDecoder() ctor..." << std::endl;
        m_is_waiting_iframe = true;
        m_id = id;
        m_opts_decoder = NULL;
    }

    StreamDecoder::~StreamDecoder() {
        std::cout << "~StreamDecoder() dtor..." << std::endl;
        av_dict_free(&m_opts_decoder);
    }

    int StreamDecoder::decode_frame(AVPacket *pkt, AVFrame *pFrame) {
        AVCodecContext *dec_ctx = nullptr;
        if (nullptr == m_external_dec_ctx) {
            dec_ctx = m_dec_ctx;
        }else{
            dec_ctx = m_external_dec_ctx;
        }

#if LIBAVCODEC_VERSION_MAJOR > 56
        int got_picture = 0;
        int ret = avcodec_send_packet(dec_ctx, pkt);
        if (ret == AVERROR_EOF) ret = 0;
        else if (ret < 0) {
            printf(" error sending a packet for decoding\n");
            return -1;
        }

        while (ret >= 0) {
            ret = avcodec_receive_frame(dec_ctx, pFrame);
            if (ret == AVERROR(EAGAIN)) {
                printf("decoder need more stream!\n");
                break;
            } else if (ret == AVERROR_EOF) {
                printf("avcodec_receive_frame() err=end of file!\n");
            }

            if (0 == ret) {
                got_picture += 1;
                break;
            }
        }

        return got_picture;

#else
        int got_frame = 0;
        int ret = avcodec_decode_video2(dec_ctx, pFrame, &got_frame, pkt);
        if (ret < 0) {
            return -1;
        }

        if (got_frame > 0) {
            return 1;
        }

        return 0;
#endif
    }


    void StreamDecoder::on_avformat_opened(AVFormatContext *ifmt_ctx) {
        if (m_pfnOnAVFormatOpened != nullptr) {
            m_pfnOnAVFormatOpened(ifmt_ctx);
        }

        if (m_external_dec_ctx == nullptr) {
            if (0 == create_video_decoder(ifmt_ctx)) {
                printf("create video decoder ok!\n");
            }
        }

        if (strcmp(ifmt_ctx->iformat->name, "h264") !=0) {
            m_is_waiting_iframe = false;
        }
    }

    void StreamDecoder::on_avformat_closed() {
        clear_packets();
        if (m_dec_ctx != nullptr) {
            avcodec_close(m_dec_ctx);
            avcodec_free_context(&m_dec_ctx);
            printf("free video decoder context!\n");
        }

        if (m_pfnOnAVFormatClosed) {
            m_pfnOnAVFormatClosed();
        }

    }

    int StreamDecoder::on_read_frame(AVPacket *pkt) {
        int ret = 0;
        
        if (m_video_stream_index != pkt->stream_index) {
            // ignore other streams if not video.
            return 0;
        }

       if (m_is_waiting_iframe) {
           if (is_key_frame(pkt)){
               m_is_waiting_iframe = false;
           }
       }

       if (m_is_waiting_iframe){
           return 0;
       }

        auto dec_ctx = m_external_dec_ctx != nullptr ? m_external_dec_ctx:m_dec_ctx;

        if (dec_ctx->codec_id == AV_CODEC_ID_H264) {
            // handle video stream
            std::unique_ptr<uint8_t[]> sei_buf_ptr(new uint8_t[pkt->size]);
            int sei_len = 0;

            if (pkt->data &&
                0 == pkt->data[0] &&
                0 == pkt->data[1] &&
                0 == pkt->data[2] &&
                1 == pkt->data[3] &&
                (pkt->data[4] & 0x1f) == 6) {

                if (m_OnDecodedSEIFunc != nullptr || m_observer != nullptr) {
                    sei_len = h264sei_packet_read(pkt->data, pkt->size, sei_buf_ptr.get(), pkt->size);
                    if (sei_len > 0) {
                        if (m_observer != nullptr) {
                            m_observer->on_decoded_sei_info(sei_buf_ptr.get(), sei_len, pkt->pts, pkt->pos);
                        }

                        if (m_OnDecodedSEIFunc != nullptr) {
                            m_OnDecodedSEIFunc(sei_buf_ptr.get(), sei_len, pkt->pts, pkt->pos);
                        }
                    }
                }
            }
        }else if (dec_ctx->codec_id == AV_CODEC_ID_H265) {
            std::unique_ptr<uint8_t[]> sei_buf_ptr(new uint8_t[pkt->size]);
            int sei_len = 0;

            if (pkt->data) {
                int nal_type = 0;
                if (0 == pkt->data[0] &&
                    0 == pkt->data[1] &&
                    0 == pkt->data[2] &&
                    1 == pkt->data[3]){
                    nal_type = (pkt->data[4] & 0x7E) >> 1;
                }else if(0 == pkt->data[0] &&
                         0 == pkt->data[1] &&
                         1 == pkt->data[2]){
                    nal_type = (pkt->data[3] & 0x7E) >> 1;
                }

                if (nal_type == 39) {
                    if (m_observer != nullptr || m_OnDecodedSEIFunc != nullptr) {
                        sei_len = h265sei_packet_read(pkt->data, pkt->size, sei_buf_ptr.get(), pkt->size);
                        if (sei_len > 0) {
                            if (m_observer != nullptr) {
                                m_observer->on_decoded_sei_info(sei_buf_ptr.get(), sei_len, pkt->pts, pkt->pos);
                            }

                            if (m_OnDecodedSEIFunc != nullptr) {
                                m_OnDecodedSEIFunc(sei_buf_ptr.get(), sei_len, pkt->pts, pkt->pos);
                            }
                        }
                    }
                }
            }
        }


        AVFrame *pFrame = av_frame_alloc();
        //bm::BMPerf perf;
        //perf.begin("Decode", 120);
        ret = decode_frame(pkt, pFrame);
        //perf.end();

        if (ret < 0) {
            printf("decode failed!\n");
            av_frame_free(&pFrame);
            return ret;
        }

        if (m_frame_decoded_num == 0) {
            printf("id=%d, ffmpeg delayed frames: %d\n", m_id, (int)m_list_packets.size());
        }

        if (ret > 0) m_frame_decoded_num++;

        put_packet(pkt);

        if (ret > 0) {
            auto pkt_s = get_packet();

            if (m_observer){
                m_observer->on_decoded_avframe(pkt_s, pFrame);
            }

            if (m_OnDecodedFrameFunc != nullptr) {
                m_OnDecodedFrameFunc(pkt_s, pFrame);
            }

            av_packet_unref(pkt_s);
            av_freep(&pkt_s);
        }

        av_frame_unref(pFrame);
        av_frame_free(&pFrame);

        return ret;
    }

    void StreamDecoder::on_read_eof(AVPacket *pkt) {
        //flush decode cache.
        // bm_ffmpeg not supported.
#if 1
        while (1) {
            int ret = on_read_frame(pkt);
            if (ret <= 0) {
                break;
            }
        }
#endif
        m_frame_decoded_num = 0;
        clear_packets();

        if (m_observer){
            m_observer->on_stream_eof();
        }

        if (m_pfnOnReadEof != nullptr) {
            m_pfnOnReadEof(nullptr);
        }

    }


    int StreamDecoder::put_packet(AVPacket *pkt) {
#if LIBAVCODEC_VERSION_MAJOR > 56
        AVPacket *pkt_new = av_packet_alloc();
#else
        AVPacket *pkt_new = (AVPacket*)av_malloc(sizeof(AVPacket));
        av_init_packet(pkt_new);
#endif

        av_packet_ref(pkt_new, pkt);
        m_list_packets.push_back(pkt_new);
        return 0;
    }

    AVPacket *StreamDecoder::get_packet() {
        if (m_list_packets.size() == 0) return nullptr;
        auto pkt = m_list_packets.front();
        m_list_packets.pop_front();
        return pkt;
    }

    void StreamDecoder::clear_packets() {
        while (m_list_packets.size() > 0) {
            auto pkt = m_list_packets.front();
            m_list_packets.pop_front();
            av_packet_unref(pkt);
            av_freep(&pkt);
        }

        return;
    }

    int StreamDecoder::get_video_stream_index(AVFormatContext *ifmt_ctx) {
        // Only video is accepted.
        for (unsigned int i = 0; i < ifmt_ctx->nb_streams; i++) {
#if LIBAVFORMAT_VERSION_MAJOR > 56
            auto codec_type = ifmt_ctx->streams[i]->codecpar->codec_type;
#else
            auto codec_type = ifmt_ctx->streams[i]->codec->codec_type;
#endif
            if (codec_type == AVMEDIA_TYPE_VIDEO) {
                m_video_stream_index = i;
                break;
            }
        }

        return m_video_stream_index;
    }

    AVCodecID StreamDecoder::get_video_codec_id() {
        if (m_dec_ctx) {
            return m_dec_ctx->codec_id;
        }
        return AV_CODEC_ID_NONE;
    }

    int StreamDecoder::create_video_decoder(AVFormatContext *ifmt_ctx) {
        int video_index = get_video_stream_index(ifmt_ctx);
        m_timebase = ifmt_ctx->streams[video_index]->time_base;

#if LIBAVCODEC_VERSION_MAJOR > 56
        auto codec_id = ifmt_ctx->streams[video_index]->codecpar->codec_id;
#else
        auto codec_id = ifmt_ctx->streams[video_index]->codec->codec_id;
#endif

        AVCodec *pCodec = avcodec_find_decoder(codec_id);
        if (NULL == pCodec) {
            printf("can't find code_id %d\n", codec_id);
            return -1;
        }

        m_dec_ctx = avcodec_alloc_context3(pCodec);
        if (m_dec_ctx == NULL) {
            printf("avcodec_alloc_context3 err");
            return -1;
        }

        int ret = 0;

#if LIBAVCODEC_VERSION_MAJOR > 56
        if ((ret = avcodec_parameters_to_context(m_dec_ctx, ifmt_ctx->streams[video_index]->codecpar)) < 0) {
            fprintf(stderr, "Failed to copy %s codec parameters to decoder context\n",
                    av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
            return ret;
        }
#else
        /* Copy codec parameters from input stream to output codec context */
        if ((ret = avcodec_copy_context(m_dec_ctx, ifmt_ctx->streams[video_index]->codec)) < 0) {
            fprintf(stderr, "Failed to copy %s codec parameters to decoder context\n",
                    av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
            return ret;
        }
#endif

        if (pCodec->capabilities & AV_CODEC_CAP_TRUNCATED) {
            m_dec_ctx->flags |= AV_CODEC_FLAG_TRUNCATED; /* we do not send complete frames */
        }

        //for PCIE
        //av_dict_set_int(&opts, "pcie_board_id", 0, 0x0);
        //av_dict_set_int(&opts, "pcie_no_copyback", 1, 0x0);

        //for SOC
        //av_dict_set_int(&m_opts_decoder, "extra_frame_buffer_num", 8, 0);
        AVDictionary *opts = NULL;
        av_dict_copy(&opts, m_opts_decoder, 0);
        if (avcodec_open2(m_dec_ctx, pCodec, &opts) < 0) {
            std::cout << "Unable to open codec";
            return -1;
        }

        return 0;
    }

    int StreamDecoder::set_observer(StreamDecoderEvents *observer)
    {
        m_observer = observer;
        return 0;
    }

    int StreamDecoder::open_stream(std::string url, bool repeat, AVDictionary *opts)
    {
        av_dict_copy(&m_opts_decoder, opts, 0);
        return m_demuxer.open_stream(url, this, repeat);
    }

    int StreamDecoder::close_stream(bool is_waiting){
        return m_demuxer.close_stream(is_waiting);
    }

    AVPacket* StreamDecoder::ffmpeg_packet_alloc() {
#if LIBAVCODEC_VERSION_MAJOR > 56
        AVPacket *pkt_new = av_packet_alloc();
#else
        AVPacket *pkt_new = (AVPacket*)av_malloc(sizeof(AVPacket));
        av_init_packet(pkt_new);
#endif
        return pkt_new;
    }

    AVCodecContext* StreamDecoder::ffmpeg_create_decoder(enum AVCodecID codec_id, AVDictionary **opts)
    {
        AVCodec *pCodec = avcodec_find_decoder(codec_id);
        if (NULL == pCodec) {
            printf("can't find code_id %d\n", codec_id);
            return nullptr;
        }



        AVCodecContext* dec_ctx = avcodec_alloc_context3(pCodec);
        if (dec_ctx == NULL) {
            printf("avcodec_alloc_context3 err");
            return nullptr;
        }

        if (pCodec->capabilities & AV_CODEC_CAP_TRUNCATED) {
            dec_ctx->flags |= AV_CODEC_FLAG_TRUNCATED;
        }

        dec_ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;

        dec_ctx->workaround_bugs = FF_BUG_AUTODETECT;
        dec_ctx->err_recognition = AV_EF_CAREFUL;
        dec_ctx->error_concealment = FF_EC_GUESS_MVS | FF_EC_DEBLOCK;
        dec_ctx->has_b_frames = 0;

        //for PCIE
        //av_dict_set_int(&opts, "pcie_board_id", 0, 0x0);
        //av_dict_set_int(&opts, "pcie_no_copyback", 1, 0x0);

        //for SOC
        //av_dict_set_int(&m_opts_decoder, "extra_frame_buffer_num", 8, 0);

        if (avcodec_open2(dec_ctx, pCodec, opts) < 0) {
            std::cout << "Unable to open codec";
            avcodec_free_context(&dec_ctx);
            return nullptr;
        }

        return dec_ctx;
    }

    bool StreamDecoder::is_key_frame(AVPacket *pkt) {
        auto dec_ctx = m_external_dec_ctx != nullptr ? m_external_dec_ctx:m_dec_ctx;
        if (dec_ctx->codec_id == AV_CODEC_ID_H264) {
            if (pkt == nullptr || pkt->data == nullptr) return false;
            int nal_type = pkt->data[4] & 0x1f;
            //std::cout << "nal_type=" << nal_type << std::endl;
            if (nal_type != 7) {
                uint8_t *p = &pkt->data[0];
                uint8_t *end = &pkt->data[pkt->size];
                p+=4;
                while(p != end) {
                    if (p[0] == 0 && p[1] == 0 &&
                        p[2] == 0 && p[3] == 1) {
                        nal_type = p[4] & 0x1f;
                        break;
                    }
                    p++;
                }
            }
            if (nal_type == 7 || nal_type == 5) { //IDR //SEI
                return true;
            }else{
                return false;
            }
        }else{
            return true;
        }
    }
}

