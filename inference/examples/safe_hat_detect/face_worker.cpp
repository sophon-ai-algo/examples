//
// Created by yuan on 3/11/21.
//

#include "face_worker.h"
#include "stream_sei.h"

void OneCardInferApp::start(const std::vector<std::string>& urls, Config& config)
{
    // connect all detector's pipe
    int model_num = m_inferPipes.size();
    for(int i = 0; i < model_num - 1;++i) {
        m_detectorDelegates[i]->set_next_inference_pipe(&m_inferPipes[i+1]);
    }

    bool enable_outputer = false;
    if (bm::start_with(m_output_url, "rtsp://") || bm::start_with(m_output_url, "udp://") ||
        bm::start_with(m_output_url, "tcp://")) {
        enable_outputer = true;
    }


    m_detectorDelegates[model_num-1]->set_detected_callback([this, enable_outputer](bm::FrameInfo2 &frameInfo) {
        for (int i = 0; i < frameInfo.frames.size(); ++i) {
            int ch = frameInfo.frames[i].chan_id;

            m_appStatis.m_chan_statis[ch]++;
            m_appStatis.m_total_statis++;
            //to display
#if USE_QTGUI
            bm::UIFrame jpgframe;
            jpgframe.jpeg_data = frameInfo.frames[i].jpeg_data;
            jpgframe.chan_id = ch;
            jpgframe.h = frameInfo.frames[i].height;
            jpgframe.w = frameInfo.frames[i].width;
            jpgframe.datum = frameInfo.out_datums[i];
            m_guiReceiver->pushFrame(jpgframe);
#endif

            if (enable_outputer) {

                std::shared_ptr<bm::ByteBuffer> buf = frameInfo.out_datums[i].toByteBuffer();
                std::string base64_str = bm::base64_enc(buf->data(), buf->size());

                AVPacket sei_pkt;
                av_init_packet(&sei_pkt);
                AVPacket *pkt1 = frameInfo.frames[i].avpkt;
                av_packet_copy_props(&sei_pkt, pkt1);
                sei_pkt.stream_index = pkt1->stream_index;

                AVCodecID codec_id = m_chans[ch]->decoder->get_video_codec_id();

                if (codec_id == AV_CODEC_ID_H264) {
                    int packet_size = h264sei_calc_packet_size(base64_str.length());
                    AVBufferRef *buf = av_buffer_alloc(packet_size << 1);
                    //assert(packet_size < 16384);
                    int real_size = h264sei_packet_write(buf->data, true, (uint8_t *) base64_str.data(),
                                                         base64_str.length());
                    sei_pkt.data = buf->data;
                    sei_pkt.size = real_size;
                    sei_pkt.buf = buf;

                } else if (codec_id == AV_CODEC_ID_H265) {
                    int packet_size = h264sei_calc_packet_size(base64_str.length());
                    AVBufferRef *buf = av_buffer_alloc(packet_size << 1);
                    int real_size = h265sei_packet_write(buf->data, true, (uint8_t *) base64_str.data(),
                                                         base64_str.length());
                    sei_pkt.data = buf->data;
                    sei_pkt.size = real_size;
                    sei_pkt.buf = buf;
                }

                m_chans[ch]->outputer->InputPacket(&sei_pkt);
                m_chans[ch]->outputer->InputPacket(frameInfo.frames[i].avpkt);
                av_packet_unref(&sei_pkt);
            }
        }
    });

    for(int i = 0;i < model_num; ++i) {
        bm::DetectorParam param;
        int cpu_num = std::thread::hardware_concurrency();
        int tpu_num = 1;
#if 0
        param.preprocess_thread_num = 2 * cpu_num;
        param.preprocess_queue_size = 3 * m_channel_num;
        param.inference_thread_num = tpu_num;
        param.inference_queue_size = 8;
        param.postprocess_thread_num = 2 * cpu_num;
        param.postprocess_queue_size = 5;
#else
        param.preprocess_thread_num = 1;
        param.preprocess_queue_size = 8;
        param.inference_thread_num = 1;
        param.inference_queue_size = 8;
        param.postprocess_thread_num = 1;
        param.postprocess_queue_size = 5;
        loadConfig(param, config);

#endif

        m_inferPipes[i].init(param, m_detectorDelegates[i]);
    }

    for(int i = 0; i < m_channel_num; ++i) {
        int ch = m_channel_start + i;
        std::cout << "push id=" << ch << std::endl;
        TChannelPtr pchan = std::make_shared<TChannel>();
        pchan->decoder = new bm::StreamDecoder(ch);
        if (enable_outputer) pchan->outputer = new bm::FfmpegOutputer();
        pchan->channel_id = ch;

        std::string media_file;
        AVDictionary *opts = NULL;
        av_dict_set_int(&opts, "sophon_idx", m_dev_id, 0);
        av_dict_set(&opts, "output_format", "101", 18);

        pchan->decoder->set_avformat_opend_callback([this, pchan](AVFormatContext *ifmt) {
            if (pchan->outputer) {
                size_t pos = m_output_url.rfind(":");
                std::string base_url = m_output_url.substr(0, pos);
                int base_port = std::strtol(m_output_url.substr(pos + 1).c_str(), 0, 10);
                std::string url = bm::format("%s:%d", base_url.c_str(), base_port + pchan->channel_id);
                pchan->outputer->OpenOutputStream(url, ifmt);
            }
        });

        pchan->decoder->set_avformat_closed_callback([this, pchan]() {
            if (pchan->outputer) pchan->outputer->CloseOutputStream();
        });

        pchan->decoder->open_stream(urls[i % urls.size()], true, opts);
        //pchan->decoder->open_stream("rtsp://admin:hk123456@11.73.11.99/test", false, opts);
        av_dict_free(&opts);
        pchan->decoder->set_decoded_frame_callback([this, pchan, ch](const AVPacket* pkt, const AVFrame *frame){
            bm::FrameBaseInfo2 fbi;
            fbi.avframe = av_frame_alloc();
            fbi.avpkt = av_packet_alloc();
            av_frame_ref(fbi.avframe, frame);
            av_packet_ref(fbi.avpkt, pkt);
            fbi.seq = pchan->seq++;
            if (m_skipN > 0) {
                if (fbi.seq % m_skipN != 0) fbi.skip = true;
            }
            fbi.chan_id = ch;

            bm::FrameInfo2 finfo;
            finfo.frames.push_back(fbi);
            m_inferPipes[0].push_frame(&finfo);
        });

        m_chans[ch] = pchan;
    }
}
