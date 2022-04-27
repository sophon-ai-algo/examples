//
// Created by yuan on 3/11/21.
//

#include "worker.h"
#include "stream_sei.h"

void OneCardInferApp::start(Config& config)
{
    m_detectorDelegate->set_detected_callback([this](bm::FrameInfo &frameInfo) {
        for (int i = 0; i < frameInfo.frames.size(); ++i) {
            int ch = frameInfo.frames[i].chan_id;

            m_appStatis.m_chan_statis[ch]++;
            m_appStatis.m_statis_lock.lock();
            m_appStatis.m_total_statis++;
            m_appStatis.m_statis_lock.unlock();
        }
    });

    bm::DetectorParam param;
    int cpu_num = std::thread::hardware_concurrency();
    int tpu_num = 1;
    param.preprocess_thread_num = cpu_num;
    param.preprocess_queue_size = std::max(m_channel_num, 4);
    param.inference_thread_num = tpu_num;
    param.inference_queue_size = m_channel_num;
    param.postprocess_thread_num = cpu_num;
    param.postprocess_queue_size = m_channel_num;
    param.batch_num = m_max_batch;
    loadConfig(param, config);

    m_inferPipe.init(param, m_detectorDelegate);

    for(auto iter = m_chans.begin(); iter != m_chans.end(); ++iter) {
        int  ch     = iter->first;
        auto &pchan = iter->second;
        std::string media_file;
        AVDictionary *opts = NULL;
        av_dict_set_int(&opts, "sophon_idx", m_dev_id, 0);
        av_dict_set(&opts, "output_format", "101", 18);
        av_dict_set(&opts, "extra_frame_buffer_num", "18", 0);

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

        
        pchan->decoder->set_decoded_frame_callback([this, pchan, ch](const AVPacket* pkt, const AVFrame *frame){
            uint64_t seq = pchan->seq++;
            if (m_skipN > 0) {
                if (seq % m_skipN != 0) {
                    return;
                }
            }
            bm::FrameBaseInfo fbi;
            fbi.avframe = av_frame_alloc();
            fbi.avpkt = av_packet_alloc();
            av_frame_ref(fbi.avframe, frame);
            av_packet_ref(fbi.avpkt, pkt);
            fbi.seq = seq;

            fbi.chan_id = ch;
#ifdef DEBUG
            if (ch == 0) std::cout << "decoded frame " << std::endl;
#endif
            m_detectorDelegate->decode_process(fbi);
            m_inferPipe.push_frame(&fbi);
        });
        
        pchan->decoder->open_stream(pchan->url, true, opts);
        //pchan->decoder->open_stream("rtsp://admin:hk123456@11.73.11.99/test", false, opts);
        av_dict_free(&opts);
    }
}
