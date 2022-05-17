//
// Created by yuan on 3/11/21.
//

#include "worker.h"
#include "stream_sei.h"

void OneCardInferApp::start(const std::vector<std::string>& urls, Config& config)
{
    bool enable_outputer = false;

    bm::DetectorParam param;
    int cpu_num = std::thread::hardware_concurrency();
    int tpu_num = 1;
    param.preprocess_thread_num = cpu_num;
    param.preprocess_queue_size = std::max(m_channel_num, 4);
    param.inference_thread_num = tpu_num;
    param.inference_queue_size = m_channel_num;
    param.postprocess_thread_num = cpu_num;
    param.postprocess_queue_size = m_channel_num;
    param.track_queue_size = 8;
    param.track_thread_num = 1;
    param.batch_num = m_max_batch;
    loadConfig(param, config);

    m_inferPipe.init(param, m_detectorDelegate,
                     bm::FrameBaseInfo::FrameBaseInfoDestroyFn,
                     bm::FrameInfo::FrameInfoDestroyFn,
                     bm::FrameInfo::FrameInfoDestroyFn,
                     bm::FrameInfo::FrameInfoDestroyFn);

    for(int i = 0; i < m_channel_num; ++i) {
        int ch = m_channel_start + i;
        std::cout << "push id=" << ch << std::endl;
        TChannelPtr pchan = std::make_shared<TChannel>();
        pchan->decoder = new bm::StreamDecoder(ch);
        //if (enable_outputer) pchan->outputer = new bm::FfmpegOutputer();
        pchan->channel_id = ch;

        std::string media_file;
        AVDictionary *opts = NULL;
        av_dict_set_int(&opts, "sophon_idx", m_dev_id, 0);
        //av_dict_set(&opts, "output_format", "101", 18);
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

        pchan->decoder->open_stream(urls[i % urls.size()], true, opts);
        av_dict_free(&opts);
        pchan->decoder->set_decoded_frame_callback([this, pchan, ch](const AVPacket* pkt, const AVFrame *frame) {
            int frame_seq = pchan->seq++;
            if (m_skipN > 0) {
                if (frame_seq % m_skipN != 0) {
                    return;
                }
            }
            bm::FrameBaseInfo fbi;
//            fbi.avframe = av_frame_alloc();
//            fbi.avpkt = av_packet_alloc();
//            av_frame_ref(fbi.avframe, frame);
//            av_packet_ref(fbi.avpkt, pkt);
            bm_image image;
            bm::BMImage::from_avframe(
                    m_handle,
                    frame, image,
                    true);
            fbi.original = image;
            fbi.seq = frame_seq;
            if (m_skipN > 0) {
                if (fbi.seq % m_skipN != 0) fbi.skip = true;
            }
            fbi.chan_id = ch;
#ifdef DEBUG
            if (ch == 0) std::cout << "decoded frame " << std::endl;
#endif
            m_detectorDelegate->decode_process(fbi);
            m_inferPipe.push_frame(&fbi);
        });

        m_chans[ch] = pchan;
    }
}
