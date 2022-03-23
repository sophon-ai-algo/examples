//
// Created by yuan on 3/11/21.
//

#include "worker.h"
#include "stream_sei.h"

void OneCardInferApp::start(const std::vector<std::string>& urls, Config& config)
{
    bool enable_outputer = false;
    if (bm::start_with(m_output_url, "rtsp://") || bm::start_with(m_output_url, "udp://") ||
        bm::start_with(m_output_url, "tcp://")) {
        enable_outputer = true;
    }

    m_detectorDelegate->set_detected_callback([this, enable_outputer](bm::cvs10FrameInfo &frameInfo) {
        for (int frame_idx = 0; frame_idx < frameInfo.frames.size(); ++frame_idx) {
            int ch = frameInfo.frames[frame_idx].chan_id;

            m_appStatis.m_chan_statis[ch]++;
            m_appStatis.m_statis_lock.lock();
            m_appStatis.m_total_statis++;
            m_appStatis.m_statis_lock.unlock();
            // tracker
            if (frameInfo.out_datums[frame_idx].obj_rects.size() > 0) {
                m_chans[ch]->tracker->update(frameInfo.out_datums[frame_idx].obj_rects, frameInfo.out_datums[frame_idx].track_rects);
            }
            // display
        }
    });

    // feature register
    m_featureDelegate->set_detected_callback([this](bm::FeatureFrameInfo &frameInfo) {
        for (int i = 0; i < frameInfo.frames.size(); ++i) {
            int ch = frameInfo.frames[i].chan_id;

            m_appStatis.m_chan_feat_stat[ch]++;
            m_appStatis.m_total_feat_stat++;
        }
    });

    //detector
    bm::DetectorParam param;
    int cpu_num = std::thread::hardware_concurrency();
    int tpu_num = 1;
    param.preprocess_thread_num = cpu_num;
    param.preprocess_queue_size = std::max(m_channel_num, 8);
    param.inference_thread_num = tpu_num;
    param.inference_queue_size = m_channel_num;
    param.postprocess_thread_num = cpu_num;
    param.postprocess_queue_size = m_channel_num;

    loadConfig(param, config);

    m_inferPipe.init(param, m_detectorDelegate);

    //feature
    m_featurePipe.init(param, m_featureDelegate);

    for(int i = 0; i < m_channel_num; ++i) {
        int ch = m_channel_start + i;
        std::cout << "push id=" << ch << std::endl;
        TChannelPtr pchan = std::make_shared<TChannel>();
        pchan->demuxer = new bm::StreamDemuxer(ch);
        //if (enable_outputer) pchan->outputer = new bm::FfmpegOutputer();
        pchan->channel_id = ch;

        std::string media_file;
        AVDictionary *opts = NULL;
        av_dict_set_int(&opts, "sophon_idx", m_dev_id, 0);
        av_dict_set(&opts, "output_format", "101", 18);
        av_dict_set(&opts, "extra_frame_buffer_num", "5", 0);

        pchan->demuxer->set_avformat_opend_callback([this, pchan](AVFormatContext *ifmt) {
            // create DDR reduction
            if (m_use_l2_ddrr) {
                pchan->m_ddrr = DDRReduction::create(this->m_dev_id, ifmt->streams[0]->codecpar->codec_id);
            }else{
                pchan->create_video_decoder(m_dev_id, ifmt);
            }
        });

        pchan->demuxer->set_avformat_closed_callback([this, pchan]() {
            //if (pchan->outputer) pchan->outputer->CloseOutputStream();
        });

        pchan->demuxer->set_read_Frame_callback([this, pchan, ch](AVPacket* pkt){
            int ret = 0;
            if (pchan->m_ddrr) {
                pchan->m_ddrr->put_packet(pkt, [this, pchan, ch](int64_t pkt_id, AVFrame *frame){
                    bm::cvs10FrameBaseInfo fbi;
                    fbi.seq = pchan->seq++;
                    fbi.chan_id = ch;
                    fbi.ddrr = pchan->m_ddrr;
                    fbi.pkt_id = pkt_id;

                    if (m_skipN > 0) {
                        if (fbi.seq % (m_skipN+1) != 0) fbi.skip = true;
                    }

#if 0
                    if (ch == 0) std::cout << " seq = " << fbi.seq << " skip= " << fbi.skip << std::endl;
#endif
                    if(!fbi.skip) {
                        fbi.avframe = av_frame_alloc();
                        av_frame_ref(fbi.avframe, frame);
                        m_inferPipe.push_frame(&fbi);
                    }

                });
            }else{
                //not use ddr reduction
                int got_picture = 0;
                AVFrame *frame = av_frame_alloc();
                pchan->decode_video2(pchan->m_decoder, frame, &got_picture, pkt);
                if (got_picture) {
                    bm::cvs10FrameBaseInfo fbi;
                    fbi.seq = pchan->seq++;
                    fbi.chan_id = ch;
                    fbi.ddrr = pchan->m_ddrr;
                    fbi.pkt_id = 0;

                    if (m_skipN > 0) {
                        if (fbi.seq % (m_skipN + 1) != 0) fbi.skip = true;
                    }

#if 0
                    if (ch == 0) std::cout << " seq = " << fbi.seq << " skip= " << fbi.skip << std::endl;
#endif
                    if (!fbi.skip) {
                        fbi.avframe = av_frame_alloc();
                        av_frame_ref(fbi.avframe, frame);
                        m_inferPipe.push_frame(&fbi);
                    }
                }

                av_frame_free(&frame);
            }

            uint64_t current_time = bm::gettime_msec();
            if (current_time - m_chans[ch]->m_last_feature_time > m_feature_delay) {
                for(int feat_idx = 0; feat_idx < m_feature_num; feat_idx++) {
                    bm::FeatureFrame featureFrame;
                    featureFrame.chan_id = ch;
                    featureFrame.seq++;
                    featureFrame.img = cv::imread("face.jpeg", cv::IMREAD_COLOR, m_dev_id);
                    if (featureFrame.img.empty()) {
                        printf("ERROR:Can't find face.jpg in workdir!\n");
                        exit(0);
                    }
                    m_featurePipe.push_frame(&featureFrame);
                }

                m_chans[ch]->m_last_feature_time = current_time;
            }


        });

        pchan->demuxer->open_stream(urls[i % urls.size()], nullptr, true, opts);
        av_dict_free(&opts);
        m_chans[ch] = pchan;
    }
}
