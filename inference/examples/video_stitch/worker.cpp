//
// Created by yuan on 3/11/21.
//

#include <string>
#include "worker.h"

void OneCardInferApp::start(const std::vector<std::string>& urls)
{
    bm::DetectorParam param;
    int cpu_num = std::thread::hardware_concurrency();
    int tpu_num = 1;
    param.preprocess_thread_num = cpu_num;
    param.preprocess_queue_size = 5*m_channel_num;
    param.inference_thread_num = tpu_num;
    param.inference_queue_size = 8*m_channel_num;
    param.postprocess_thread_num = cpu_num;
    param.postprocess_queue_size = 5*m_channel_num;
    param.stitch_thread_num = 1;
    param.stitch_queue_size = 20;
    param.encode_thread_num = 1;
    param.encode_queue_size = 20;

    m_inferPipe.init(param, m_detectorDelegate);

    for (int i = 0; i < m_channel_num; ++i) {
        int ch = m_channel_start + i;
        std::cout << "push id=" << ch << std::endl;
        TChannelPtr pchan = std::make_shared<TChannel>();
        pchan->decoder = new bm::CvStreamDecoder(ch);
        pchan->channel_id = ch;
        std::string url = urls[i % urls.size()];
        pchan->mat = new cv::Mat;


        pchan->decoder->set_cvcapture_opened_callback([this, url](std::shared_ptr<cv::VideoCapture> pCap) {
            if (pCap != nullptr && pCap->isOpened()) {
                pCap->set(cv::CAP_PROP_OUTPUT_YUV, PROP_TRUE);

                int fps    = pCap->get(cv::CAP_PROP_FPS);
                int height = (int)pCap->get(cv::CAP_PROP_FRAME_HEIGHT);
                int width  = (int)pCap->get(cv::CAP_PROP_FRAME_WIDTH);
                std::cout << url             << " opened!"
                          << "\tfps: "       << fps
                          << "\theight: "    << height
                          << "\twidth: "     << width << std::endl;
            }
        });

        pchan->decoder->set_cvcapture_closed_callback([this, url](std::shared_ptr<cv::VideoCapture> pCap) {
            std::cout << url << " closed!";
        });

        pchan->decoder->open_stream(url, true);
        pchan->decoder->set_decoded_frame_callback([this, pchan, ch, url](bm::CvMatPtr pMat){
            CvFrameBaseInfo fbi;
            fbi.mat = pMat;
            fbi.seq = pchan->seq++;
            if (m_skipN > 0) {
                if (fbi.seq % m_skipN != 0) {
                    delete pMat;
                    return;
                }
            }
            fbi.chan_id = ch;
            m_detectorDelegate->decode_process(fbi);
            m_inferPipe.push_frame(fbi);
        });

        m_chans[ch] = pchan;
    }
}