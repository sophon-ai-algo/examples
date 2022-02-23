//
// Created by yuan on 3/4/21.
//

#ifndef INFERENCE_FRAMEWORK_WORKER_H
#define INFERENCE_FRAMEWORK_WORKER_H
#include "bmutility.h"
#include "bmgui.h"
#include "stream_pusher.h"
#include "configuration.h"
#include "stream_cvdecode.h"
#include "yolov5s.h"

struct TChannel: public bm::NoCopyable {
    int channel_id;
    uint64_t seq;
    bm::CvStreamDecoder *decoder;
    bm::FfmpegOutputer *outputer;
    cv::Mat            *mat;
    TChannel():channel_id(0), seq(0), decoder(nullptr), outputer(nullptr), mat(nullptr) {
    }

    ~TChannel() {
        if (decoder)  delete decoder;
        if (outputer) delete outputer;
        if (mat)      delete mat;
        std::cout << "TChannel(chan_id=" << channel_id << ") dtor" <<std::endl;
    }
};
using TChannelPtr = std::shared_ptr<TChannel>;
class OneCardInferApp {
    std::shared_ptr<bm::DetectorDelegate<CvFrameBaseInfo, CvFrameInfo>> m_detectorDelegate;
    bm::BMNNContextPtr m_bmctx;
    bm::TimerQueuePtr m_timeQueue;
    int m_channel_start;
    int m_channel_num;
    int m_dev_id;
    int m_skipN;
    std::string m_output_url;

    bm::BMInferencePipe<CvFrameBaseInfo, CvFrameInfo> m_inferPipe;
    std::map<int, TChannelPtr> m_chans;
    std::vector<std::string> m_urls;
    

public:
    OneCardInferApp(bm::TimerQueuePtr tq, bm::BMNNContextPtr ctx, int start_index, int num, int skip = 0):
    m_detectorDelegate(nullptr), m_channel_num(num), m_bmctx(ctx)
    {
        m_dev_id = m_bmctx->dev_id();
        m_timeQueue = tq;
        m_channel_start = start_index;
        m_skipN = skip;
    }

    ~OneCardInferApp()
    {
        std::cout << cv::format("OneCardInfoApp (devid=%d) dtor", m_dev_id) <<std::endl;
    }

    void setDetectorDelegate(std::shared_ptr<bm::DetectorDelegate<CvFrameBaseInfo, CvFrameInfo>> delegate){
        m_detectorDelegate = delegate;
    }

    void start(const std::vector<std::string>& vct_urls);
};

using OneCardInferAppPtr = std::shared_ptr<OneCardInferApp>;


#endif //INFERENCE_FRAMEWORK_MAIN_H
