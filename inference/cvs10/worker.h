//
// Created by yuan on 3/4/21.
//

#ifndef INFERENCE_FRAMEWORK_WORKER_H
#define INFERENCE_FRAMEWORK_WORKER_H
#include "bmutility.h"
#include "bmgui.h"
#include "inference.h"
#include "stream_pusher.h"
#include "configuration.h"
#include "face_detector.h"
#include "bm_tracker.h"
#include "common_types.h"

struct TChannel: public bm::NoCopyable {
    int channel_id;
    uint64_t seq;
    bm::StreamDecoder *decoder;
    bm::FfmpegOutputer *outputer;
    std::shared_ptr<bm::BMTracker> tracker;
    uint64_t m_last_feature_time=0; // last do feature time

    TChannel():channel_id(0), seq(0), decoder(nullptr) {
         outputer = nullptr;
         tracker = bm::BMTracker::create();
         m_last_feature_time = 0;
    }

    ~TChannel() {
        if (decoder) delete decoder;
        std::cout << "TChannel(chan_id=" << channel_id << ") dtor" <<std::endl;
    }
};
using TChannelPtr = std::shared_ptr<TChannel>;

class OneCardInferApp {
    bm::VideoUIAppPtr m_guiReceiver;
    AppStatis &m_appStatis;
    std::shared_ptr<bm::DetectorDelegate<bm::FrameBaseInfo, bm::FrameInfo>> m_detectorDelegate;
    std::shared_ptr<bm::DetectorDelegate<bm::FeatureFrame, bm::FeatureFrameInfo>> m_featureDelegate;
    bm::BMNNContextPtr m_bmctx;
    bm::TimerQueuePtr m_timeQueue;
    int m_channel_start;
    int m_channel_num;
    int m_dev_id;
    int m_skipN;
    std::string m_output_url;
    int m_feature_delay;
    int m_feature_num;

    bm::BMInferencePipe<bm::FrameBaseInfo, bm::FrameInfo> m_inferPipe;
    bm::BMInferencePipe<bm::FeatureFrame, bm::FeatureFrameInfo> m_featurePipe;

    std::map<int, TChannelPtr> m_chans;
    std::vector<std::string> m_urls;
public:
    OneCardInferApp(AppStatis& statis,bm::VideoUIAppPtr gui, bm::TimerQueuePtr tq, bm::BMNNContextPtr ctx,
            std::string& output_url, int start_index, int num, int skip=0, int feat_delay=1000, int feat_num=8):
    m_detectorDelegate(nullptr), m_channel_num(num), m_bmctx(ctx), m_appStatis(statis)
    {
        m_guiReceiver = gui;
        m_dev_id = m_bmctx->dev_id();
        m_timeQueue = tq;
        m_channel_start = start_index;
        m_skipN = skip;
        m_output_url = output_url;
        m_feature_delay = feat_delay;
        m_feature_num = feat_num;

    }

    ~OneCardInferApp()
    {
        std::cout << cv::format("OneCardInfoApp (devid=%d) dtor", m_dev_id) <<std::endl;
    }

    void setDetectorDelegate(std::shared_ptr<bm::DetectorDelegate<bm::FrameBaseInfo, bm::FrameInfo>> delegate){
        m_detectorDelegate = delegate;
    }

    void setFeatureDelegate(std::shared_ptr<bm::DetectorDelegate<bm::FeatureFrame, bm::FeatureFrameInfo>> delegate){
        m_featureDelegate = delegate;
    }

    void start(const std::vector<std::string>& vct_urls);
};

using OneCardInferAppPtr = std::shared_ptr<OneCardInferApp>;


#endif //INFERENCE_FRAMEWORK_MAIN_H
