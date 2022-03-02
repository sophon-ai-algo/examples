//
// Created by yuan on 3/4/21.
//

#ifndef INFERENCE_FRAMEWORK_FACE_WORKER_H
#define INFERENCE_FRAMEWORK_FACE_WORKER_H
#include "bmutility.h"
#include "bmgui.h"
#include "inference2.h"
#include "common.h"
#include "stream_pusher.h"
#include "configuration.h"


#include "safe_hat_detect/safety_hat_detect.hpp"
#include "safe_hat_detect/person_detect.hpp"

struct TChannel: public bm::NoCopyable {
    int channel_id;
    uint64_t seq;
    bm::StreamDecoder *decoder;
    bm::FfmpegOutputer *outputer;
    TChannel():channel_id(0), seq(0), decoder(nullptr) {
         outputer = nullptr;
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

    bm::BMNNContextPtr m_bmctx;
    bm::TimerQueuePtr m_timeQueue;
    int m_channel_start;
    int m_channel_num;
    int m_dev_id;
    int m_skipN;
    std::string m_output_url;

    std::vector<bm::BMInferencePipe<bm::FrameInfo2>> m_inferPipes;
    std::vector<std::shared_ptr<bm::DetectorDelegate<bm::FrameInfo2>>> m_detectorDelegates;

    std::map<int, TChannelPtr> m_chans;
    std::vector<std::string> m_urls;
public:
    OneCardInferApp(AppStatis& statis,bm::VideoUIAppPtr gui, bm::TimerQueuePtr tq, bm::BMNNContextPtr ctx,
            std::string& output_url, int start_index, int num, int skip=0, int model_num=1):
    m_channel_num(num), m_bmctx(ctx), m_appStatis(statis)
    {
        m_guiReceiver = gui;
        m_dev_id = m_bmctx->dev_id();
        m_timeQueue = tq;
        m_channel_start = start_index;
        m_output_url = output_url;
        m_inferPipes.resize(model_num);
        m_detectorDelegates.resize(model_num);
        m_skipN = 0;
    }

    ~OneCardInferApp()
    {
        std::cout << cv::format("OneCardInfoApp (devid=%d) dtor", m_dev_id) <<std::endl;
    }

    void setDetectorDelegate(int model_idx, std::shared_ptr<bm::DetectorDelegate<bm::FrameInfo2>> delegate){
        m_detectorDelegates[model_idx] = delegate;
    }

    void start(const std::vector<std::string>& vct_urls, Config& config);

    inline void loadConfig(bm::DetectorParam& param, Config& config) {
        SConcurrencyConfig cfg;
        if (config.get_phrase_config("preprocess", cfg)){
            param.preprocess_thread_num    = cfg.thread_num;
            param.preprocess_queue_size    = cfg.queue_size;
        }
        if (config.get_phrase_config("inference", cfg)){
            param.inference_thread_num    = cfg.thread_num;
            param.inference_queue_size    = cfg.queue_size;
        }
        if (config.get_phrase_config("postprocess", cfg)){
            param.postprocess_thread_num    = cfg.thread_num;
            param.postprocess_queue_size    = cfg.queue_size;
        }
    }
};

using OneCardInferAppPtr = std::shared_ptr<OneCardInferApp>;


#endif //INFERENCE_FRAMEWORK_MAIN_H
