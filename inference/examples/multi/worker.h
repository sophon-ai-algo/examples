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

#include "yolov5s.h"

struct TChannel: public bm::NoCopyable {
    std::string url;
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
    std::shared_ptr<bm::DetectorDelegate<bm::FrameBaseInfo, bm::FrameInfo>> m_detectorDelegate;
    bm::BMNNContextPtr m_bmctx;
    int m_channel_num;
    int m_dev_id;
    int m_skipN;
    std::string m_output_url;
    int m_max_batch;

    bm::BMInferencePipe<bm::FrameBaseInfo, bm::FrameInfo> m_inferPipe;
    std::map<int, TChannelPtr> m_chans;
    std::vector<std::string> m_urls;
public:
    OneCardInferApp(AppStatis& statis, bm::BMNNContextPtr ctx, int max_batch, int skip = 0):
    m_detectorDelegate(nullptr), m_channel_num(0), m_bmctx(ctx), m_appStatis(statis)
    {
        m_dev_id    = m_bmctx->dev_id();
        m_skipN     = skip;
        m_max_batch = max_batch;
    }

    ~OneCardInferApp()
    {
        std::cout << cv::format("OneCardInfoApp (devid=%d) dtor", m_dev_id) <<std::endl;
    }

    void setDetectorDelegate(std::shared_ptr<bm::DetectorDelegate<bm::FrameBaseInfo, bm::FrameInfo>> delegate){
        m_detectorDelegate = delegate;
    }

    void start(Config& config);

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
    int addStream(std::string url, int ch) {
        std::cout << "addStream: channel id=" << ch << std::endl;
        TChannelPtr pchan = std::make_shared<TChannel>();
        pchan->decoder = new bm::StreamDecoder(ch);
        pchan->url = url;
        pchan->channel_id = ch;
        m_chans[ch] = pchan;
        m_channel_num += 1;
        return m_channel_num;
    }
};

using OneCardInferAppPtr = std::shared_ptr<OneCardInferApp>;


#endif //INFERENCE_FRAMEWORK_MAIN_H
