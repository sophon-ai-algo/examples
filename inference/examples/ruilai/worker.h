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
#include "retinaface.h"
#include "resnet50.h"
#include "common_types.h"
#include "mobilenetv2.h"


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
    std::shared_ptr<ruilai::DetectorDelegate<bm::FrameBaseInfo, bm::FrameInfo>> m_detectorDelegate;
    std::shared_ptr<ruilai::ClassifyDelegate<bm::ResizeFrameInfo>> m_classifyDelegate224;
    std::vector<std::shared_ptr<ruilai::ClassifyDelegate<bm::ResizeFrameInfo>>> m_vClassifyDelegate320;

    bm::TimerQueuePtr m_timeQueue;
    bm::TimerQueuePtr m_callbackQueue;
    bm_handle_t m_handle;
    int m_channel_start;
    int m_channel_num;
    int m_dev_id;
    int m_skipN;
    std::string m_output_url;
    std::mutex m_mutex;
    std::map<uint64_t, float> m_image_score_record;
    std::function<void(uint64_t, bool, float)> m_img_result_cb_func;
    ruilai::BMInferencePipe<bm::FrameBaseInfo, bm::FrameInfo> m_inferPipe;
    ruilai::ClassifyPipe<bm::ResizeFrameInfo> m_224ClassifyPipe;
    std::vector<ruilai::ClassifyPipe<bm::ResizeFrameInfo>> m_v320ClassifyPipes;

    std::map<int, TChannelPtr> m_chans;
    std::vector<std::string> m_urls;
    std::shared_ptr<BlockingQueue<bm::CropFrameInfo>> m_resizeQueue;
    WorkerPool<bm::CropFrameInfo> m_resizeWorkerPool;
;

public:
    OneCardInferApp(
        AppStatis& statis,bm::VideoUIAppPtr gui, bm::TimerQueuePtr tq, bm::BMNNHandlePtr handle,
        int start_index, int num, int resize_queue_num, int skip = 0);

    ~OneCardInferApp()
    {
        std::cout << cv::format("OneCardInfoApp (devid=%d) dtor", m_dev_id) <<std::endl;
    }

    void setDetectorDelegate(std::shared_ptr<ruilai::DetectorDelegate<bm::FrameBaseInfo, bm::FrameInfo>> delegate) {
        m_detectorDelegate = delegate;
    }
    void setClassifyDelegate_224(std::shared_ptr<ruilai::ClassifyDelegate<bm::ResizeFrameInfo>> delegate) {
        m_classifyDelegate224 = delegate;
    }
    void setClassifyDelegate_320(std::shared_ptr<ruilai::ClassifyDelegate<bm::ResizeFrameInfo>> delegate) {
        m_vClassifyDelegate320.push_back(delegate);
    }

    void start(const std::vector<std::string>& vct_urls, Config& config);

    template<typename T>
    inline void loadConfig(T& param, Config& config) {
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
    void initClassifyPipes(Config& config);
    void unifyResizeProcess(std::vector<bm::CropFrameInfo> &items,
                            std::vector<bm_image>& resized_image_224,
                            std::vector<bm_image>& resized_image_320);
    inline int pushFrame(bm::FrameBaseInfo *frame) { m_inferPipe.push_frame(frame); }
    inline void setImgResultCallback(std::function<void(int, bool, float score)> func) { m_img_result_cb_func = func; }
};

using OneCardInferAppPtr = std::shared_ptr<OneCardInferApp>;


#endif //INFERENCE_FRAMEWORK_MAIN_H
