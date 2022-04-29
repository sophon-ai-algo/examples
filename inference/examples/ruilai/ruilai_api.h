#ifndef _INFERENCE_FRAMEWORK_RUILAIAPI_H_
#define _INFERENCE_FRAMEWORK_RUILAIAPI_H_

#include <vector>
#include <atomic>
#include "pipeline.h"
#include "bmutility_profile.h"
#include "bmutility_pool.h"
#include "worker.h"

struct JPGUint {
    const unsigned char* jpeg_data;
    int len;
    uint64_t image_id; 
};

using ImgResultCallBackFunc = std::function<void(uint64_t, bool, float)>;

class RuiLaiAPIWrapper {
    using AppPtr = std::shared_ptr<OneCardInferApp>;
private:
    int m_cards;                    // 芯片数
    std::vector<AppPtr> m_vApps;    // 每个芯片对一个的app实例
    std::shared_ptr<BlockingQueue<JPGUint>> m_jpegQueue;
    WorkerPool<JPGUint> m_jpegWorkerPool;
    std::atomic<uint64_t> m_imageIndex;
    ImgResultCallBackFunc m_callbackFunc;
    AppStatis m_appStatis;
public:
    RuiLaiAPIWrapper(int card_num,
                     std::string retinaface_bmodel, float face_threshold,  // 这些threshold接口层先暴露给客户
                     std::string cls_bmodel1, float cls1_threshold,        // 底层还未实现对应可配参数
                     std::string cls_bmodel2, float cls2_threshold,        // 先保留下来
                     std::string cls_bmodel3, float cls3_threshold,
                     std::string cls_bmodel4, float cls4_threshold,
                     ImgResultCallBackFunc func,
                     std::string config_file="./cameras.json");
    virtual ~RuiLaiAPIWrapper();
    // TODO
    uint64_t Infer(const unsigned char *jpeg_data, int len);
    // TODO
    uint64_t InferBatch() {}

};


#endif // _INFERENCE_FRAMEWORK_RUILAIAPI_H_