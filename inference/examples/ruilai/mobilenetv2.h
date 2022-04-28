#ifndef INFERENCE_FRAMEWORK_MOBILENETV2_H
#define INFERENCE_FRAMEWORK_MOBILENETV2_H

#include "pipeline.h"
#include "bmcv_api_ext.h"
#include "common_types.h"


class MobileNetV2 : public ruilai::ClassifyDelegate<bm::ResizeFrameInfo>  {
    bm::BMNNContextPtr m_bmctx;
    bm::BMNNNetworkPtr m_bmnet;

    float m_alpha;
    float m_beta;

    int m_net_h;
    int m_net_w;

public:
    MobileNetV2(bm::BMNNContextPtr bmctx);
    ~MobileNetV2();

    virtual int preprocess(std::vector<bm::ResizeFrameInfo> &frames) override;
    virtual int forward(std::vector<bm::ResizeFrameInfo> &frames) override;
    virtual int postprocess(std::vector<bm::ResizeFrameInfo> &frames) override;

private:
    bm::BMNNTensorPtr get_output_tensor(const std::string &name, bm::ResizeFrameInfo& frame_info, float scale);
    void extract_feature_cpu(bm::ResizeFrameInfo& frame);
};


#endif //INFERENCE_FRAMEWORK_MOBILENETV2_H
