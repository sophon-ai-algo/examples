#ifndef INFERENCE_FRAMEWORK_WSDAN_H
#define INFERENCE_FRAMEWORK_WSDAN_H

#include "pipeline.h"
#include "bmcv_api_ext.h"
#include "common_types.h"


class WSDAN : public ruilai::ClassifyDelegate<bm::ResizeFrameInfo>  {
    bm::BMNNContextPtr m_bmctx;
    bm::BMNNNetworkPtr m_bmnet;

    float m_alpha;
    float m_beta;

    int m_net_h;
    int m_net_w;

public:
    WSDAN(bm::BMNNContextPtr bmctx);
    ~WSDAN();

    virtual int preprocess(std::vector<bm::ResizeFrameInfo> &frames) override;
    virtual int forward(std::vector<bm::ResizeFrameInfo> &frames) override;
    virtual int postprocess(std::vector<bm::ResizeFrameInfo> &frames) override;

private:
    bm::BMNNTensorPtr get_output_tensor(const std::string &name, bm::ResizeFrameInfo& frame_info, float scale);
    void extract_feature_cpu(bm::ResizeFrameInfo& frame);
};


#endif //INFERENCE_FRAMEWORK_WSDAN_H
