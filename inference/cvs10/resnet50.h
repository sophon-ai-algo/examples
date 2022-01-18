//
// Created by yuan on 11/24/21.
//

#ifndef INFERENCE_FRAMEWORK_RESNET50_H
#define INFERENCE_FRAMEWORK_RESNET50_H

#include "inference.h"
#include "bmcv_api_ext.h"
#include "common_types.h"

class Resnet : public bm::DetectorDelegate<bm::cvs10FrameBaseInfo, bm::cvs10FrameInfo>  {
    bm::BMNNContextPtr m_bmctx;
    bm::BMNNNetworkPtr m_bmnet;

    float m_alpha;
    float m_beta;

    int MAX_BATCH=1;
    int m_net_h;
    int m_net_w;

public:
    Resnet(bm::BMNNContextPtr bmctx, int max_batch);
    ~Resnet();

    virtual int preprocess(std::vector<bm::cvs10FrameBaseInfo> &in, std::vector<bm::cvs10FrameInfo> &of) override;
    virtual int forward(std::vector<bm::cvs10FrameInfo> &frames) override;
    virtual int postprocess(std::vector<bm::cvs10FrameInfo> &frames) override;

private:
    bm::BMNNTensorPtr get_output_tensor(const std::string &name, bm::cvs10FrameInfo& frame_info, float scale);
    void extract_feature_cpu(bm::cvs10FrameInfo& frame);
};


#endif //INFERENCE_FRAMEWORK_RESNET50_H
