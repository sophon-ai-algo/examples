//
// Created by yuan on 6/9/21.
//

#ifndef INFERENCE_FRAMEWORK_FACE_EXTRACT_H
#define INFERENCE_FRAMEWORK_FACE_EXTRACT_H
#include "inference2.h"
#include "bmcv_api_ext.h"
#include "face_common.h"

class FaceExtract : public bm::DetectorDelegate<bm::FrameInfo2> {
    bm::BMNNContextPtr m_bmctx;
    bm::BMNNNetworkPtr m_bmnet;
    float m_alpha_fp32;
    float m_beta_fp32;
    float m_alpha_int8;
    float m_beta_int8;
    int MAX_BATCH=1;
    cv::Size m_inputSize;

public:
    FaceExtract(bm::BMNNContextPtr bmctx, int max_batch);
    ~FaceExtract();

    virtual int preprocess(std::vector<bm::FrameInfo2> &in) override;
    virtual int forward(std::vector<bm::FrameInfo2> &frames) override;
    virtual int postprocess(std::vector<bm::FrameInfo2> &frames) override;
private:
    bm::BMNNTensorPtr get_output_tensor(const std::string &name, bm::NetForward *inferIO, float scale=1.0);
    void extract_facefeature_cpu(bm::FrameInfo2& frame);
    int get_complex_idx(int idx, std::vector<bm::NetOutputDatum> out, int *p_frameIdx, int *prc_idx);
    void free_fwds(std::vector<bm::NetForward> &fwds);
};


#endif //INFERENCE_FRAMEWORK_FACE_EXTRACT_H
