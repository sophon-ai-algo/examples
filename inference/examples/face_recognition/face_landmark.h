//
// Created by yuan on 6/9/21.
//

#ifndef INFERENCE_FRAMEWORK_FACE_LANDMARK_H
#define INFERENCE_FRAMEWORK_FACE_LANDMARK_H

#include "inference2.h"
#include "bmcv_api_ext.h"
#include "face_common.h"

class FaceLandmark : public bm::DetectorDelegate<bm::FrameInfo2> {
    int MAX_BATCH = 4;
    bm::BMNNContextPtr m_bmctx;
    bm::BMNNNetworkPtr m_bmnet;
    cv::Size m_inputSize;
    const double m_alpha_int8 = 1.003921316;
    const double m_beta_int8  = -127.5 * 1.003921316;
    const double m_alpha_fp32 = 0.0078125;
    const double m_beta_fp32  = -127.5 * 0.0078125;

public:
    FaceLandmark(bm::BMNNContextPtr bmctx, int maxBatch=4);

    ~FaceLandmark();

    virtual int preprocess(std::vector <bm::FrameInfo2> &frame_info) override;

    virtual int forward(std::vector <bm::FrameInfo2> &frame_info) override;

    virtual int postprocess(std::vector <bm::FrameInfo2> &frame_info) override;

private:
    bm::BMNNTensorPtr get_output_tensor(const std::string &name, bm::NetForward *inferIO, float scale=1.0);
    int forward_subnet(std::vector<bm::NetForward> &ios);
    void free_fwds(std::vector<bm::NetForward> &ios);
    int get_complex_idx(int idx, std::vector<bm::NetOutputDatum> out, int *p_frameIdx, int *prc_idx);

};

#endif //INFERENCE_FRAMEWORK_FACE_LANDMARK_H
