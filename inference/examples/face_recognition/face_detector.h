//
// Created by yuan on 2/23/21.
//

#ifndef INFERENCE_FRAMEWORK_FACE_DETECTOR_H
#define INFERENCE_FRAMEWORK_FACE_DETECTOR_H

#include "inference2.h"
#include "bmcv_api_ext.h"
#include "face_common.h"

class FaceDetector : public bm::DetectorDelegate<bm::FrameInfo2> {
    int MAX_BATCH = 4;
    bm::BMNNContextPtr bmctx_;
    bm::BMNNNetworkPtr bmnet_;
    bool               is4N_;

    double             target_size_{400};
    double             max_size_ {800};
    double             im_scale_;
    float              nms_threshold_{0.3};
    float              base_threshold_{0.05};
    std::vector<float> anchor_ratios_;
    std::vector<float> anchor_scales_;
    int                per_nms_topn_{1000};
    int                base_size_{16};
    int                min_size_{2};
    int                feat_stride_{8};
    int                anchor_num_;
    double             img_x_scale_;
    double             img_y_scale_;
    int  m_net_h, m_net_w;
public:
    FaceDetector(bm::BMNNContextPtr bmctx, int max_batch=4);
    ~FaceDetector();

    virtual int preprocess(std::vector<bm::FrameInfo2>& frame_info) override ;
    virtual int forward(std::vector<bm::FrameInfo2>& frame_info) override ;
    virtual int postprocess(std::vector<bm::FrameInfo2> &frame_info) override;
private:
    int extract_facebox_cpu(bm::FrameInfo2 &frame_info);
    bm::BMNNTensorPtr get_output_tensor(const std::string &name, bm::FrameInfo2& frame_info, float scale=1.0, int k=0);
    void generate_proposal(const float *          scores,
                           const float *          bbox_deltas,
                           const float            scale_factor,
                           const int              feat_factor,
                           const int              feat_w,
                           const int              feat_h,
                           const int              width,
                           const int              height,
                           bm::NetOutputObjects &proposals);
    void nms(const bm::NetOutputObjects &proposals,
             bm::NetOutputObjects&      nmsProposals);

    void calc_resized_HW(int image_h, int image_w, int *p_h, int *p_w);
};


#endif //INFERENCE_FRAMEWORK_FACE_DETECTOR_H
