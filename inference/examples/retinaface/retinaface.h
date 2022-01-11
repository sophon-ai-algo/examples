#pragma once

#include "inference.h"
#include "bmutility_profile.h"
#include "bmutility_pool.h"

struct RetinafaceImpl;

class Retinaface : public bm::DetectorDelegate<bm::FrameBaseInfo, bm::FrameInfo> {
protected:
    RetinafaceImpl *impl_;
    bm::Watch *w_;
    std::vector<bm::NetOutputObject> parse_boxes(
        size_t input_width,
        size_t input_height,
        const float *cls_data,
        const float *land_data,
        const float *loc_data,
        const bm::FrameBaseInfo &frame);
    void extract_facebox_cpu(bm::FrameInfo &frame_info);
    bm::BMNNTensorPtr get_output_tensor(const std::string &name, bm::FrameInfo &frame_info);

public:
    Retinaface(
        bm::BMNNContextPtr bmctx,
        bool keep_original = false,
        float nms_threshold = 0.5,
        float conf_threshold = 0.6,
        std::string net_name = "",
        bm::Watch *watch = nullptr);
    ~Retinaface();

    virtual void decode_process(bm::FrameBaseInfo &) override;
    virtual int preprocess(
        std::vector<bm::FrameBaseInfo> &frames,
        std::vector<bm::FrameInfo> &frame_info) override;
    virtual int forward(std::vector<bm::FrameInfo> &frame_info) override;
    virtual int postprocess(std::vector<bm::FrameInfo> &frame_info) override;
    bm_image read_image(bm::FrameBaseInfo &frame);
};

class RetinafaceEval : public Retinaface
{
public:
    RetinafaceEval(
        bm::BMNNContextPtr bmctx,
        size_t target_size = 1600,
        bool keep_original = false,
        float nms_threshold = 0.5,
        float conf_threshold = 0.1,
        std::string net_name = "");
    virtual int preprocess(
        std::vector<bm::FrameBaseInfo> &frames,
        std::vector<bm::FrameInfo> &frame_info) override;
};
