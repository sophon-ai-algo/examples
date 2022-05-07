#ifndef _INFERENCE_FRAMEWORK_RETINAFACE_H_
#define _INFERENCE_FRAMEWORK_RETINAFACE_H_

#include "pipeline.h"
#include "bmutility_profile.h"
#include "bmutility_pool.h"

struct RetinafaceImpl;

class Retinaface : public ruilai::DetectorDelegate<bm::FrameBaseInfo, bm::FrameInfo> {
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
        bool keep_original = true,
        float nms_threshold = 0.5,
        float conf_threshold = 0.5,
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


#endif // _INFERENCE_FRAMEWORK_RETINAFACE_H_