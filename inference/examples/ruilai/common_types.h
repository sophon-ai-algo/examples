#ifndef INFERENCE_FRAMEWORK_COMMON_TYPES_H
#define INFERENCE_FRAMEWORK_COMMON_TYPES_H

#include <glog/logging.h>
#include "bmcv_api_ext.h"

#define call(fn, ...) \
    do { \
        auto ret = fn(__VA_ARGS__); \
        if (ret != BM_SUCCESS) \
        { \
            LOG(ERROR) << #fn << " failed " << ret; \
            throw std::runtime_error("api error"); \
        } \
    } while (false);
    
namespace bm {  
struct ResizeFrameInfo{
    int chan_id;
    std::vector<uint64_t> v_seq;
    int batch_size;
    std::vector<bm_image> v_resized_imgs;
    std::vector<bm_tensor_t> input_tensors;
    std::vector<bm_tensor_t> output_tensors;
    ResizeFrameInfo() : chan_id(0), batch_size(0) {
        v_seq.clear();
        v_resized_imgs.clear();
    }
};
struct CropFrameInfo {
    int chan_id;
    uint64_t seq;
    bm_image crop_img_224;
    bm_image crop_img_320;

    CropFrameInfo() : chan_id(0), seq(0) {
        memset(&crop_img_224, 0, sizeof(bm_image));
        memset(&crop_img_320, 0, sizeof(bm_image));

    }
};
}



#endif