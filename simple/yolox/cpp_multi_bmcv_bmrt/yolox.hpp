#ifndef YOLOX_HPP
#define YOLOX_HPP

#include <string>

#ifndef USE_OPENCV
#define USE_OPENCV 1
#endif
#ifndef USE_FFMPEG
#define USE_FFMPEG 1
#endif
#include "bm_wrapper.hpp"
#include "utils.hpp"

struct ObjRect
{
    unsigned int class_id;
    float score;
    float left;
    float top;
    float right;
    float bottom;
    float width;
    float height;
};

class YOLOX{
public:
    YOLOX(bm_handle_t& bm_handle, const std::string bmodel, std::vector<int> strides);
    ~YOLOX();
    int preForward(std::vector<bm_image> &input);
    void forward(float threshold, float nms_threshold);
    void postForward(std::vector<bm_image> &input, std::vector<std::vector<ObjRect>> &detections);
    void enableProfile(TimeStamp *ts);
    bool getPrecision();
    int getInputBatchSize();
private:
    void preprocess_bmcv(std::vector<bm_image> &input);
    void get_source_label(float* data_ptr, int classes, float &score, int &class_id);

    //handel of runtime contxt
    void *p_bmrt_;

    //handel of low level device
    bm_handle_t bm_handle_;

    //model info
    const bm_net_info_t *net_info_;

    //indicate current bmodel type INT8 or FP32
    bool is_int8_;

    //output type INT8 or FP32
    bool output_is_int8_;
    float output_scale_;


    float threshold_;
    float nms_threshold_;

    //buffer of  inference results
    float *output_;

    // input image shape used for inference call
    bm_shape_t input_shape_;
    // output shape
    bm_shape_t output_shape_;

    // bm image objects for storing intermediate results
    bm_image *resize_bmcv_;
    bm_image *linear_trans_bmcv_;

    // crop arguments of BMCV
    bmcv_rect_t crop_rect_;

    // linear transformation arguments of BMCV
    bmcv_convert_to_attr linear_trans_param_;

    // for profiling
    TimeStamp *ts_;

    int input_width;
    int input_height;
    int output_size;

    std::string net_name_;

    int* grids_x_;
    int* grids_y_;
    int* expanded_strides_;
    int classes_;
    int outlen_dim1_;
    int channels_resu_;

    //尺度变换
    float scale_x;
    float scale_y;

    int max_batch_;
};



#endif /* YOLOX_HPP */