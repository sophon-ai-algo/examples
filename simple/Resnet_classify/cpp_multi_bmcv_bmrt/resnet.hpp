#ifndef RESNET_HPP
#define RESNET_HPP

#include <string>
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
// Define USE_FFMPEG for enabling FFMPEG related funtions in bm_wrapper.hpp
#define USE_FFMPEG 1
#include "bm_wrapper.hpp"
#include "utils.hpp"

#define MAX_BATCH 4

#define INPUT_WIDTH 224
#define INPUT_HEIGHT 224

struct ObjRect {
  int class_id;
  float score;
  float x1;
  float y1;
  float x2;
  float y2;
};

class RESNET {
public:
  RESNET(bm_handle_t bm_handle, const std::string bmodel);
  ~RESNET();
  void preForward(std::vector<bm_image> &input);
  void forward();
  void postForward(std::vector<bm_image> &input, std::vector<std::vector<ObjRect>> &detections);
  void enableProfile(TimeStamp *ts);

private:
  void preprocess_bmcv (std::vector<bm_image> &input);

  // handle of runtime contxt
  void *p_bmrt_;

  // handle of low level device 
  bm_handle_t bm_handle_;

  // model info 
  const bm_net_info_t *net_info_;

  // indicate current bmodel type INT8 or FP32
  bool is_int8_;

  // buffer of inference results
  float *output_;

  // input image shape used for inference call
  bm_shape_t input_shape_;

  // bm image objects for storing intermediate results
  bm_image linear_trans_bmcv_[MAX_BATCH];

  // linear transformation arguments of BMCV
  bmcv_convert_to_attr linear_trans_param_;

  // for profiling
  TimeStamp *ts_ = NULL;
};

#endif /* RESNET_HPP */
