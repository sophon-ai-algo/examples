#ifndef FACE_DETECT_HPP_
#define FACE_DETECT_HPP_
#include <vector>
#include <opencv2/opencv.hpp>
#include "boost/make_shared.hpp"
#include <face_common.hpp>

class FaceDetector {
 public:
  FaceDetector(const std::string& proto_file, const std::string& model_file);

  ~FaceDetector(){
      delete net_;
  }
  void detect(const cv::Mat& image, const float threshold,
                    std::vector<FaceRect>& faceRects);

 private:
  void calculatePyramidScale(const cv::Mat& image, cv::Mat& resizedImage);
  void generateProposals(const float* scores, const float* bbox_deltas,
                               const float scale_factor, const int feat_factor,
                               const int feat_w, const int feat_h,
                               const int width, const int height,
                               std::vector<FaceRect>& proposals);
  void nms(const std::vector<FaceRect>& proposals,
                 std::vector<FaceRect>& nmsProposals);
  void initParameters(void);

  Net<float> *net_;

  double target_size_;
  double max_size_;
  double im_scale_;
  float nms_threshold_;
  float base_threshold_;
  std::vector<float> anchor_ratios_;
  std::vector<float> anchor_scales_;
  int per_nms_topn_;
  int base_size_;
  int min_size_;
  int feat_stride_;
  int anchor_num_;
};

#endif
