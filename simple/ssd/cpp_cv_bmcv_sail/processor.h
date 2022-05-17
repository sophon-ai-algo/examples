/* Copyright 2016-2022 by Sophgo Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

==============================================================================*/

#pragma once
#include <string>
#include <vector>
#include <array>
#include "cvwrapper.h"

/**
 * @brief Judge if a file exists..
 *
 * @param file_path Path to the file
 * @return True for exist, false for not.
 */
bool file_exists(const std::string& file_path);

/**
 * @brief Struct to hold detetion result.
 */
struct DetectRect {
  int class_id;
  float score;
  float x1;
  float y1;
  float x2;
  float y2;

  bool operator==(const DetectRect& t) const {
    return (t.class_id == this->class_id &&
            std::abs(t.x1 - this->x1) < 2 &&
            std::abs(t.y1 - this->y1) < 2 &&
            std::abs(t.x2 - this->x2) < 2 &&
            std::abs(t.y2 - this->y2) < 2 &&
            std::abs(t.score - this->score) < 1.8e-1);
  }
};

class PreProcessor {
 public:
  /**
   * @brief Constructor.
   *
   * @param scale Scale factor from float32 to int8
   */
  PreProcessor(float scale = 1.0);

  /**
   * @brief Execution function of preprocessing.
   *
   * @param input Input data
   * @param input Output data
   */
  void process(cv::Mat& input, cv::Mat& output);

 protected:
  float ab_[6];
};

#ifdef USE_BMCV
class BmcvPreProcessor : public PreProcessor {
 public:
  /**
   * @brief Constructor.
   *
   * @param bmcv  Reference to a Bmcv instance
   * @param scale Scale factor from float32 to int8
   */
  BmcvPreProcessor(sail::Bmcv& bmcv, float scale = 1.0);

  /**
   * @brief Execution function of preprocessing.
   *
   * @param input Input data
   * @param input Output data
   */
  void process(sail::BMImage& input, sail::BMImage& output);

  /**
   * @brief Execution function of preprocessing for multiple images.
   *
   * @param input Input data
   * @param input Output data
   */
  template<std::size_t N>
  void process(sail::BMImageArray<N>& input, sail::BMImageArray<N>& output);

#ifdef USE_OPENCV
  /**
   * @brief Execution function of preprocessing.
   *
   * @param input Input data
   * @param input Output data
   */
  void process(cv::Mat& input, sail::BMImage& output);
#endif

 protected:
  sail::Bmcv& bmcv_;
};

template<std::size_t N>
void BmcvPreProcessor::process(
    sail::BMImageArray<N>& input,
    sail::BMImageArray<N>& output) {
  sail::BMImageArray<N> tmp = bmcv_.vpp_resize(input, 300, 300);
  bmcv_.convert_to(tmp, output,
                   std::make_tuple(std::make_pair(ab_[0], ab_[1]),
                                   std::make_pair(ab_[2], ab_[3]),
                                   std::make_pair(ab_[4], ab_[5])));
}
#endif

class PostProcessor {
 public:
  /**
   * @brief Constructor.
   *
   * @param threshold Threshold
   */
  PostProcessor(float threshold) : threshold_(threshold) {}

  /**
   * @brief Execution function of postprocessing.
   *
   * @param output      Detected result
   * @param input_data  Input data
   * @param input_shape Input shape
   * @param img_w       Image width
   * @param img_h       Image height
   */
  void process(
      std::vector<DetectRect>& output,
      const float*             input_data,
      const std::vector<int>&  input_shape,
      int                      img_w,
      int                      img_h);

  /**
   * @brief Execution function of postprocessing for multiple images.
   *
   * @param output      Detected result
   * @param input_data  Input data
   * @param input_shape Input shape
   * @param img_w       Image width
   * @param img_h       Image height
   */
  template<std::size_t N>
  void process(
      std::array<std::vector<DetectRect>, N>& output,
      const float*                            input_data,
      const std::vector<int>&                 input_shape,
      int                                     img_w,
      int                                     img_h);

  /**
   * @brief Get correct result from given file.
   *
   * @param compare_path Path to correct result file
   * @return correct result
   */
  std::vector<DetectRect> get_reference(const std::string& compare_path);

  /**
   * @brief Get correct result from given file, with input batch size is 4.
   *
   * @param compare_path Path to correct result file
   * @return correct result
   */
  std::vector<std::vector<DetectRect>> get_reference_4b(
      const std::string& compare_path);

  /**
   * @brief Compare result.
   *
   * @param reference Correct result
   * @param result    Output result
   * @param loop_id   Loop iterator number
   * @return True for success and false for failure
   */
  bool compare(
      std::vector<DetectRect>& reference,
      std::vector<DetectRect>& result,
      int                      loop_id);

  /**
   * @brief Compare result, with input batch size is 4.
   *
   * @param reference Correct result
   * @param result    Output result
   * @param loop_id   Loop iterator number
   * @return True for success and false for failure
   */
  bool compare_4b(
      std::vector<std::vector<DetectRect>>&   reference,
      std::array<std::vector<DetectRect>, 4>& result,
      int                                     loop_id);

 protected:
  float threshold_;
};

template<std::size_t N>
void PostProcessor::process(
    std::array<std::vector<DetectRect>, N>& output,
    const float*                            input_data,
    const std::vector<int>&                 input_shape,
    int                                     img_w,
    int                                     img_h) {
  if(input_shape.size() < 4){
    spdlog::error("input_shape.size error: {}",input_shape.size());
    return;
  }

  int data_size = std::accumulate(input_shape.begin(), input_shape.end(),
                                  1, std::multiplies<int>());
  int step = input_shape[3];
  for (int i = 0; i < data_size; i += step) {
    const float* proposal = &input_data[i];
    if (proposal[2] < threshold_) {
      continue;
    }
    size_t idx = static_cast<size_t>(proposal[0]);
    if (idx >= N) break;

    output[idx].push_back(DetectRect());
    output[idx].back().class_id = proposal[1];
    output[idx].back().score = proposal[2];
    output[idx].back().x1 = proposal[3] * img_w;
    output[idx].back().y1 = proposal[4] * img_h;
    output[idx].back().x2 = proposal[5] * img_w;
    output[idx].back().y2 = proposal[6] * img_h;
  }
}
