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


#ifdef USE_BMCV
class BmcvPreProcessor {
 public:
  /**
   * @brief Constructor.
   *
   * @param bmcv  Reference to a Bmcv instance
   * @param scale Scale factor from float32 to int8
   */
  BmcvPreProcessor(sail::Bmcv& bmcv, int result_width, int result_height, float scale = 1.0);

  /**
   * @brief Execution function of preprocessing.
   *
   * @param input Input data
   * @param input Output data
   */
  void process(sail::BMImage& input, sail::BMImage& output);

  /**
   * @brief Execution function of preprocessing.
   *
   * @param input Input data
   * @param input Output data
   */
  void process(cv::Mat& input, sail::BMImage& output);

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
  float ab_[6];
  int resize_w;
  int resize_h;
};

template<std::size_t N>
void BmcvPreProcessor::process(sail::BMImageArray<N>& input, sail::BMImageArray<N>& output) {
  sail::BMImageArray<N> tmp = bmcv_.vpp_resize(input, resize_w, resize_h);
  bmcv_.convert_to(tmp, output,
                   std::make_tuple(std::make_pair(ab_[0], ab_[1]),
                                   std::make_pair(ab_[2], ab_[3]),
                                   std::make_pair(ab_[4], ab_[5])));
}

#endif


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


class YoloX_PostForward
{
private:
  /* data */
public:
  YoloX_PostForward(int net_w, int net_h, std::vector<int> strides);
  ~YoloX_PostForward();
  void process(float* data_ptr,std::vector<int> output_shape, std::vector<std::pair<int,int>> ost_size, 
    float threshold, float nms_threshold, std::vector<std::vector<ObjRect>> &detections);

private:
  int outlen_diml ;
  int* grids_x_;
  int* grids_y_;
  int* expanded_strides_;
  int network_width;
  int network_height;
};

