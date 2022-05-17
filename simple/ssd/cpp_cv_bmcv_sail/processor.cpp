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

#include <numeric>
#include <fstream>
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif
#include "inireader.hpp"
#include "processor.h"

bool file_exists(const std::string& file_path) {
  std::ifstream f(file_path.c_str());
  return f.good();
}

PreProcessor::PreProcessor(float scale) :
  ab_{1, -123, 1, -117, 1, -104} {
  for (int i = 0; i < 6; i ++) {
    ab_[i] *= scale;
  }
}

#if defined(USE_OPENCV)
#include <opencv2/opencv.hpp>
void PreProcessor::process(cv::Mat& input, cv::Mat& output) {
  cv::Mat tmp;
#ifdef USE_BMCV
  if (input.avOK()) {
      tmp.create(cv::Size(300, 300), CV_8UC3);
      cv::bmcv::resize(input, tmp);
  }else{
      cv::resize(input, tmp, cv::Size(300, 300), 0, 0, cv::INTER_NEAREST);
  }
#else
  cv::resize(input, tmp, cv::Size(300, 300), 0, 0, cv::INTER_NEAREST);
#endif
  tmp.convertTo(tmp, CV_32FC3);
  tmp += cv::Scalar(ab_[1], ab_[3], ab_[5]);
  tmp.convertTo(output, output.type(), ab_[0]);
}
#endif

#ifdef USE_BMCV
BmcvPreProcessor::BmcvPreProcessor(sail::Bmcv& bmcv, float scale)
    : PreProcessor(scale), bmcv_(bmcv) {
}

void BmcvPreProcessor::process(sail::BMImage& input, sail::BMImage& output) {
  // resize: bgr-packed -> bgr-planer
  sail::BMImage tmp;
  bmcv_.vpp_resize(input, tmp, 300, 300);
  // linear: bgr-planer -> bgr-planar
  bmcv_.convert_to(tmp, output,
                   std::make_tuple(std::make_pair(ab_[0], ab_[1]),
                                   std::make_pair(ab_[2], ab_[3]),
                                   std::make_pair(ab_[4], ab_[5])));
}

#ifdef USE_OPENCV
#include <opencv2/core.hpp>
void BmcvPreProcessor::process(cv::Mat& input, sail::BMImage& output) {
  int w = input.cols;
  int h = input.rows;
  // resize
  std::vector<cv::Rect> crops{cv::Rect(0, 0, w, h)};
  std::vector<cv::Mat> mats{cv::Mat(300, 300, CV_8UC3)};
  std::vector<cv::Size> sizes{cv::Size(300, 300)};
  cv::bmcv::convert(input, crops, sizes, mats);
  // mat -> bm_image
  sail::BMImage img0;
  cv::bmcv::toBMI(mats[0], &img0.data());
  // change format: nv12 -> bgr planar
  sail::BMImage img1;
  bmcv_.convert_format(img0, img1);
  // linear transform
  bmcv_.convert_to(img1, output,
                   std::make_tuple(std::make_pair(ab_[0], ab_[1]),
                                   std::make_pair(ab_[2], ab_[3]),
                                   std::make_pair(ab_[4], ab_[5])));
}
#endif
#endif

void PostProcessor::process(
    std::vector<DetectRect>& output,
    const float*             input_data,
    const std::vector<int>&  input_shape,
    int                      img_w,
    int                      img_h) {
  std::array<std::vector<DetectRect>, 1> outputs;
  process(outputs, input_data, input_shape, img_w, img_h);
  output = std::move(outputs[0]);
}

std::vector<DetectRect> PostProcessor::get_reference(
    const std::string& compare_path) {
  std::vector<DetectRect> reference;
  if (!compare_path.empty()) {
    INIReader reader(compare_path);
    if (reader.ParseError()) {
      spdlog::error("Can't load reference file: {}!", compare_path);
      std::terminate();
    }
    int num = reader.GetInteger("summary", "num", 0);
    for (int i = 0 ; i < num; ++i) {
      DetectRect box;
      std::string section("object_");
      section += std::to_string(i);
      box.x1 = reader.GetReal(section, "x1", 0.0);
      box.y1 = reader.GetReal(section, "y1", 0.0);
      box.x2 = reader.GetReal(section, "x2", 0.0);
      box.y2 = reader.GetReal(section, "y2", 0.0);
      box.score = reader.GetReal(section, "score", 0.0);
      box.class_id = reader.GetReal(section, "category", 0);
      reference.push_back(box);
    }
  }
  return std::move(reference);
}

std::vector<std::vector<DetectRect>> PostProcessor::get_reference_4b(
    const std::string& compare_path) {
  std::vector<std::vector<DetectRect>> reference;
  if (!compare_path.empty()) {
    INIReader reader(compare_path);
    if (reader.ParseError()) {
      spdlog::error("Can't load reference file: {}!", compare_path);
      std::terminate();
    }
    std::string number("num_");
    for (int i = 0; i < 4; ++i) {
      std::vector<DetectRect> ref;
      int num = reader.GetInteger("summary", number + std::to_string(i), 0);
      std::string sec("frame_");
      sec += std::to_string(i);
      sec += "_object_";
      for (int j = 0 ; j < num; ++j) {
        DetectRect box;
        std::string section = sec + std::to_string(j);
        box.x1 = reader.GetReal(section, "x1", 0.0);
        box.y1 = reader.GetReal(section, "y1", 0.0);
        box.x2 = reader.GetReal(section, "x2", 0.0);
        box.y2 = reader.GetReal(section, "y2", 0.0);
        box.score = reader.GetReal(section, "score", 0.0);
        box.class_id = reader.GetReal(section, "category", 0);
        ref.push_back(box);
      }
      reference.push_back(ref);
    }
  }
  return std::move(reference);
}

bool PostProcessor::compare(
    std::vector<DetectRect>& reference,
    std::vector<DetectRect>& result,
    int                      loop_id) {
  if (reference.empty()) {
    //spdlog::info("No verify_files file or verify_files err.");
    return true;
  }
  if (loop_id > 0) {
    return true;
  }
  if (reference.size() != result.size()) {
    spdlog::error("Expected deteted number is {}, but detected {}!",
                  reference.size(), result.size());
    return false;
  }
  bool ret = true;
  std::string message("Category: {}, Score: {}, Box: [{}, {}, {}, {}]");
  std::string fail_info("Compare failed! Expect: ");
  fail_info += message;
  std::string ret_info("Result Box: ");
  ret_info += message;
  for (size_t i = 0; i < result.size(); ++i) {
    auto& box = result[i];
    auto& ref = reference[i];
    if (!(box == ref)) {
      spdlog::error(fail_info.c_str(), ref.class_id, ref.score,
                    ref.x1, ref.y1, ref.x2, ref.y2);
      spdlog::info(ret_info.c_str(), box.class_id, box.score,
                   box.x1, box.y1, box.x2, box.y2);
      ret = false;
    }
  }
  return ret;
}

bool PostProcessor::compare_4b(
    std::vector<std::vector<DetectRect>>&   reference,
    std::array<std::vector<DetectRect>, 4>& result,
    int                                     loop_id) {
  if (reference.empty()) {
    //spdlog::info("No verify_files file or verify_files err.");
    return true;
  }
  if (loop_id > 0) {
    return true;
  }
  if (reference.size() != result.size()) {
    spdlog::error("Expected frame number is {}, but get {}!",
                  reference.size(), result.size());
    return false;
  }
  bool ret = true;
  std::string message("[Frame {}] Category: {}, Score: {},");
  message += ("Box: [{}, {}, {}, {}]");
  std::string fail_info("Compare failed! Expect: ");
  fail_info += message;
  std::string ret_info("Result: ");
  ret_info += message;
  for (size_t i = 0; i < result.size(); ++i) {
    auto& boxes = result[i];
    auto& refs = reference[i];
    if (refs.size() != boxes.size()) {
      std::string msg("Expected deteted number for Frame {} is {},");
      msg += (" but detected {}!");
      spdlog::error(msg.c_str(), i, refs.size(), boxes.size());
      return false;
    }
    for (size_t j = 0; j < boxes.size(); ++j) {
      auto& box = boxes[j];
      auto& ref = refs[j];
      if (!(box == ref)) {
        spdlog::error(fail_info.c_str(), i, ref.class_id, ref.score,
                      ref.x1, ref.y1, ref.x2, ref.y2);
        spdlog::info(ret_info.c_str(), i, box.class_id, box.score,
                     box.x1, box.y1, box.x2, box.y2);
        ret = false;
      }
    }
  }
  return ret;
}
