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

#include "spdlog/spdlog.h"
#include "cvdecoder.h"

CvDecoder* CvDecoder::create(const std::string& file_path) {
  return is_image_file(file_path) ?
    static_cast<CvDecoder*>(new CvImageDecoder({file_path})) :
#ifdef USE_BMCV
    static_cast<CvDecoder*>(new BmcvVideoDecoder(file_path, false));
#else
    static_cast<CvDecoder*>(new CvVideoDecoder(file_path));
#endif
}

CvImageDecoder::CvImageDecoder(const std::vector<std::string>& file_paths)
    : CvDecoder(), file_paths_(file_paths), curr_frame_(0) {
}

bool CvImageDecoder::read(cv::Mat& mat) {
  mat = cv::imread(file_paths_[curr_frame_]);
  if (++curr_frame_ % file_paths_.size() == 0) {
    curr_frame_ = 0;
  }
  return true;
}

CvVideoDecoder::CvVideoDecoder(const std::string& file_path)
    : CvDecoder(), cap_(file_path) {
  if (!cap_.isOpened()) {
    spdlog::error("open stream {} failed!", file_path);
    return;
  }

// For OpenCV2
#if CV_VERSION_EPOCH == 2
  printf("CV_VERSION:%d.%d\n", CV_VERSION_MAJOR, CV_VERSION_MINOR);
  int w = int(cap_.get(CV_CAP_PROP_FRAME_WIDTH));
  int h = int(cap_.get(CV_CAP_PROP_FRAME_HEIGHT));
#else
  int w = int(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
  int h = int(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
#endif

  spdlog::info("resolution of input stream: {}, {}", h, w);
}

bool CvVideoDecoder::read(cv::Mat& mat) {
  return cap_.read(mat);
}

#ifdef USE_BMCV
BmcvVideoDecoder::BmcvVideoDecoder(
    const std::string& file_path,
    bool               as_yuv)
    : CvVideoDecoder(file_path) {
  // set output format to YUV-nv12
  if (as_yuv) {
    cap_.set(cv::CAP_PROP_OUTPUT_YUV, 1.0);
  }
}
#endif
