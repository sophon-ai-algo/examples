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
#include "opencv2/opencv.hpp"
#include "cvwrapper.h"

/**
 * @brief Judge if the input is an image file.
 *
 * @param file_path Path to file
 * @return True if it is an image file.
 */
inline bool is_image_file(const std::string& file_path) {
  auto len = file_path.size();
  return (file_path.compare(len - 3, 3, "jpg")  == 0 ||
          file_path.compare(len - 3, 3, "JPG")  == 0 ||
          file_path.compare(len - 4, 4, "jpeg") == 0 ||
          file_path.compare(len - 4, 3, "JPEG") == 0);
}

/**
 * @brief Virtual base class to decode image or video.
 */
class CvDecoder {
 public:
  CvDecoder() {}
  virtual ~CvDecoder() {}

  /**
   * @brief Read an image.
   *
   * @param mat Container to receive the decoded image.
   * @return Program state
   *     @retval true  Success
   *     @retval false Failure
   */
  virtual bool read(cv::Mat& mat) = 0;

  /**
   * @brief Create an decocder instance.
   *
   * @param file_path Path to image or video
   * @return Pointer of a decoder instance. Need to delete after using it.
   */
  static CvDecoder* create(const std::string& file_path);
};

/**
 * @brief Image decoder.
 */
class CvImageDecoder : public CvDecoder {
 public:
  /**
   * @brief Constructor.
   *
   * @param file_paths Paths to image files
   */
  CvImageDecoder(const std::vector<std::string>& file_paths);

  /**
   * @brief Read an image.
   *
   * @param mat Container to receive the decoded image.
   * @return Program state
   *     @retval true  Success
   *     @retval false Failure
   */
  bool read(cv::Mat& mat);

 protected:
  std::vector<std::string> file_paths_;
  int curr_frame_;
};

/**
 * @brief Video decoder.
 */
class CvVideoDecoder : public CvDecoder {
public:
  /**
   * @brief Constructor.
   *
   * @param file_path Path to video file
   */
  CvVideoDecoder(const std::string& file_path);

  /**
   * @brief Read an image.
   *
   * @param mat Container to receive the decoded image.
   * @return Program state
   *     @retval true  Success
   *     @retval false Failure
   */
  bool read(cv::Mat& mat);
protected:
  cv::VideoCapture cap_;
};

#ifdef USE_BMCV
/**
 * @brief Video decoder using bmcv.
 */
class BmcvVideoDecoder : public CvVideoDecoder {
public:
  /**
   * @brief Constructor.
   *
   * @param file_path Path to video file
   * @param as_yuv    Whether decode as YUV format
   */
  BmcvVideoDecoder(const std::string& file_path, bool as_yuv=true);
};
#endif
