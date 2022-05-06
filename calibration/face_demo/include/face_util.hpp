#ifndef FACE_UTIL_HPP_
#define FACE_UTIL_HPP_
#include <vector>
#include <opencv2/opencv.hpp>
#include "face_common.hpp"

bool compareArea(const FaceRect& a, const FaceRect& b);
bool compareScore(const FaceRect& a, const FaceRect& b);
cv::Mat align_face(const cv::Mat& src, const FaceRect rect,
                   const FacePts facePt, int width, int height);
float calc_cosine(const std::vector<float>& feature1,
                  const std::vector<float>& feature2);
float calc_cosine(const std::vector<char>& feature1,
                  const std::vector<char>& feature2);
float calc_cosine(const std::vector<char>& feature1,
                  const std::vector<float>& feature2);
#endif
