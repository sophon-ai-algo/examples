#ifndef FACE_COMMON_HPP_
#define FACE_COMMON_HPP_

#include <vector>
#include <opencv2/opencv.hpp>


#include <ufw/ufw.hpp>
using namespace ufw;


typedef struct FaceRect {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
} FaceRect;

typedef struct FacePts {
  float x[5];
  float y[5];
  float score;
} FacePts;

#endif