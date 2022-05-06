#ifndef FACE_HPP_
#define FACE_HPP_
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


typedef void* FaceDetector;
typedef void* FaceExtractor;

typedef struct FaceRect {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
} FaceRect;

typedef struct FacePts {
  float x[5], y[5];
} FacePts;

typedef struct FaceInfo {
  FaceRect bbox;
  cv::Vec4f regression;
  FacePts facePts;
  double roll;
  double pitch;
  double yaw;
  double distance;
  int imgid;
} FaceInfo;


void detector_init(FaceDetector* face_detector, const std::string& model_dir);
void detector_detect(FaceDetector face_detector, const cv::Mat image, std::vector<FaceInfo>& faceInfo, int minSize, double* threshold, double factor);
void detector_dump_init(FaceDetector* face_detector, const std::string& model_dir, int max_iterations);
bool detector_dump(FaceDetector face_detector, const cv::Mat image, std::vector<FaceInfo>& faceInfo, int minSize, double* threshold, double factor);
void detector_destroy(FaceDetector face_detector);

void init_extractor(FaceExtractor* face_extractor, const std::string& model_file, const std::string& trained_file);
void do_face_extract(FaceExtractor face_extractor, const std::vector<cv::Mat>& imgs, std::vector<std::vector<float> >& features);
void destroy_extractor(FaceExtractor face_extractor);

cv::Mat align_face(const cv::Mat& src, const FaceInfo faceInfo, int width, int height);
bool compareDis(const FaceInfo& a, const FaceInfo& b);
float calc_cosine(const std::vector<float>& feature1, const std::vector<float>& feature2);

bool compareImgId(const FaceInfo& a, const FaceInfo& b);
#endif
