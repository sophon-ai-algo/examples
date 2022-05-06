#include <face_detect.hpp>

int main(int argc, char** argv) {
  const std::string& proto_file = argv[1];
  const std::string& model_file = argv[2];
  const float threshold = atof(argv[3]);
  const std::string& image_file = argv[4];
  std::string donot_show = "";
  std::cout << "atgc" << argc << std::endl;
  if (argc >= 6)
    donot_show = std::string(argv[5]);
  else
    donot_show = std::string("show");

  bool show_window = (donot_show.find(std::string("noshow")) == std::string::npos);

  FaceDetector detector(proto_file, model_file);

  cv::Mat image;
  image = cv::imread(image_file);

  std::vector<FaceRect> detected;
  detector.detect(image, threshold, detected);

  for (int i = 0; i < detected.size(); ++i) {
    FaceRect rect = detected[i];
    cv::rectangle(image, cv::Rect(rect.x1, rect.y1, rect.x2 - rect.x1 + 1, rect.y2 - rect.y1 + 1),
                  cv::Scalar(255, 0, 0), 2);

    std::cout << "x1=" << rect.x1 << std::endl;
    std::cout << "y1=" << rect.y1 << std::endl;
    std::cout << "w=" << rect.x2 - rect.x1 + 1 << std::endl;
    std::cout << "h=" << rect.y2 - rect.y1 + 1 << std::endl;
  }
#ifdef INT8_MODE
  cv::imwrite("detection_int8.png", image);
#else
  cv::imwrite("detection.png", image);
#endif
  if (show_window) {
    cv::namedWindow("detection.png", CV_WINDOW_AUTOSIZE);
    cv::imshow("detection.png", image);
    std::cout << "wait any key pressed." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
  }
  return 1;
}
