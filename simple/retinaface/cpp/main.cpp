//
//  main.cpp
//  NelivaSDK
//
//  Created by Bitmain on 2020/11/5.
//  Copyright © 2020年 AnBaolei. All rights reserved.
//

#include <iostream>
#include <chrono>
#include <mutex>
#include <thread>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include "face_detection.hpp"

using namespace std;
using namespace cv;

static void save_imgs(const std::vector<std::vector<stFaceRect> >& results,
                              const vector<cv::Mat>& batch_imgs,
                              const vector<string>& batch_names,
                              string save_foler) {
  for (size_t i = 0; i < batch_imgs.size(); i++) {
     cv::Mat img = batch_imgs[i];
    for (size_t j = 0; j < results[i].size(); j++) {
      Rect rc;
      rc.x = results[i][j].left;
      rc.y = results[i][j].top;
      rc.width = results[i][j].right - results[i][j].left + 1;
      rc.height = results[i][j].bottom - results[i][j].top + 1;
      cv::rectangle(img, rc, cv::Scalar(0, 0, 255), 2, 1, 0);
      for (size_t k = 0; k < 5; k++) {
        cv::circle(img, Point(results[i][j].points_x[k],
                  results[i][j].points_y[k]), 1, cv::Scalar(255, 0, 0), 3);
      }
    }
    string save_name = save_foler + "/" + batch_names[i];
    imwrite(save_name, img);
  }
}

int main(int argc, const char * argv[]) {
  if (argc < 3) {
    cout << "USAGE:" << endl;
    cout << "  " << argv[0] << " <input_mode> <image_list/video_list> <bmodel path> " << endl;
    exit(1);
  }

  int input_mode = atoi(argv[1]); // 0 image 1 video
  string image_list = argv[2];
  string bmodel_folder_path = argv[3];
  int device_id = 0;

  string save_foler = "result_imgs";
  if (0 != access(save_foler.c_str(), 0)) {
    system("mkdir -p result_imgs");
  }

  shared_ptr<FaceDetection> face_detection_share_ptr(
                    new FaceDetection(bmodel_folder_path, device_id));
  FaceDetection* face_detection_ptr = face_detection_share_ptr.get();

  struct timeval tpstart, tpend;
  float timeuse = 0.f;
  char image_path[1024] = {0};
  ifstream fp_img_list(image_list);
  int batch_size = face_detection_ptr->batch_size();

  if (0 == input_mode) { // image mode
    vector<cv::Mat> batch_imgs;
    vector<string> batch_names;
    while(fp_img_list.getline(image_path, 1024)) {
      string img_full_path = image_path;
      size_t index = img_full_path.rfind("/");
      string img_name = img_full_path.substr(index + 1);
      gettimeofday(&tpstart, NULL);
      Mat img = imread(image_path, cv::IMREAD_COLOR, 0);
      if (img.empty()) {
        cout << "read image " << image_path << "error!" << endl;
        exit(1);
      }
      batch_imgs.push_back(img);
      batch_names.push_back(img_name);
      if (static_cast<int>(batch_imgs.size()) == batch_size) {
        std::vector<std::vector<stFaceRect> > results;
        face_detection_ptr->run(batch_imgs, results);
        gettimeofday(&tpend, NULL);
        timeuse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
        timeuse /= 1000;
        cout << "detect used time: " << timeuse << " ms" << endl;
        save_imgs(results, batch_imgs, batch_names, save_foler);
        batch_imgs.clear();
        batch_names.clear();
        gettimeofday(&tpstart, NULL);
      }
    }
  } else { // video mode
    vector <cv::VideoCapture> caps;
    vector <string> cap_srcs;
    while(fp_img_list.getline(image_path, 1024)) {
      cv::VideoCapture cap(image_path);
      caps.push_back(cap);
      cap_srcs.push_back(image_path);
    }

    if ((int)caps.size() != batch_size) {
      cout << "video num should equal model's batch size" << endl;
      exit(1);
    }

    uint32_t batch_id = 0;
    const uint32_t run_frame_no = 200; 
    uint32_t frame_id = 0;
    while(1) {
      if (frame_id == run_frame_no) {
        break;
      }
      vector<cv::Mat> batch_imgs;
      vector<string> batch_names;
      gettimeofday(&tpstart, NULL);
      for (size_t i = 0; i < caps.size(); i++) {
         if (caps[i].isOpened()) {
           int w = int(caps[i].get(cv::CAP_PROP_FRAME_WIDTH));
           int h = int(caps[i].get(cv::CAP_PROP_FRAME_HEIGHT));
           cv::Mat img;
           caps[i] >> img;
           if (img.rows != h || img.cols != w) {
             break;
           }
           batch_imgs.push_back(img);
           batch_names.push_back(to_string(batch_id) + "_" +
                            to_string(i) + "_video.jpg");
           batch_id++;
         }else{
           cout << "VideoCapture " << i << " "
                   << cap_srcs[i] << " open failed!" << endl;
         }
      }
      if ((int)batch_imgs.size() < batch_size) {
        break;
      }
      std::vector<std::vector<stFaceRect> > results;
      face_detection_ptr->run(batch_imgs, results);
      gettimeofday(&tpend, NULL);
      timeuse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
      timeuse /= 1000;
      cout << "detect used time: " << timeuse << " ms" << endl;
      save_imgs(results, batch_imgs, batch_names, save_foler);
      batch_imgs.clear();
      batch_names.clear();
      frame_id += 1;
    }

  }
  return 0;
}
