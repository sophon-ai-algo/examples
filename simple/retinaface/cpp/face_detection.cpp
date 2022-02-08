//
//  face_detection.cpp
//  NelivaSDK
//
//  Created by Bitmain on 2020/11/5.
//  Copyright © 2020年 AnBaolei. All rights reserved.
//

#include <iostream>
#include <assert.h>
#include "face_detection.hpp"

using namespace std;
using namespace cv;

FaceDetection::FaceDetection(const std::string bmodel_path, int device_id) {
  bmodel_path_ = bmodel_path;
  device_id_ = device_id;
  load_model();

  float input_scale = 1.0;
  input_scale *= net_info_->input_scales[0];
  convert_attr_.alpha_0 = input_scale;
  convert_attr_.beta_0 = 0;
  convert_attr_.alpha_1 = input_scale;
  convert_attr_.beta_1 = 0;
  convert_attr_.alpha_2 = input_scale;
  convert_attr_.beta_2 = 0;
  bm_status_t ret = bm_image_create_batch(bm_handle_, net_h_, net_w_,
                        FORMAT_RGB_PLANAR,
                        data_type_,
                        scaled_inputs_, batch_size_);
  if (BM_SUCCESS != ret) {
    std::cerr << "ERROR: bm_image_create_batch failed" << std::endl;
    exit(-1);
  }
  shared_ptr<RetinaFacePostProcess> post_ptr(new RetinaFacePostProcess);
  post_process_ = post_ptr;
}

FaceDetection::~FaceDetection() {
}

bool FaceDetection::run(vector<Mat>& input_imgs,
                               vector<vector<stFaceRect> >& results) {
  std::vector<bm_image> input_bm_imgs;
  for (size_t i = 0; i < input_imgs.size(); i++) {
    bm_image bmimg;
    bm_image_from_mat(bm_handle_, input_imgs[i], bmimg);
    input_bm_imgs.push_back(bmimg);
  }
  std::vector<bm_image> processed_imgs;
  preprocess(input_bm_imgs, processed_imgs);
  assert(static_cast<int>(input_imgs.size()) == batch_size_);
  bmcv_image_convert_to(bm_handle_, batch_size_,
             convert_attr_, &processed_imgs[0], scaled_inputs_);
  forward();
  postprocess(results);

  for (size_t i = 0; i < input_bm_imgs.size(); i++) {
    bm_image_destroy(input_bm_imgs[i]);
    bm_image_destroy(processed_imgs[i]);
  }
  for (size_t i = 0; i < results.size(); i++) {
    for (size_t j = 0; j < results[i].size(); j++) {
      int resize_size = max(input_imgs[i].cols, input_imgs[i].rows);
      results[i][j].left = (results[i][j].left * resize_size) / net_w_;
      results[i][j].right = (results[i][j].right * resize_size) / net_w_;
      results[i][j].top = (results[i][j].top * resize_size) / net_h_;
      results[i][j].bottom = (results[i][j].bottom * resize_size) / net_h_;

      for (size_t k = 0; k < 5; k++) {
        results[i][j].points_x[k] =
            (results[i][j].points_x[k] * resize_size) / net_w_;
        results[i][j].points_y[k] =
            (results[i][j].points_y[k] * resize_size) / net_h_;
      }
    }
  }
  return true;
}

void FaceDetection::preprocess(const std::vector<bm_image>& input_imgs,
                                   std::vector<bm_image>& processed_imgs) {
  for (size_t i = 0; i < input_imgs.size(); i++) {
    bm_image resize_img;
    int resize_width = (input_imgs[i].width > input_imgs[i].height) ? net_w_ :
                                  net_w_ * input_imgs[i].width / input_imgs[i].height;
    int resize_height = (input_imgs[i].height > input_imgs[i].width) ? net_h_ :
                                  net_h_ * input_imgs[i].height / input_imgs[i].width;
    bm_image_create(bm_handle_, resize_height, resize_width, 
                          FORMAT_RGB_PLANAR, 
                          DATA_TYPE_EXT_1N_BYTE, &resize_img, NULL);
    bmcv_rect_t resize_rect = {0, 0, input_imgs[i].width, input_imgs[i].height};
    bmcv_image_vpp_convert(bm_handle_, 1, input_imgs[i], &resize_img, &resize_rect);

    bm_image processed_img;
    bm_image_create(bm_handle_, net_h_, net_w_,
             FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &processed_img, NULL);
    bmcv_copy_to_atrr_t copy_to_attr;
    copy_to_attr.start_x = 0;
    copy_to_attr.start_y = 0;
    copy_to_attr.padding_r = 0;
    copy_to_attr.padding_g = 0;
    copy_to_attr.padding_b = 0;
    copy_to_attr.if_padding = 1;
    bmcv_image_copy_to(bm_handle_, copy_to_attr, resize_img, processed_img);
    processed_imgs.push_back(processed_img);
    bm_image_destroy(resize_img);
  }
  return;
}

void FaceDetection::postprocess(vector<vector<stFaceRect> >& results) {
  for (int i = 0; i < batch_size_; i++) {
    float *preds[output_num_];
    vector<stFaceRect> det_result;
    results.push_back(det_result);

    for (int j = 0; j < output_num_; j++) {
      if (BM_FLOAT32 == net_info_->output_dtypes[j]) {
        preds[j] = reinterpret_cast<float*>(outputs_[j]) + output_sizes_[j] * i;
      } else {
        signed char* int8_ptr = reinterpret_cast<signed char*>(outputs_[j])
                                                       + output_sizes_[j] * i;
        preds[j] = new float[output_sizes_[j]];
        for (int k = 0; k < output_sizes_[j]; k++) {
          preds[j][k] = int8_ptr[k] * net_info_->output_scales[j];
        }
      }
    }
    post_process_->run(*net_info_, preds,
                  results[i], max_face_count_, score_threshold_);
    for (int j = 0; j < output_num_; j++) {
      if (BM_FLOAT32 != net_info_->output_dtypes[j]) {
        delete []preds[j];
      }
    }
  }
  return;
}

 void FaceDetection::set_max_face_count(int max_face_count) {
   max_face_count_ = max_face_count;
 }
 
 void FaceDetection::set_score_threshold(float score_threshold) {
   score_threshold_ = score_threshold;
 }
