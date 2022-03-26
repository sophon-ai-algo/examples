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

#ifdef USE_BMCV
BmcvPreProcessor::BmcvPreProcessor(sail::Bmcv& bmcv, int result_width, int result_height, float scale) : bmcv_(bmcv),
  ab_{1, 0, 1, 0, 1, 0},resize_w(result_width),resize_h(result_height) {
  for (int i = 0; i < 6; i ++) {
    ab_[i] *= scale;
  }
}

void BmcvPreProcessor::process(sail::BMImage& input, sail::BMImage& output) {
  // resize: bgr-packed -> bgr-planer
  sail::BMImage tmp;
  bmcv_.vpp_resize(input, tmp, resize_w, resize_h);
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
  std::vector<cv::Mat> mats{cv::Mat(resize_h, resize_w, CV_8UC3)};
  std::vector<cv::Size> sizes{cv::Size(resize_w, resize_h)};
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

float overlap_FM(float x1, float w1, float x2, float w2)
{
	float l1 = x1;
	float l2 = x2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1;
	float r2 = x2 + w2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

float box_intersection_FM(ObjRect a, ObjRect b)
{
	float w = overlap_FM(a.left, a.width, b.left, b.width);
	float h = overlap_FM(a.top, a.height, b.top, b.height);
	if (w < 0 || h < 0) return 0;
	float area = w*h;
	return area;
}

float box_union_FM(ObjRect a, ObjRect b)
{
	float i = box_intersection_FM(a, b);
	float u = a.width*a.height + b.width*b.height - i;
	return u;
}

float box_iou_FM(ObjRect a, ObjRect b)
{
	return box_intersection_FM(a, b) / box_union_FM(a, b);
}

static bool sort_ObjRect(ObjRect a, ObjRect b)
{
    return a.score > b.score;
}

static void nms_sorted_bboxes(const std::vector<ObjRect>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    const int n = objects.size();

    for (int i = 0; i < n; i++)    {
        const ObjRect& a = objects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const ObjRect& b = objects[picked[j]];

            float iou = box_iou_FM(a, b);
            if (iou > nms_threshold)
                keep = 0;
        }
        if (keep)
            picked.push_back(i);
    }
}

YoloX_PostForward::YoloX_PostForward(int net_w, int net_h, std::vector<int> strides):network_width(net_w),network_height(net_h)
{
  outlen_diml = 0;
  for (int i=0;i<strides.size();++i)  {
    int layer_w = net_w/strides[i];
    int layer_h = net_h/strides[i];
    outlen_diml += layer_h*layer_w;
  }
  grids_x_ = new int[outlen_diml];
  grids_y_ = new int[outlen_diml];
  expanded_strides_ = new int[outlen_diml];

  int channel_len = 0;
  for (int i=0;i<strides.size();++i)  {
    int layer_w = net_w/strides[i];
    int layer_h = net_h/strides[i];
    for (int m = 0; m < layer_h; ++m)   {
      for (int n = 0; n < layer_w; ++n)    {
          grids_x_[channel_len+m*layer_h+n] = n;
          grids_y_[channel_len+m*layer_h+n] = m;
          expanded_strides_[channel_len+m*layer_h+n] = strides[i];
      }
    }
    channel_len += layer_w * layer_h;
  }
}

YoloX_PostForward::~YoloX_PostForward()
{
  delete grids_x_;
  grids_x_ = NULL;
  delete grids_y_;
  grids_y_ = NULL;
  delete expanded_strides_;
  expanded_strides_ = NULL;
}

void YoloX_PostForward::process(float* data_ptr,std::vector<int> output_shape, std::vector<std::pair<int,int>> ost_size, 
  float threshold, float nms_threshold, std::vector<std::vector<ObjRect>> &detections)
{
  int size_one_batch = output_shape[1]*output_shape[2];
  int channels_resu_ = output_shape[2];
  int classes_ = channels_resu_ - 5;

  for (int batch_idx=0; batch_idx<ost_size.size();batch_idx++){
    int batch_start_ptr = size_one_batch * batch_idx;
    std::vector<ObjRect> dect_temp;
    dect_temp.clear();
    float scale_x = (float)ost_size[batch_idx].first/network_width;
    float scale_y = (float)ost_size[batch_idx].second/network_height;
    for (size_t i = 0; i < outlen_diml; i++)    {
        int ptr_start=i*channels_resu_;
        float box_objectness = data_ptr[batch_start_ptr + ptr_start+4];
        if(data_ptr[batch_start_ptr + ptr_start+4] >= threshold){
            float center_x = (data_ptr[batch_start_ptr +ptr_start+0] + grids_x_[i]) * expanded_strides_[i];
            float center_y = (data_ptr[batch_start_ptr +ptr_start+1] + grids_y_[i]) * expanded_strides_[i];
            float w_temp = exp(data_ptr[batch_start_ptr +ptr_start+2])*expanded_strides_[i];
            float h_temp = exp(data_ptr[batch_start_ptr +ptr_start+3])*expanded_strides_[i];
            float score = data_ptr[batch_start_ptr +ptr_start+4];
            center_x *= scale_x;
            center_y *= scale_y;
            w_temp *= scale_x;
            h_temp *= scale_y;
            float left = center_x - w_temp/2;
            float top = center_y - h_temp/2;
            float right = center_x + w_temp/2;
            float bottom = center_y + h_temp/2;

            // printf("[%.2f,%.2f,%.2f,%.2f]::::%f,%f\n",center_x,center_y,w_temp,h_temp,scale_x,scale_y);

            for (int class_idx = 0; class_idx < classes_; class_idx++)       {
                float box_cls_score = data_ptr[batch_start_ptr +ptr_start + 5 + class_idx];
                float box_prob = box_objectness * box_cls_score;
                if (box_prob > threshold)         {
                    ObjRect obj_temp;
                    obj_temp.width = w_temp;
                    obj_temp.height = h_temp;
                    obj_temp.left = left;
                    obj_temp.top = top;
                    obj_temp.right = right;
                    obj_temp.bottom = bottom;
                    obj_temp.score = box_prob;
                    obj_temp.class_id = class_idx;
                    dect_temp.push_back(obj_temp);
                }
            }
        }
    }

    std::sort(dect_temp.begin(),dect_temp.end(),sort_ObjRect);

    std::vector<ObjRect> dect_temp_batch;
    std::vector<int> picked;
    dect_temp_batch.clear();
    nms_sorted_bboxes(dect_temp, picked, nms_threshold);

    for (size_t i = 0; i < picked.size(); i++)    {
        dect_temp_batch.push_back(dect_temp[picked[i]]);
    }
    
    detections.push_back(dect_temp_batch);
}
}
