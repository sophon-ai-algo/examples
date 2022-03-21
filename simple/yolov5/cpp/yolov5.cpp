/*==========================================================================
 * Copyright 2016-2022 by Sophgo Technologies Inc. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
============================================================================*/

#include "yolov5.hpp"
#include <fstream>
#include <vector>
/**
  YOLOV5 input && output
# input: x.1, [1, 3, 640, 640], float32, scale: 1
# output: 5, [1, box_num, 85], float32, scale: 1
*/

#define USE_ASPECT_RATIO 1
#define DUMP_FILE 1

YoloV5::YoloV5(std::shared_ptr<BMNNContext> context):m_bmContext(context) {
  std::cout << "YoloV5 ctor .." << std::endl;
}

YoloV5::~YoloV5() {
  std::cout << "YoloV5 dtor ..." << std::endl;
  for(int i=0; i<max_batch; i++){
    bm_image_destroy(m_converto_imgs[i]);
    bm_image_destroy(m_resized_imgs[i]);
  }
}


int YoloV5::Init(float confThresh, float objThresh, float nmsThresh, const std::string& coco_names_file)
{
  m_confThreshold= confThresh;
  m_objThreshold = objThresh;
  m_nmsThreshold = nmsThresh;
  std::ifstream ifs(coco_names_file);
  if (ifs.is_open()) {
    std::string line;
    while(std::getline(ifs, line)) {
      m_class_names.push_back(line);
    }
  }

  //1. get network
  m_bmNetwork = m_bmContext->network(0);

  //2. initialize bmimages
  // input shape is NCHW
  max_batch = m_bmNetwork->maxBatch();
  m_resized_imgs.resize(max_batch);
  m_converto_imgs.resize(max_batch);

  auto tensor = m_bmNetwork->inputTensor(0);
  m_net_h = tensor->get_shape()->dims[2];
  m_net_w = tensor->get_shape()->dims[3];

  // some API only accept bm_image whose stride is aligned to 64
  int aligned_net_w = FFALIGN(m_net_w, 64);
  int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
  for(int i=0; i<max_batch; i++){
    auto ret= bm_image_create(m_bmContext->handle(), m_net_h, m_net_w,
        FORMAT_RGB_PLANAR,
        DATA_TYPE_EXT_1N_BYTE,
        &m_resized_imgs[i], strides);
    assert(BM_SUCCESS == ret);
  }
  bm_image_alloc_contiguous_mem (max_batch, m_resized_imgs.data());

  bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
  if (tensor->get_dtype() == BM_INT8) {
    img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
  }

  auto ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w,
      FORMAT_RGB_PLANAR,
      img_dtype,
      m_converto_imgs.data(), max_batch);
  assert(BM_SUCCESS == ret);

  return 0;
}

float YoloV5::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *pIsAligWidth)
{
  float ratio;
  ratio = (float) dst_w / src_w;
  int dst_h1 = src_h * ratio;
  if (dst_h1 > dst_h) {
    *pIsAligWidth = false;
    ratio = (float) src_w / src_h;
  } else {
    *pIsAligWidth = true;
    ratio = (float)src_h/src_w;
  }

  return ratio;
}

int YoloV5::pre_process(const std::vector<cv::Mat>& images, int image_n)
{

  // Check input parameters
  if( image_n > max_batch) {
    std::cout << "input image size > MAX_BATCH(" << max_batch<< ")." << std::endl;
    return -1;
  }

  //1. resize image
  std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
  if (image_n > input_tensor->get_shape()->dims[0]) {
    std::cout << "input image size > input_shape.batch(" << input_tensor->get_shape()->dims[0] << ")." << std::endl;
    return -1;
  }

  int ret = 0;
  for(int i = 0; i < image_n; ++i) {
    bm_image image1;
    bm_image image_aligned;
    // src_img
    cv::bmcv::toBMI((cv::Mat&)images[i], &image1);
    bool need_copy = image1.width & (64-1);

    if(need_copy){
      int stride1[3], stride2[3];
      bm_image_get_stride(image1, stride1);
      stride2[0] = FFALIGN(stride1[0], 64);
      stride2[1] = FFALIGN(stride1[1], 64);
      stride2[2] = FFALIGN(stride1[2], 64);
      bm_image_create(m_bmContext->handle(), image1.height, image1.width,
          image1.image_format, image1.data_type, &image_aligned, stride2);

      bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN);
      bmcv_copy_to_atrr_t copyToAttr;
      memset(&copyToAttr, 0, sizeof(copyToAttr));
      copyToAttr.start_x = 0;
      copyToAttr.start_y = 0;
      copyToAttr.if_padding = 1;

      bmcv_image_copy_to(m_bmContext->handle(), copyToAttr, image1, image_aligned);
    } else {
      image_aligned = image1;
    }

#if USE_ASPECT_RATIO
    bool isAlignWidth = false;
    float ratio = get_aspect_scaled_ratio(images[i].cols, images[i].rows, m_net_w, m_net_h, &isAlignWidth);
    bmcv_padding_atrr_t padding_attr;
    memset(&padding_attr, 0, sizeof(padding_attr));
    padding_attr.dst_crop_sty = 0;
    padding_attr.dst_crop_stx = 0;
    padding_attr.padding_b = 114;
    padding_attr.padding_g = 114;
    padding_attr.padding_r = 114;
    padding_attr.if_memset = 1;
    if (isAlignWidth) {
      padding_attr.dst_crop_h = m_net_w*ratio;
      padding_attr.dst_crop_w = m_net_w;
    }else{
      padding_attr.dst_crop_h = m_net_h;
      padding_attr.dst_crop_w = m_net_w*ratio;
    }

    bmcv_rect_t crop_rect{0, 0, image1.width, image1.height};
    auto ret = bmcv_image_vpp_convert_padding(m_bmContext->handle(), 1, image_aligned, &m_resized_imgs[i],
        &padding_attr, &crop_rect);
#else
    auto ret = bmcv_image_vpp_convert(m_bmContext->handle(), 1, images[i], &m_resized_imgs[i]);
#endif
    assert(BM_SUCCESS == ret);

#if DUMP_FILE
    cv::Mat resized_img;
    cv::bmcv::toMAT(&m_resized_imgs[i], resized_img);
    std::string fname = cv::format("resized_img_%d.jpg", i);
    cv::imwrite(fname, resized_img);
#endif

    bm_image_destroy(image1);
    if(need_copy) bm_image_destroy(image_aligned);
  }

  //2. converto
  float input_scale = input_tensor->get_scale();
  input_scale = input_scale* (float)1.0/255;
  bmcv_convert_to_attr converto_attr;
  converto_attr.alpha_0 = input_scale;
  converto_attr.beta_0 = 0;
  converto_attr.alpha_1 = input_scale;
  converto_attr.beta_1 = 0;
  converto_attr.alpha_2 = input_scale;
  converto_attr.beta_2 = 0;

  ret = bmcv_image_convert_to(m_bmContext->handle(), image_n, converto_attr, m_resized_imgs.data(), m_converto_imgs.data());
  CV_Assert(ret == 0);

  //3. attach to tensor
  if(image_n != max_batch) image_n = m_bmNetwork->get_nearest_batch(image_n); 
  bm_device_mem_t input_dev_mem;
  bm_image_get_contiguous_device_mem(image_n, m_converto_imgs.data(), &input_dev_mem);
  input_tensor->set_device_mem(&input_dev_mem);
  input_tensor->set_shape_by_dim(0, image_n);  // set real batch number
  return 0;
}

int YoloV5::Detect(const std::vector<cv::Mat>& input_images, std::vector<YoloV5BoxVec>& boxes)
{
  int ret = 0;
  int total_batch = input_images.size();
  int detected_batch = 0;

  std::vector<cv::Mat> images(max_batch);
  while(detected_batch<total_batch){
    int input_batch = total_batch-detected_batch;
    if(input_batch>max_batch) input_batch = max_batch;
    for(int i=0; i<input_batch; i++){
      images[i] = input_images[detected_batch+i];
    }
    detected_batch += input_batch;

    //3. preprocess
    LOG_TS(m_ts, "YoloV5 preprocess");
    ret = pre_process(images, input_batch);
    CV_Assert(ret == 0);
    LOG_TS(m_ts, "YoloV5 preprocess");


    //4. forward
    LOG_TS(m_ts, "YoloV5 inference");
    ret = m_bmNetwork->forward();
    CV_Assert(ret == 0);
    LOG_TS(m_ts, "YoloV5 inference");

    //5. post process
    LOG_TS(m_ts, "YoloV5 postprocess");

    ret = post_process(images, boxes, input_batch);
    CV_Assert(ret == 0);
    LOG_TS(m_ts, "YoloV5 postprocess");
  }
  return ret;
}

int YoloV5::argmax(float* data, int num) {
  float max_value = 0.0;
  int max_index = 0;
  for(int i = 0; i < num; ++i) {
    float value = data[i];
    if (value > max_value) {
      max_value = value;
      max_index = i;
    }
  }

  return max_index;
}


static float sigmoid(float x)
{
  return 1.0 / (1 + expf(-x));
}

int YoloV5::post_process(const std::vector<cv::Mat> &images, std::vector<YoloV5BoxVec>& detected_boxes, int input_batch)
{
  YoloV5BoxVec yolobox_vec;
  std::vector<cv::Rect> bbox_vec;
	int output_num = m_bmNetwork->outputTensorNum();
  std::vector<std::shared_ptr<BMNNTensor>> outputTensors(output_num);
  for(int i=0; i<output_num; i++){
      outputTensors[i] = m_bmNetwork->outputTensor(i);
  }
  

  for(int batch_idx = 0; batch_idx < input_batch; ++ batch_idx)
  {
    yolobox_vec.clear();
    auto& frame = images[batch_idx];
    int frame_width = frame.cols;
    int frame_height = frame.rows;

#if USE_ASPECT_RATIO
    bool isAlignWidth = false;
    get_aspect_scaled_ratio(frame.cols, frame.rows, m_net_w, m_net_h, &isAlignWidth);

    if (isAlignWidth) {
      frame_height = frame_width*(float)m_net_h/m_net_w;
    }else{
      frame_width = frame_height*(float)m_net_w/m_net_h;
    }
#endif

    assert(output_num>0);
    int min_dim = m_bmNetwork->outputTensor(0)->get_shape()->num_dims;
    int min_idx = 0;
    int box_num = 0;
    for(int i=0; i<output_num; i++){
      auto output_shape = m_bmNetwork->outputTensor(i)->get_shape();
      auto output_dims = output_shape->num_dims;
      assert(output_dims == 3 || output_dims == 5);
      if(output_dims == 5){
        box_num += output_shape->dims[1] * output_shape->dims[2] * output_shape->dims[3];
      }

      if(min_dim>output_dims){
        min_idx = i;
        min_dim = output_dims;
      }
    }

    auto out_tensor = outputTensors[min_idx];
    int nout = out_tensor->get_shape()->dims[min_dim-1];
    m_class_num = nout - 5;

    float* output_data = nullptr;
    std::vector<float> decoded_data;

    if(min_dim ==3 && output_num !=1){
      std::cout<<"--> WARNING: the current bmodel has redundant outputs"<<std::endl;
      std::cout<<"             you can remove the redundant outputs to improve performance"<< std::endl;
      std::cout<<std::endl;
    }

    if(min_dim == 5){
      LOG_TS(m_ts, "1: get and decode");
      std::cout<<"--> Note: Decoding Boxes"<<std::endl;
      std::cout<<"          you can put the process into model during trace"<<std::endl;
      std::cout<<"          which can reduce post process time, but forward time increases 1ms"<<std::endl;
      std::cout<<std::endl;
      const std::vector<std::vector<std::vector<int>>> anchors{
        {{10, 13}, {16, 30}, {33, 23}},
          {{30, 61}, {62, 45}, {59, 119}},
          {{116, 90}, {156, 198}, {373, 326}}};
      const int anchor_num = anchors[0].size();
      assert(output_num == (int)anchors.size());
      assert(box_num>0);
      if((int)decoded_data.size() != box_num*nout){
        decoded_data.resize(box_num*nout);
      }
      float *dst = decoded_data.data();
      for(int tidx = 0; tidx < output_num; ++tidx) {
        auto output_tensor = outputTensors[tidx];
        int feat_c = output_tensor->get_shape()->dims[1];
        int feat_h = output_tensor->get_shape()->dims[2];
        int feat_w = output_tensor->get_shape()->dims[3];
        int area = feat_h * feat_w;
        assert(feat_c == anchor_num);
        int feature_size = feat_h*feat_w*nout;
        float *tensor_data = (float*)output_tensor->get_cpu_data() + batch_idx*feat_c*area*nout;
        for (int anchor_idx = 0; anchor_idx < anchor_num; anchor_idx++)
        {
          float *ptr = tensor_data + anchor_idx*feature_size;
          for (int i = 0; i < area; i++) {
            dst[0] = (sigmoid(ptr[0]) * 2 - 0.5 + i % feat_w) / feat_w * m_net_w;
            dst[1] = (sigmoid(ptr[1]) * 2 - 0.5 + i / feat_h) / feat_h * m_net_h;
            dst[2] = pow((sigmoid(ptr[2]) * 2), 2) * anchors[tidx][anchor_idx][0];
            dst[3] = pow((sigmoid(ptr[3]) * 2), 2) * anchors[tidx][anchor_idx][1];
            dst[4] = sigmoid(ptr[4]);
            float score = dst[4];
            if (score > m_objThreshold) {
              for(int d=5; d<nout; d++){
                dst[d] = sigmoid(ptr[d]);
              }
            }
            dst += nout;
            ptr += nout;
          }
        }
      }
      output_data = decoded_data.data();
      LOG_TS(m_ts, "1: get and decode");
    } else {
      LOG_TS(m_ts, "post 1: get output");
      assert(box_num == 0 || box_num == out_tensor->get_shape()->dims[1]);
      box_num = out_tensor->get_shape()->dims[1];
      output_data = (float*)out_tensor->get_cpu_data() + batch_idx*box_num*nout;
      LOG_TS(m_ts, "post 1: get output");
    }


    LOG_TS(m_ts, "post 2: filter boxes");
    for (int i = 0; i < box_num; i++) {
      float* ptr = output_data+i*nout;
      float score = ptr[4];
      if (score > m_objThreshold)
      {
        int class_id = argmax(&ptr[5], m_class_num);
        float confidence = ptr[class_id + 5];
        if (confidence >= m_confThreshold)
        {
          float centerX = (ptr[0]+1)/m_net_w*frame_width-1;
          float centerY = (ptr[1]+1)/m_net_h*frame_height-1;
          float width = (ptr[2]+0.5) * frame_width / m_net_w;
          float height = (ptr[3]+0.5) * frame_height / m_net_h;

          YoloV5Box box;
          box.x = int(centerX - width / 2);
          box.y = int(centerY - height / 2);
          box.width = width;
          box.height = height;
          box.class_id = class_id;
          box.score = confidence * score;
          yolobox_vec.push_back(box);
        }
      }
    }
    LOG_TS(m_ts, "post 2: filter boxes");

    printf("\n --> valid boxes number = %d\n", (int)yolobox_vec.size());

    LOG_TS(m_ts, "post 3: nms");
    NMS(yolobox_vec, m_nmsThreshold);
    LOG_TS(m_ts, "post 3: nms");

    detected_boxes.push_back(yolobox_vec);
  }

  return 0;
}

void YoloV5::NMS(YoloV5BoxVec &dets, float nmsConfidence)
{
  int length = dets.size();
  int index = length - 1;

  std::sort(dets.begin(), dets.end(), [](const YoloV5Box& a, const YoloV5Box& b) {
      return a.score < b.score;
      });

  std::vector<float> areas(length);
  for (int i=0; i<length; i++)
  {
    areas[i] = dets[i].width * dets[i].height;
  }

  while (index  > 0)
  {
    int i = 0;
    while (i < index)
    {
      float left    = std::max(dets[index].x,   dets[i].x);
      float top     = std::max(dets[index].y,    dets[i].y);
      float right   = std::min(dets[index].x + dets[index].width,  dets[i].x + dets[i].width);
      float bottom  = std::min(dets[index].y + dets[index].height, dets[i].y + dets[i].height);
      float overlap = std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
      if (overlap / (areas[index] + areas[i] - overlap) > nmsConfidence)
      {
        areas.erase(areas.begin() + i);
        dets.erase(dets.begin() + i);
        index --;
      }
      else
      {
        i++;
      }
    }
    index--;
  }
}


void YoloV5::drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)   // Draw the predicted bounding box
{
  //Draw a rectangle displaying the bounding box
  cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 3);

  //Get the label for the class name and its confidence
  std::string label = cv::format("%.2f", conf);
  if ((int)m_class_names.size() >= m_class_num) {
    label = this->m_class_names[classId] + ":" + label;
  }else{
    label = std::to_string(classId) + ":" + label;
  }

  //Display the label at the top of the bounding box
  int baseLine;
  cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
  top = std::max(top, labelSize.height);
  //rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
  cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
}


void YoloV5::enableProfile(TimeStamp *ts)
{
  m_ts = ts;
}
