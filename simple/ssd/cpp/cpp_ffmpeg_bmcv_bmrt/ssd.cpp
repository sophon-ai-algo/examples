#include "ssd.hpp"
#include "utils.hpp"
#include <iostream>
#include "sys/stat.h"
#include "boost/filesystem.hpp"

#ifdef __linux__
#include <unistd.h>
#endif

using namespace std;
namespace fs = boost::filesystem;

#define IS_END_OF_DATA(entry) (entry[1] == 0 && entry[2] == 0 &&\
                               entry[3] == 0 && entry[4] == 0 &&\
                               entry[5] == 0 && entry[6] == 0)

#define INPUT_WIDTH 300
#define INPUT_HEIGHT 300
#define BUFFER_SIZE (1024 * 500)

string net_name_;
//const char *model_name = "VGG_VOC0712_SSD_300x300_deploy";

SSD::SSD(bm_handle_t bm_handle, int dev_id, const string bmodel) {

  bool ret;

  // get device handle
  bm_handle_ = bm_handle;
  dev_id_ = dev_id;

  // init bmruntime contxt
  p_bmrt_ = bmrt_create(bm_handle_);
  if (NULL == p_bmrt_) {
    cout << "ERROR: get handle failed!" << endl;
    exit(1);
  }

  // load bmodel from file
  ret = bmrt_load_bmodel(p_bmrt_, bmodel.c_str());
  if (!ret) {
    cout << "ERROR: Load bmodel[" << bmodel << "] failed" << endl;
    exit(1);
  }

  const char **net_names;
  bmrt_get_network_names(p_bmrt_, &net_names);
  net_name_ = net_names[0];
  free(net_names);

  // get model info by model name
  net_info_ = bmrt_get_network_info(p_bmrt_, net_name_.c_str());
  if (NULL == net_info_) {
    cout << "ERROR: get net-info failed!" << endl;
    exit(1);
  }

  // get data type
  if (NULL == net_info_->input_dtypes) {
    cout << "ERROR: get net input type failed!" << endl;
    exit(1);
  }
  if (BM_FLOAT32 == net_info_->input_dtypes[0]) {
    threshold_ = 0.6;
    is_int8_ = false;
  } else {
    threshold_ = 0.52;
    is_int8_ = true;
  }

  // allocate output buffer
  output_ = new float[BUFFER_SIZE];

  // init bm images for storing results of combined operation of resize & crop & split
  bm_status_t bm_ret = bm_image_create_batch(bm_handle_,
                              INPUT_HEIGHT,
                              INPUT_WIDTH,
                              FORMAT_BGR_PLANAR,
                              DATA_TYPE_EXT_1N_BYTE,
                              resize_bmcv_,
                              MAX_BATCH);
  if (BM_SUCCESS != bm_ret) {
    cout << "ERROR: bm_image_create_batch failed" << endl;
    exit(1);
  }

  // bm images for storing inference inputs
  bm_image_data_format_ext data_type;
  if (is_int8_) { // INT8
    data_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
  } else { // FP32
    data_type = DATA_TYPE_EXT_FLOAT32;
  }

  bm_ret = bm_image_create_batch (bm_handle_,
                               INPUT_HEIGHT,
                               INPUT_WIDTH,
                               FORMAT_BGR_PLANAR,
                               data_type,
                               linear_trans_bmcv_,
                               MAX_BATCH);

  if (BM_SUCCESS != bm_ret) {
    cout << "ERROR: bm_image_create_batch failed" << endl;
    exit(1);
  }


  // init linear transform parameter, X*a + b, int8 model need to consider scales
  float input_scale = net_info_->input_scales[0];
  linear_trans_param_.alpha_0 = input_scale;
  linear_trans_param_.beta_0 = -123.0 * input_scale;
  linear_trans_param_.alpha_1 = input_scale;
  linear_trans_param_.beta_1 = -117.0 * input_scale;
  linear_trans_param_.alpha_2 = input_scale;
  linear_trans_param_.beta_2 = -104.0 * input_scale;
}

SSD::~SSD() {

  // deinit bm images
  bm_image_destroy_batch (resize_bmcv_, MAX_BATCH);
  bm_image_destroy_batch (linear_trans_bmcv_, MAX_BATCH);

  // free output buffer
  delete []output_;

  // deinit contxt handle
  bmrt_destroy(p_bmrt_);
}

void SSD::enableProfile(TimeStamp *ts) {
  ts_ = ts;
}

void SSD::preForward(vector<bm_image> &input) {
  LOG_TS(ts_, "ssd pre-process")
  preprocess_bmcv (input);
  LOG_TS(ts_, "ssd pre-process")
}

void SSD::forward() {
  LOG_TS(ts_, "ssd inference")
  memset(output_, 0, sizeof(float) * BUFFER_SIZE);
  bool res = bm_inference (p_bmrt_, linear_trans_bmcv_, (void*)output_, input_shape_, net_name_.c_str());
  LOG_TS(ts_, "ssd inference")

  if (!res) {
    cout << "ERROR : inference failed!!"<< endl;
    exit(1);
  }
}

void SSD::postForward (vector<bm_image> &input, vector<vector<ObjRect>> &detections) {
  int stage_num = net_info_->stage_num;
  bm_shape_t output_shape;
  for (int i = 0; i < stage_num; i++) {
    if(net_info_->stages[i].input_shapes[0].dims[0] == (int)input.size()) {
      output_shape = net_info_->stages[i].output_shapes[0];
      break;
    }

    if ( i == (stage_num - 1)) {
      cout << "ERROR: output not match stages" << endl;
      return;
    }
  }

  LOG_TS(ts_, "ssd post-process")
  int output_count = bmrt_shape_count(&output_shape);
  unsigned int image_id=0;
  int img_size = input.size();
  //int count_per_img = output_count/img_size;
  detections.clear();
  std::vector<ObjRect> tmp;

  //init detections
  for (int i = 0; i < img_size; i++) {
    detections.push_back(tmp);
  }

  for (int i = 0; i < output_count; i+=7) {

    ObjRect detection;
    float *proposal = &output_[i];

    if (IS_END_OF_DATA(proposal))
      break;

    /*
     * Detection_Output format:
     *   [image_id, label, score, xmin, ymin, xmax, ymax]
     */
    //std::cout << "** score: " << proposal[2] << "/" << threshold_ << std::endl;
    if (proposal[2] < threshold_)
      continue;

    image_id = proposal[0];
    if (image_id >= input.size()) {
      std::cout << "!! err  image_id:" << image_id
		<< " input size:" << input.size() << std::endl;
      continue;
    }

    detection.class_id = proposal[1];
    detection.score = proposal[2];
    detection.x1 = proposal[3] * input[image_id].width;
    detection.y1 = proposal[4] * input[image_id].height;
    detection.x2 = proposal[5] * input[image_id].width;
    detection.y2 = proposal[6] * input[image_id].height;
//#define DEBUG_SSD
#ifdef DEBUG_SSD
    std::cout << "image id: " << std::setw(2) << image_id << std::setw(10)
            << "class id: " << std::setw(2) << detection.class_id
            << std::fixed << std::setprecision(5)
            << " upper-left: (" << std::setw(10) << detection.x1 << ", "
            << std::setw(10) << detection.y1 << ") "
            << " object-size: (" << std::setw(10) << detection.x2 - detection.x1 + 1 << ", "
            << std::setw(10)  << detection.y2 - detection.y1 + 1 << ")"
            << std::endl;
#endif

    detections[image_id].push_back(detection);
  }
  LOG_TS(ts_, "ssd post-process")
}

void SSD::preprocess_bmcv (vector<bm_image> &input) {
  if (input.empty()) {
    cout << "mul-batch bmcv input empty!!!" << endl;
    return ;
  }

  if (!((1 == input.size()) || (4 == input.size()))) {
    cout << "mul-batch bmcv input error!!!" << endl;
    return ;
  }

  // set input shape according to input bm images
  input_shape_ = {4, {(int)input.size(), 3, INPUT_HEIGHT, INPUT_WIDTH}};

  // do not crop
  crop_rect_ = {0, 0, input[0].width, input[0].height};

  // resize && split by bmcv
  for (size_t i = 0; i < input.size(); i++) {
    LOG_TS(ts_, "ssd pre-process-vpp")
    bmcv_image_vpp_convert (bm_handle_, 1, input[i], &resize_bmcv_[i], &crop_rect_);
    LOG_TS(ts_, "ssd pre-process-vpp")
  }

  // do linear transform
  LOG_TS(ts_, "ssd pre-process-linear_tranform")
  bmcv_image_convert_to (bm_handle_, input.size(), linear_trans_param_, resize_bmcv_, linear_trans_bmcv_);
  LOG_TS(ts_, "ssd pre-process-linear_tranform")
}

void SSD::dumpImg(bm_image &input, vector<ObjRect> &detect) {
  static int frame_count = 0;
  char file_name[250];
  FILE *file_handle;
  bmcv_rect_t rect;
  size_t out_size;
  void *p_jpeg_data = NULL;
  printf("[%d]frame %d - need to draw %d bbox\r\n", dev_id_, (int)frame_count, (int)detect.size());
  for (int j = 0; j < (int)detect.size(); j++) {
    rect.start_x = detect[j].x1;
    rect.start_y = detect[j].y1;
    rect.crop_w = detect[j].x2 - detect[j].x1 + 1;
    rect.crop_h = detect[j].y2 - detect[j].y1 + 1;

#ifdef DEBUG_SSD
    std::cout << std::fixed << std::setprecision(5)
            << " upper-left: (" << std::setw(10) << rect.start_x << ", "
            << std::setw(10) << rect.start_y << ") "
            << " object-size: (" << std::setw(10) << rect.crop_w << ", "
            << std::setw(10)  << rect.crop_h << ")"
            << std::endl;
#endif

    //draw rectangles
    bmcv_image_draw_rectangle(bm_handle_, input, 1, &rect, 3, 0, 0, 255);
  }

  // check result directory
  if (!fs::exists("./results")) {
	  fs::create_directory("results");
  }

  //encoded to jpg
  int ret = bmcv_image_jpeg_enc(bm_handle_, 1, &input, &p_jpeg_data, &out_size);
  if (ret == 0) {
    printf("encoded image size is %d\r\n", (int)out_size);
    if (getPrecision()) {
      sprintf(file_name, "results/int8_dev_%d_frame_%d.jpg", dev_id_, frame_count++);
    } else {
      sprintf(file_name, "results/fp32_dev_%d_frame_%d.jpg", dev_id_, frame_count++);
    }
    //write to file
    file_handle = fopen(file_name, "wb+");
    if (file_handle != NULL) {
      fwrite(p_jpeg_data, out_size, 1, file_handle);
      fclose(file_handle);
    }
  }
  if (p_jpeg_data) {
    free(p_jpeg_data);
  }

}

bool SSD::getPrecision() {
  return is_int8_;
}
