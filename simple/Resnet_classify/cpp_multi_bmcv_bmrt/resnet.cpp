#include <fstream>
#include "resnet.hpp"

using namespace std;

#define BUFFER_SIZE (1024 * 500)

const char *model_name = "resnet-50";

RESNET::RESNET(bm_handle_t bm_handle, const string bmodel) {

  bool ret;

  // get device handle
  bm_handle_ = bm_handle;

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

  // get model info by model name
  net_info_ = bmrt_get_network_info(p_bmrt_, model_name);
  if (NULL == net_info_) {
    cout << "ERROR: get net-info failed!" << endl;
    exit(1);
  }

  // get data type
  if (NULL == net_info_->input_dtypes) {
    cout << "ERROR: get net input type failed!" << endl;
    exit(1);
  }

  if (BM_FLOAT32 == net_info_->input_dtypes[0])
    is_int8_ = false;
  else
    is_int8_ = true;

  // allocate output buffer
  output_ = new float[BUFFER_SIZE];

  // bm images for storing inference inputs
  bm_image_data_format_ext data_type;
  if (is_int8_) { // INT8
    data_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
  } else { // FP32
    data_type = DATA_TYPE_EXT_FLOAT32;
  }
  bm_status_t bm_ret = bm_image_create_batch (bm_handle_,
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

  // initialize linear transform parameter
  // - mean value
  // - scale value (mainly for INT8 calibration)
  float input_scale = net_info_->input_scales[0];
  linear_trans_param_.alpha_0 = input_scale;
  linear_trans_param_.beta_0 = -103.94 * input_scale;
  linear_trans_param_.alpha_1 = input_scale;
  linear_trans_param_.beta_1 = -116.78 * input_scale;
  linear_trans_param_.alpha_2 = input_scale;
  linear_trans_param_.beta_2 = -123.68 * input_scale;
}

RESNET::~RESNET() {

  // deinit bm images
  bm_image_destroy_batch (linear_trans_bmcv_, MAX_BATCH);

  // free output buffer
  delete []output_;

  // deinit contxt handle
  bmrt_destroy(p_bmrt_);
}

void RESNET::enableProfile(TimeStamp *ts) {
  ts_ = ts;
}

void RESNET::preForward(vector<bm_image> &input) {

  preprocess_bmcv (input);
}

void RESNET::forward() {

  memset(output_, 0, sizeof(float) * BUFFER_SIZE);
  bool res = bm_inference (p_bmrt_, linear_trans_bmcv_, (void*)output_, input_shape_, model_name);

  if (!res) {
    cout << "ERROR : inference failed!!"<< endl;
    exit(1);
  }
}

static bool comp(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

void RESNET::postForward (vector<bm_image> &input, vector<vector<ObjRect>> &detections) {
  int stage_num = net_info_->stage_num;
  bm_shape_t output_shape;
  for (int i = 0; i < stage_num; i++) {
    if (net_info_->stages[i].input_shapes[0].dims[0] == (int)input.size()) {
      output_shape = net_info_->stages[i].output_shapes[0];
      break;
    }

    if ( i == (stage_num - 1)) {
      cout << "ERROR: output not match stages" << endl;
      return;
    }
  }

  LOG_TS(ts_, "argmax")
  int output_count = bmrt_shape_count(&output_shape);
  int img_size = input.size();
  int count_per_img = output_count/img_size;
  detections.clear();

  //int N = 5;
  int N = std::min(5, count_per_img);
  float *image_output = output_;
  vector<std::pair<float , int>> pairs;
  for (int i = 0; i < img_size; i++) {
    vector<ObjRect> results;
    pairs.clear();
    for (int j = 0; j < count_per_img; ++j)
      pairs.push_back(make_pair(image_output[j], j));

    partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), comp);
    for (int j = 0; j < N; ++j) {
      ObjRect r;
      r.class_id = pairs[j].second;
      r.score = pairs[j].first;
#ifdef DEBUG_RESULT
      cout << "image-" << i << ": class[" << j << "] = " << r.class_id
           << ", score = " << r.score << endl;
#endif
      results.push_back(r);
    }
    detections.push_back(results);
    image_output += count_per_img;
  }
  LOG_TS(ts_, "argmax")
}

void RESNET::preprocess_bmcv (vector<bm_image> &input) {
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

  // do linear transform
  LOG_TS(ts_, "linear transform")
  bmcv_image_convert_to(bm_handle_, input.size(), linear_trans_param_, &input[0], linear_trans_bmcv_);
  LOG_TS(ts_, "linear transform")
}
