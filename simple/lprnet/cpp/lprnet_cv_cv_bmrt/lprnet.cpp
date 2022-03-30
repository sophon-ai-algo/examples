#include <fstream>
#include "lprnet.hpp"
#include "utils.hpp"

using namespace std;

string net_name_;
//const char *model_name = "lprnet";
string get_res(int pred_num[], int len_char, int clas_char);

LPRNET::LPRNET(const string bmodel, int dev_id){
  // init device id
  dev_id_ = dev_id;

  //create device handle
  bm_status_t ret = bm_dev_request(&bm_handle_, dev_id_);
  if (BM_SUCCESS != ret) {
      std::cout << "ERROR: bm_dev_request err=" << ret << std::endl;
      exit(-1);
  }

  // init bmruntime contxt
  p_bmrt_ = bmrt_create(bm_handle_);
  if (NULL == p_bmrt_) {
    cout << "ERROR: get handle failed!" << endl;
    exit(1);
  }

#ifdef SOC_MODE
  set_bmrt_mmap(p_bmrt_, true);
#endif

  // load bmodel by file
  bool flag = bmrt_load_bmodel(p_bmrt_, bmodel.c_str());
  if (!flag) {
    cout << "ERROR: Load bmodel[" << bmodel << "] failed" << endl;
    exit(-1);
  }

  const char **net_names;
  bmrt_get_network_names(p_bmrt_, &net_names);
  net_name_ = net_names[0];
  free(net_names);

  // get model info by model name
  auto net_info = bmrt_get_network_info(p_bmrt_, net_name_.c_str());
  if (NULL == net_info) {
    cout << "ERROR: get net-info failed!" << endl;
    exit(1);
  }

  // get data type
  if (NULL == net_info->input_dtypes) {
    cout << "ERROR: get net input type failed!" << endl;
    exit(1);
  }
  cout << "input_dtypes:" << net_info->input_dtypes[0] <<endl;
  if (BM_FLOAT32 == net_info->input_dtypes[0])
    flag_int8 = false;
  else
    flag_int8 = true;  //true

  // allocate output buffer
  output_ = new float[BUFFER_SIZE];

  // only one input shape supported in the pre-built model
  //you can get stage_num from net_info
  int stage_num = net_info->stage_num;
  bm_shape_t input_shape;
  bm_shape_t output_shape;
  for (int i = 0; i < stage_num; i++) {
    if(net_info->stages[i].input_shapes[0].dims[0] == 1) {
      output_shape = net_info->stages[i].output_shapes[0];
      input_shape = net_info->stages[i].input_shapes[0];
      break;
    }

    if ( i == (stage_num - 1)) {
      cout << "ERROR: output not match stages" << endl;
      return;
    }
  }

  //malloc device_memory for inference input and output data
  bmrt_tensor(&input_tensor_, p_bmrt_, net_info->input_dtypes[0], input_shape);
  bmrt_tensor(&output_tensor_, p_bmrt_, net_info->output_dtypes[0], output_shape);

  int count;
  count = bmrt_shape_count(&input_shape);
  cout << "** input count:" << count << endl;
  //malloc system memory for preprocess data
  if (flag_int8) {
    input_int8 = new int8_t[count];
  } else {
    input_f32 = new float[count];
  }
  count = bmrt_shape_count(&output_shape);
  cout << "** output count:" << count << endl;
  output_ = new float[count];

  //input_shape contain dims value(n,c,h,w)
  batch_size_ = input_shape.dims[0];
  num_channels_ = input_shape.dims[1];
  input_geometry_.height = input_shape.dims[2];
  input_geometry_.width = input_shape.dims[3];

  vector<float> mean_values;
  mean_values.push_back(127.5);
  mean_values.push_back(127.5);
  mean_values.push_back(127.5);
  setMean(mean_values);
  //input_scale
  input_scale = 0.0078125 * net_info->input_scales[0];

  ts_ = nullptr;
}

LPRNET::~LPRNET() {
  if (flag_int8) {
    delete []input_int8;
  } else {
    delete []input_f32;
  }
  delete []output_;
  bm_free_device(bm_handle_, input_tensor_.device_mem);
  bm_free_device(bm_handle_, output_tensor_.device_mem);
  bmrt_destroy(p_bmrt_);
  bm_dev_free(bm_handle_);
}

void LPRNET::enableProfile(TimeStamp *ts) {
  ts_ = ts;
}

void LPRNET::preForward(const cv::Mat &image) {
  LOG_TS(ts_, "lprnet pre-process")
  //cout << "input image size:" << image.size() << endl;
  vector<cv::Mat> input_channels;
  wrapInputLayer(&input_channels);
  preprocess(image, &input_channels);
  LOG_TS(ts_, "lprnet pre-process")
}

void LPRNET::forward() {
  if (flag_int8) {
    bm_memcpy_s2d(bm_handle_, input_tensor_.device_mem, ((void *)input_int8));
  } else {
    bm_memcpy_s2d(bm_handle_, input_tensor_.device_mem, ((void *)input_f32));
  }
  LOG_TS(ts_, "lprnet inference")
  bool ret = bmrt_launch_tensor_ex(p_bmrt_, net_name_.c_str(),
                                  &input_tensor_, 1, &output_tensor_, 1, true, false);
  if (!ret) {
    cout << "ERROR: Failed to launch network" << net_name_.c_str() << "inference" << endl;
  }

  // sync, wait for finishing inference
  bm_thread_sync(bm_handle_);
  LOG_TS(ts_, "lprnet inference")

  size_t size = bmrt_tensor_bytesize(&output_tensor_);
  bm_memcpy_d2s_partial(bm_handle_, output_, output_tensor_.device_mem, size);
}

static bool comp(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

void LPRNET::postForward (const cv::Mat &image, vector<string> &detections) {
  auto net_info = bmrt_get_network_info(p_bmrt_, net_name_.c_str());
  int stage_num = net_info->stage_num;
  bm_shape_t output_shape;
  for (int i = 0; i < stage_num; i++) {
    if (net_info->stages[i].input_shapes[0].dims[0] == 1) {
      output_shape = net_info->stages[i].output_shapes[0];
      break;
    }
    if ( i == (stage_num - 1)) {
      cout << "ERROR: output not match stages" << endl;
      return;
    }
  }

  LOG_TS(ts_, "lprnet post-process")
  int output_count = bmrt_shape_count(&output_shape);
  int img_size = batch_size_;
  int count_per_img = output_count/img_size;
  //cout << "img_size = " << img_size << endl;
  //cout << "count_per_img = " << count_per_img << endl;
  detections.clear();

  int N = 1;
  int len_char = net_info->stages[0].output_shapes[0].dims[2];
  int clas_char = net_info->stages[0].output_shapes[0].dims[1];
  //cout << "len_char = " << len_char << endl;
  //cout << "clas_char = " << clas_char << endl;
  //cout << "output_scales=" << net_info->output_scales[0] << endl;


  float *image_output = output_;
  
  vector<std::pair<float , int>> pairs;
  //vector<string> res;
  for (int i = 0; i < img_size; i++) {
    //res.clear();
    int pred_num[len_char]={1000};
    for (int j = 0; j < len_char; j++){
      pairs.clear();
      for (int k = 0; k < clas_char; k++){
        pairs.push_back(make_pair(image_output[i * count_per_img + k * len_char + j], k));
      }
      partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), comp);
      //cout << pairs[0].second << " : " << pairs[0].first << endl;
      pred_num[j] = pairs[0].second;
    }
    string res = get_res(pred_num, len_char, clas_char);
#ifdef DEBUG_RESULT
    cout << "res = " << res << endl;
#endif
    detections.push_back(res);
  }
  LOG_TS(ts_, "lprnet post-process")
}

void LPRNET::setMean(vector<float> &values) {
    vector<cv::Mat> channels;

    for (int i = 0; i < num_channels_; i++) {
      /* Extract an individual channel. */
      if (flag_int8) {
        cv::Mat channel(input_geometry_.height, input_geometry_.width,
                   CV_8SC1,cv::Scalar(0), cv::SophonDevice(this->dev_id_));
        channels.push_back(channel);
      } else {
        cv::Mat channel(input_geometry_.height, input_geometry_.width,
                   CV_32FC1,cv::Scalar(0), cv::SophonDevice(this->dev_id_));
        channels.push_back(channel); 
      }
    }
    //init mat mean_
    vector<cv::Mat> channels_;
    for (int i = 0; i < num_channels_; i++) {
      /* Extract an individual channel. */
      cv::Mat channel_(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar((float)values[i]), cv::SophonDevice(this->dev_id_));
      channels_.push_back(channel_);
    }
    if (flag_int8) {
        mean_.create(input_geometry_.height, input_geometry_.width, CV_8SC3, dev_id_);
    }else{
        mean_.create(input_geometry_.height, input_geometry_.width, CV_32FC3, dev_id_);
    }

    cv::merge(channels_, mean_);
}

void LPRNET::wrapInputLayer(std::vector<cv::Mat>* input_channels) {
  int h = input_geometry_.height;
  int w = input_geometry_.width;


  //init input_channels
  if (flag_int8) {
    int8_t *channel_base = input_int8;
    for (int i = 0; i < num_channels_; i++) {
      cv::Mat channel(h, w, CV_8SC1, channel_base);
      input_channels->push_back(channel);
      channel_base += h * w;
    }
  } else {
    float *channel_base = input_f32;
    for (int i = 0; i < num_channels_; i++) {
      cv::Mat channel(h, w, CV_32FC1, channel_base);
      input_channels->push_back(channel);
      channel_base += h * w;
    }
  }
}

void LPRNET::preprocess (const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
   /* Convert the input image to the input image format of the network. */
  cv::Mat sample = img;
  cv::Mat sample_resized(input_geometry_.height, input_geometry_.width, CV_8UC3, cv::SophonDevice(dev_id_));
  if (sample.size() != input_geometry_) {
    cv::resize(sample, sample_resized, input_geometry_);
  }
  else {
    sample_resized = sample;
  }

  cv::Mat sample_float(cv::SophonDevice(this->dev_id_));
  sample_resized.convertTo(sample_float, CV_32FC3);
  
  cv::Mat sample_normalized(cv::SophonDevice(this->dev_id_));
  cv::subtract(sample_float, mean_, sample_normalized);

  //cout << sample_normalized << endl;
  
  /*note: int8 in convert need mul input_scale*/
  if (flag_int8) {
    cout << "** int8 ** input_scale=" << input_scale << endl;
    cv::Mat sample_int8(cv::SophonDevice(this->dev_id_));
    sample_normalized.convertTo(sample_int8, CV_8SC1, input_scale); 
    cv::split(sample_int8, *input_channels);
  } else {
    //cout << "** f32" << "input_scale:" << input_scale  << endl;
    cv::Mat sample_fp32(cv::SophonDevice(this->dev_id_));
    sample_normalized.convertTo(sample_fp32, CV_32FC3, input_scale);
    cv::split(sample_fp32, *input_channels);
  }
}

string get_res(int pred_num[], int len_char, int clas_char){
  int no_repeat_blank[20];
  //int num_chars = sizeof(CHARS) / sizeof(CHARS[0]);
  int cn_no_repeat_blank = 0;
  int pre_c = pred_num[0];
  if (pre_c != clas_char - 1) {
      no_repeat_blank[0] = pre_c;
      cn_no_repeat_blank++;
  }
  for (int i = 0; i < len_char; i++){
      if (pred_num[i] == pre_c) continue;
      if (pred_num[i] == clas_char - 1){
          pre_c = pred_num[i];
          continue;
      }
      no_repeat_blank[cn_no_repeat_blank] = pred_num[i];
      pre_c = pred_num[i];
      cn_no_repeat_blank++;
  }

  //static char res[10];
  string res="";
  for (int j = 0; j < cn_no_repeat_blank; j++){
    res = res + arr_chars[no_repeat_blank[j]];
    //cout << arr_chars[no_repeat_blank[j]] << endl;
    //strcat(res, arr_chars[no_repeat_blank[j]]);  
  }
  //cout << temp << endl;
  //strcpy(res, temp);
  return res;
}