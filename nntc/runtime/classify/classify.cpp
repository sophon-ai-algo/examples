#include <opencv2/opencv.hpp>
#include "string.h"
#include "util.h"

using namespace std;

/**
 * Map input layer memory to Opencv Mat
 * Opencv:  bgr bgr bgr
 *          bgr bgr bgr
 *          bgr bgr bgr
 * TPU:     b b b
 *          b b b
 *          b b b
 *          g g g
 *          g g g
 *          g g g
 *          r r r
 *          r r r
 *          r r r
 */
void WrapInputLayer(std::vector<cv::Mat> *input_channels, float *input_data, const int c,
                    const int height, const int width) {
  for (int i = 0; i < c; ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    cout << "usage: ./test bmodel_path image" << endl;
  }
  string context_dir = argv[1];
  if (context_dir[context_dir.length() - 1] != '/') {
    context_dir += "/";
  }
  context_dir += "compilation.bmodel";
  int dev_count = 0;
  bm_status_t status = bm_dev_getcount(&dev_count);
  if (BM_SUCCESS != status) {
    cout << "can not get device count on the host" << endl;
    return -1;
  }

  /**
   * Create bmruntime context and load bmodel
   * Load Bitmain net model(bmodel) which compile from BMcompiler
   **/
  bm_handle_t handle;
  status = bm_dev_request(&handle, 0);
  if (BM_SUCCESS != status) {
    cout << "can not get device on the host" << endl;
    return false;
  }

  void *p_bmrt = bmrt_create(handle);
  bmrt_load_bmodel(p_bmrt, context_dir.c_str());

  /**
   * Get net numuber and net name form bmodel
   **/
  int net_num = bmrt_get_network_number(p_bmrt);
  const char **network_names;
  bmrt_get_network_names(p_bmrt, &network_names);

  std::string image_name = argv[2];
  cv::Mat image = cv::imread(image_name.c_str());
  cv::Mat sample_single, resized;
  image.copyTo(sample_single);

  /**
   * Launch each net in bmodel in three steps
   **/
  for (int net_idx = 0; net_idx < net_num; ++net_idx) {
    /**
     * Step1: Prepare input data
     **/
    string net_name = network_names[net_idx];
    std::cout << "net name :" << net_name << std::endl;
    const bm_net_info_t *network_info = bmrt_get_network_info(p_bmrt, net_name.c_str());
    int stage_num = network_info->stage_num;
    cout << "stage num: " << stage_num << endl;

    int input_num = network_info->input_num;
    int output_num = network_info->output_num;
    std::vector<string> input_name;
    std::vector<string> output_name;

    void *output_datas[output_num];
    for (int input_idx = 0; input_idx < input_num; ++input_idx)
      input_name.push_back(network_info->input_names[input_idx]);
    assert(network_info->input_num == 1);
    for (int output_idx = 0; output_idx < output_num; ++output_idx)
      output_name.push_back(network_info->output_names[output_idx]);

    bm_tensor_t *input_tensors = new bm_tensor_t[input_num];
    for (int i = 0; i < input_num; i++) {
      bm_shape_t input_shape = network_info->stages[0].input_shapes[i];
      bmrt_tensor(&input_tensors[i], p_bmrt, network_info->input_dtypes[i], input_shape);

      int ws = input_shape.dims[3];
      int hs = input_shape.dims[2];
      int c = input_shape.dims[1];
      cout << "net idx:" << net_idx << " c h w " << c << " " << hs << " " << ws << " " << endl;
      float *input_data = new float[ws * hs * c];
      cv::resize(sample_single, resized, cv::Size(ws, hs), 0, 0, cv::INTER_NEAREST);
      resized.convertTo(resized, CV_32FC3, 0.0078125, -127.5 * 0.0078125);

      /**
       * Change input memory storage format form Opencv to BMTPU
       **/
      std::vector<cv::Mat> input_channels;
      WrapInputLayer(&input_channels, input_data, c, hs, ws);
      cv::split(resized, input_channels);

      // Copy input data from system memory to device memory
      bm_memcpy_s2d(handle, input_tensors[i].device_mem, input_data);
      delete[] input_data;
    }
    bm_tensor_t *output_tensors = new bm_tensor_t[output_num];

    /**
     * Step2: Lauches the Bitmain Net on the TPU
     **/
    double start_time = what_time_is_it_now();
    bool ret = bmrt_launch_tensor(p_bmrt, net_name.c_str(), input_tensors, input_num,
                                  output_tensors, output_num);
    if (!ret) {
      printf("+++ launch failed, launch net[%s] stage[%d] +++\n", net_name.c_str(), 0);
      exit(-1);
    }
    // sync, wait for finishing inference or get an undefined output
    status = bm_thread_sync(handle);
    if (BM_SUCCESS != status) {
      printf("+++ thread sync failed, net[%s] stage[%d] +++\n", net_name.c_str(), 0);
      exit(-1);
    }

    /**
     * Step3: Get output form TPU
     **/
    for (int i = 0; i < output_num; ++i) {
      auto &output_tensor = output_tensors[i];
      size_t size = bmrt_tensor_bytesize(&output_tensor);
      output_datas[i] = malloc(size);

      bm_memcpy_d2s_partial(handle, output_datas[i], output_tensor.device_mem, size);

      // Free output device memory
      bm_free_device(handle, output_tensors[i].device_mem);

      vector<float> prob_vec((float *)output_datas[i],
                             (float *)output_datas[i] + size / sizeof(float));
      vector<float>::iterator biggest = max_element(begin(prob_vec), end(prob_vec));
      cout << "Max element is " << *biggest << " at position " << distance(begin(prob_vec), biggest)
           << endl;
    }

    // Free input device memory
    for (int i = 0; i < input_num; i++) {
      bm_free_device(handle, input_tensors[i].device_mem);
    }

    double start_end = what_time_is_it_now();
    // long unsigned int last_api_process_time_us = 0;
    // bmrt_get_last_api_process_time_us(p_bmrt, &last_api_process_time_us);
    // printf("the last_api_process_time_us is %lu us\n", last_api_process_time_us);
    printf("the total time comsumption  is %f s\n", start_end - start_time);

    for (int i = 0; i < output_num; ++i) {
      free(output_datas[i]);
    }
    delete[] input_tensors;
    delete[] output_tensors;
  }

  // Destroy bmruntime context
  bmrt_destroy(p_bmrt);
}
