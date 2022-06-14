#include <dirent.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "bmlib_runtime.h"
#include "bmruntime_interface.h"

const std::vector<std::string> &bmrt_get_input_tensor_set(
    void *p_bmrt, int net_idx, const std::string &net_name,
    std::vector<std::string> &g_input_tensors) {
  auto net_info = bmrt_get_network_info(p_bmrt, net_name.c_str());
  if (net_info == NULL) {  // do nothing
    return g_input_tensors;
  }
  for (int i = 0; i < net_info->input_num; i++) {
    g_input_tensors.push_back(net_info->input_names[i]);
  }
  return g_input_tensors;
}

const std::vector<std::string> &bmrt_get_output_tensor_set(
    void *p_bmrt, int net_idx, const std::string &net_name,
    std::vector<std::string> &g_output_tensors) {
  auto net_info = bmrt_get_network_info(p_bmrt, net_name.c_str());
  if (net_info == NULL) {  // do nothing
    return g_output_tensors;
  }
  for (int i = 0; i < net_info->output_num; i++) {
    g_output_tensors.push_back(net_info->output_names[i]);
  }
  return g_output_tensors;
}

double what_time_is_it_now() {
  struct timespec now;
  clock_gettime(CLOCK_REALTIME, &now);
  return now.tv_sec + now.tv_nsec * 1e-9;
}
