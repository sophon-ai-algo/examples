#include <boost/filesystem.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include "lprnet.hpp"
#include <string>

namespace fs = boost::filesystem;
using namespace std;

static vector<string> detect(bm_handle_t         &bm_handle,
                             LPRNET              &net,
                             cv::Mat             &image,
                             TimeStamp           *ts) {

  vector<string> detections;
  vector<cv::Mat> images;
  images.push_back (image);

  vector<bm_image> input_img_bmcv;
  ts->save("attach input");
  bm_image_from_mat(bm_handle, images, input_img_bmcv);
  ts->save("attach input");

  ts->save("detection");
  net.preForward (input_img_bmcv);

  // do inference
  net.forward();

  net.postForward (input_img_bmcv , detections);
  ts->save("detection");

  // destory bm_image
  for (size_t i = 0; i < input_img_bmcv.size();i++) {
    bm_image_destroy (input_img_bmcv[i]);
  }
  return detections;
}


int main(int argc, char** argv) {
  //system("chcp 65001");
  cout.setf(ios::fixed);

  // sanity check
  if (argc != 5) {
    cout << "USAGE:" << endl;
    cout << "  " << argv[0] << " <mode> <image path> <bmodel path> <device id>" << endl;
    exit(1);
  }

  string mode = argv[1];
  if (strcmp(mode.c_str(), "test") && strcmp(mode.c_str(), "val")){
    cout << "mode must be test or val." << endl;
  }

  string bmodel_file = argv[3];
  if (!fs::exists(bmodel_file)) {
    cout << "Cannot find valid model file." << endl;
    exit(1);
  }

  string input_url = argv[2];
  //cout << "is_directory: " << fs::is_directory(input_url) << endl;
  //cout << "is_regular_file: " << fs::is_regular_file(input_url) << endl;
  if (!fs::exists(input_url)) {
    cout << "Cannot find input image path." << endl;
    exit(1);
  }

  // set device id
  string dev_str = argv[4];
  stringstream checkdevid(dev_str);
  double t;
  if (!(checkdevid >> t)) {
    cout << "Is not a valid dev ID: " << dev_str << endl;
    exit(1);
  }
  int dev_id = stoi(dev_str);
  cout << "set device id:"  << dev_id << endl;

  // profiling
  TimeStamp lprnet_ts;
  TimeStamp *ts = &lprnet_ts;

  // initialize handle of low level device
  int max_dev_id = 0;
  bm_dev_getcount(&max_dev_id);
  if (dev_id >= max_dev_id) {
      cout << "ERROR: Input device id=" << dev_id
                << " exceeds the maximum number " << max_dev_id << endl;
      exit(-1);
  }
  bm_handle_t  bm_handle;
  bm_status_t ret = bm_dev_request (&bm_handle, dev_id);
  if (ret != BM_SUCCESS) {
    cout << "Initialize bm handle failed, ret = " << ret << endl;
    exit(-1);
  }

  // initialize LPRNET class
  LPRNET net(bm_handle , bmodel_file);

  // for profiling
  net.enableProfile(ts);

  bool val_flag = !strcmp(mode.c_str(), "val");
  int tp = 0;
  int cn = 0;

  if (fs::is_regular_file(input_url)){
    //fs::path image_file(input_url);
    //string name = image_file.filename().string();
    ts->save("lprnet overall");
    ts->save("read image");
    // decode jpg file to Mat object
    cv::Mat img = cv::imread(input_url, cv::IMREAD_COLOR, dev_id);
    ts->save("read image");
    // do detect
    vector<string> results = detect(bm_handle, net, img, ts);
    ts->save("lprnet overall");
    // output results
    cout << input_url << " pred: " << results[0] <<endl;
  }
  else if (fs::is_directory(input_url)){
    fs::recursive_directory_iterator beg_iter(input_url);
    fs::recursive_directory_iterator end_iter;
    for (; beg_iter != end_iter; ++beg_iter){
      if (fs::is_directory(*beg_iter)) continue;
      else {
        string img_file = beg_iter->path().string();
        //cout << img_file << endl;
        cv::Mat img = cv::imread(img_file, cv::IMREAD_COLOR, dev_id);
        vector<string> results = detect(bm_handle, net, img, ts);
        cout << img_file << " pred: " << results[0] <<endl;
        if (val_flag){
          cn++;
          fs::path image_file(img_file);
          string label = image_file.stem().string();
          if (!strcmp(label.c_str(), results[0].c_str())) tp++;
          else cout << "label:" << label << " pred:" << results[0] << endl;
        }
      }
    }
    if (val_flag) cout << "Acc = " << tp << "/" << cn << "=" \
    << float(tp)/cn << endl;
  }else {
    cout << "Is not a valid path: " << input_url << endl;
    exit(1);
  }

  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  lprnet_ts.calbr_basetime(base_time);
  lprnet_ts.build_timeline("lprnet detect");
  lprnet_ts.show_summary("lprnet detect");
  lprnet_ts.clear();

  bm_dev_free(bm_handle);
  cout << endl;
  return 0;
}
