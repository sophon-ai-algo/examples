#include <boost/filesystem.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include "lprnet.hpp"
#include <string>
#include "opencv2/opencv.hpp"

namespace fs = boost::filesystem;
using namespace std;

static vector<string> detect(LPRNET              &net,
                             vector<cv::Mat>     &images,
                             TimeStamp           *ts) {

  vector<string> detections;

  ts->save("detection");
  net.preForward(images);

  // do inference
  net.forward();

  net.postForward(detections);
  ts->save("detection");

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

  // initialize handle of low level device
  int max_dev_id = 0;
  bm_dev_getcount(&max_dev_id);
  if (dev_id >= max_dev_id) {
      cout << "ERROR: Input device id=" << dev_id
                << " exceeds the maximum number " << max_dev_id << endl;
      exit(-1);
  }
  // profiling
  TimeStamp lprnet_ts;
  TimeStamp *ts = &lprnet_ts;

  // initialize LPRNET class
  LPRNET net(bmodel_file, dev_id);

  // for profiling
  net.enableProfile(ts);
  int batch_size = net.batch_size();
  bool val_flag = !strcmp(mode.c_str(), "val");
  int tp = 0;
  int cn = 0;
  vector<cv::Mat> batch_imgs;
  vector<string> batch_names;
  
  if (fs::is_regular_file(input_url)){
    if (batch_size != 1){
      cout << "ERROR: batch_size of model is  " << batch_size << endl;
      exit(-1);
    }
    //fs::path image_file(input_url);
    //string name = image_file.filename().string();
    ts->save("lprnet overall");
    ts->save("read image");
    // decode jpg file to Mat object
    cv::Mat img = cv::imread(input_url);
    ts->save("read image");
    size_t index = input_url.rfind("/");
    string img_name = input_url.substr(index + 1);
    batch_imgs.push_back(img);
    batch_names.push_back(img_name);
    // do detect
    vector<string> results = detect(net, batch_imgs, ts);
    ts->save("lprnet overall");
    // output results
    cout << input_url << " pred: " << results[0] <<endl;
  }
  else if (fs::is_directory(input_url)){
    fs::recursive_directory_iterator beg_iter(input_url);
    fs::recursive_directory_iterator end_iter;
    ts->save("lprnet overall");
    for (; beg_iter != end_iter; ++beg_iter){
      if (fs::is_directory(*beg_iter)) continue;
      else {
        string img_file = beg_iter->path().string();
        //cout << img_file << endl;
        ts->save("read image");
        cv::Mat img = cv::imread(img_file, cv::IMREAD_COLOR, dev_id);
        ts->save("read image");
        size_t index = img_file.rfind("/");
        string img_name = img_file.substr(index + 1);
        batch_imgs.push_back(img);
        batch_names.push_back(img_name);
        if ((int)batch_imgs.size() == batch_size) {
          vector<string> results = detect(net, batch_imgs, ts);
          for(int i = 0; i < batch_size; i++){
            fs::path image_file(batch_names[i]);
            cout << image_file.string() << " pred:" << results[i].c_str() << endl;
            if (val_flag){
              string label = image_file.stem().string();
              cn++;
              if (!strcmp(label.c_str(), results[i].c_str())) tp++;
            }
          }
          batch_imgs.clear();
          batch_names.clear();
        }        
      }
    }
    ts->save("lprnet overall");
    
    cout << "===========================" << endl;
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

  cout << endl;
  return 0;
}
