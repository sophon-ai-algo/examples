#include <boost/filesystem.hpp>
#include "ssd.hpp"

namespace fs = boost::filesystem;
using namespace std;

static void detect(bm_handle_t         bm_handle,
                   SSD                 &net,
                   cv::Mat             &image,
                   string              name,
                   TimeStamp           *ts) {

  vector<vector<ObjRect>> detections;
  vector<cv::Mat> images;
  images.push_back (image);

  ts->save("detection");
  net.preForward (images);

  // do inference
  net.forward();

  net.postForward (images , detections);
  ts->save("detection");

  // save results to jpg file
  for (size_t i = 0; i < detections.size(); i++) {
    for (size_t j = 0; j < detections[i].size(); j++) {
      ObjRect rect = detections[i][j];
      cv::rectangle(image, cv::Rect(rect.x1, rect.y1, rect.x2 - rect.x1 + 1,
                                    rect.y2 - rect.y1 + 1), cv::Scalar(255, 0, 0), 2);
    }
    // jpg encode
    cv::imwrite("out-batch-" + name, image);
  }

}

int main(int argc, char **argv) {

  cout.setf(ios::fixed);

  // sanity check
  if (argc != 5) {
    cout << "USAGE:" << endl;
    cout << "  " << argv[0] << " image <image file> <bmodel path> <test count>" << endl;
    cout << "  " << argv[0] << " video <video url>  <bmodel path> <test count>" << endl;
    exit(1);
  }

  string bmodel_file = argv[3];
  if (!fs::exists(bmodel_file)) {
    cout << "Cannot find valid model file." << endl;
    exit(1);
  }

  if (strcmp(argv[1], "video") != 0 && strcmp(argv[1], "image") != 0){
    cout << "mode must be image or video" << endl;
    exit(1);
  }

  bool is_video = false;
  if (strcmp(argv[1], "video") == 0)
    is_video = true;

  string input_url = argv[2];
  if (!is_video && !fs::exists(input_url)) {
    cout << "Cannot find input image file." << endl;
    exit(1);
  }

  unsigned long test_loop = stoul(string(argv[4]), nullptr, 0);

  // profiling
  TimeStamp ssd_ts;
  TimeStamp *ts = &ssd_ts;

  // initialize handle of low level device
  bm_handle_t bm_handle;
  bm_status_t ret = bm_dev_request (&bm_handle, 0);
  if (ret != BM_SUCCESS) {
    cout << "Initialize bm handle failed, ret = " << ret << endl;
    exit(-1);
  }

  // initialize SSD class
  SSD net(bm_handle , bmodel_file);

  // for profiling
  net.enableProfile(ts);

  // decode and detect
  if (!is_video) {

    fs::path image_file(input_url);
    string name = image_file.filename().string();
    for (uint32_t i = 0; i < test_loop; i++) {
      ts->save("ssd overall");
      ts->save("read image");

      // decode jpg file to Mat object
      cv::Mat img = cv::imread(input_url);

      ts->save("read image");

      // do detect
      string img_out = "t_" + to_string(i) + "_" + name;
      detect(bm_handle, net, img, img_out, ts);
      ts->save("ssd overall");
    }

  } else {

    // open stream
    cv::VideoCapture cap(input_url);
    if (!cap.isOpened()) {
      cout << "open stream " << input_url << " failed!" << endl;
      exit(1);
    }

    // get resolution
    int w = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cout << "resolution of input stream: " << h << "," << w << endl;

    // set output format to YUV-nv12
    cap.set(cv::CAP_PROP_OUTPUT_YUV, 1.0);

    for (uint32_t c = 0; c < test_loop; c++) {

      // get one frame from decoder
      cv::Mat *p_img = new cv::Mat;
      cap.read(*p_img);

      // sanity check
      if (p_img->avRows() != h || p_img->avCols() != w) {
        if (p_img != nullptr) delete p_img;
        continue;
      }

      // do detct
      string img_out = "t_" + to_string(c) + "_video.jpg";
      detect(bm_handle, net, *p_img, img_out, ts);

      // release Mat object
      if (p_img != nullptr) delete p_img;
    }
  }

  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  ssd_ts.calbr_basetime(base_time);
  ssd_ts.build_timeline("ssd detect");
  ssd_ts.show_summary("detect ");
  ssd_ts.clear();

  bm_dev_free(bm_handle);

  cout << endl;

  return 0;
}
