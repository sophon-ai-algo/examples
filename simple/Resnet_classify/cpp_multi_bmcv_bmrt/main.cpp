#include <chrono>
#include <condition_variable>
#include <mutex>
#include <iostream>
#include <sstream>
#include <thread>
#include <boost/atomic.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/filesystem.hpp>
#include <boost/thread/thread.hpp>
#include <pthread.h>
#include <opencv2/core.hpp>
#include "resnet.hpp"

namespace fs = boost::filesystem;
using namespace std;
using namespace chrono_literals;

std::mutex g_ts_mtx;

boost::atomic_int image_count(0);
boost::atomic_int decode_count(0);
boost::atomic_int preproc_count(0);
boost::atomic_int infer_count(0);
boost::atomic_int result_count(0);

mutex decode_mtx;
mutex preproc_mtx;
mutex infer_mtx;

condition_variable decode_prod, decode_cons;
condition_variable preproc_prod, preproc_cons;
condition_variable infer_prod, infer_cons;

boost::atomic<bool> decode_last(false);
boost::atomic<bool> preproc_last(false);
boost::atomic<bool> infer_last(false);
boost::atomic<bool> output_last(false);

class ImageData;

boost::lockfree::queue<ImageData *, boost::lockfree::fixed_sized<true>> decode_queue(16);
boost::lockfree::queue<ImageData *, boost::lockfree::fixed_sized<true>> preproc_queue(16);
boost::lockfree::queue<ImageData *, boost::lockfree::fixed_sized<true>> infer_queue(16);
boost::lockfree::queue<ImageData *, boost::lockfree::fixed_sized<true>> output_queue(512);

list<ImageData *> sorted_outputs;

class ImageData {

public:
  int imageId;             // image index from base filename
  string filename;         // full path filename
  string basename;         // filename
  cv::Mat cvMat;           // OpenCV Mat object
  bm_image bmi;            // bm_image object
  vector<ObjRect> objects; // detection result
};

bm_handle_t g_bm_handle; // device handle
string g_bmodel_file;    // bmodel file

const int decode_threads_num = 4;
const int infer_threads_num = 2;

//#define PER_THREAD_PROFILING
#define DUMP_RESULTS
//#define IMAGENET_VALSET_TEST 1
//#define DEBUG_EN


void input_collector(char *image_path) {

  fs::path dir(image_path);
  if (fs::exists(dir)) {
    fs::directory_iterator it(dir), end;
    for (; it != end; it++) {
      if (fs::is_regular_file(it->status())) {
        size_t pos_temp = it->path().string().find_last_of(".");
        if(pos_temp <= 0){
          continue;
        }
        std::string temp_format = it->path().string().substr(pos_temp+1,it->path().size());
        if(temp_format != "jpg" && temp_format != "png" && temp_format != "bmp"){
          continue;
        }


        ImageData *d = new ImageData;
        d->filename = it->path().string();
        d->basename = it->path().filename().string();

        size_t pos = d->basename.find(".");
        string s = d->basename.substr(0, pos);
        pos = s.find_last_of("_");
        string n = s.substr(pos+1);
        n.erase(0, n.find_first_not_of('0'));
        d->imageId = atoi(n.c_str());

        unique_lock<mutex> lck(decode_mtx);
        while (!decode_queue.push(d))
          decode_prod.wait(lck);
        lck.unlock();
        decode_cons.notify_all();

        ++image_count;
      }
    }
  }
}

void decode(ImageData *d, TimeStamp &t) {

  t.save("decode");
#if defined(IMAGENET_VALSET_TEST)
  /*
   * The imagenet validation set contains a few gray images, which
   * will resort to soft-decode. If using YUV as output format, it
   * requires 'UPLOAD' operations to synchronize the YUV image data
   * from system memory (Host) to device memory (SC5). Thus, it is
   * preferred to use BGR as output format.
   * 
   * In the meanwhile, since too many warnings by soft-decode are
   * present during the test, it's even recommended to change to
   * soft-decode mode thoroghtly by means of setting the following
   * ENV variable.
   * 
   *    export USE_SOFT_JPGDEC=1
   * 
   * For sure, soft-decode will slow down the performance at a cost.
   */
  d->cvMat = cv::imread(d->filename);
#else
  d->cvMat = cv::imread(d->filename, cv::IMREAD_AVFRAME);
#endif
  t.save("decode");

  ++decode_count;

  unique_lock<mutex> lck(preproc_mtx);
  while (!preproc_queue.push(d))
    preproc_prod.wait(lck);
  lck.unlock();
  preproc_cons.notify_all();
}

void image_decoder(int thread_id) {

  stringstream ss;
  string name;

  ss << "decode-" << thread_id;
  ss >> name;

#ifdef __linux__
  pthread_setname_np(pthread_self(), name.c_str());
#endif

  TimeStamp t;
  ImageData *d;

  while (!decode_last) {
    unique_lock<mutex> lck(decode_mtx);
    decode_cons.wait(lck, [&d]{ return decode_queue.pop(d); });
    lck.unlock();
    decode_prod.notify_all();

    decode(d, t);
  }

  while (decode_queue.pop(d))
    decode(d, t);

#ifdef PER_THREAD_PROFILING
  t.show_summary("pre-process thread - " + to_string(thread_id));
#endif
}

void preprocess(ImageData *d, TimeStamp &t) {

  bm_image in;

  t.save("preprocess");

  bm_image_from_mat(g_bm_handle, d->cvMat, in);

  bm_image_create(g_bm_handle,
                  INPUT_HEIGHT,
                  INPUT_WIDTH,
                  FORMAT_BGR_PLANAR,
                  DATA_TYPE_EXT_1N_BYTE,
                  &d->bmi,
                  NULL);

  bmcv_rect_t crop_rect = {0, 0, in.width, in.height};
  t.save("resize & split");
  bmcv_image_vpp_convert(g_bm_handle, 1, in, &d->bmi, &crop_rect, BMCV_INTER_LINEAR);
  t.save("resize & split");
  d->cvMat.release();

  t.save("preprocess");

  unique_lock<mutex> lck(infer_mtx);
  while (!infer_queue.push(d))
    infer_prod.wait(lck);
  lck.unlock();
  infer_cons.notify_all();

  preproc_count++;
}

void image_preproc(int thread_id) {

  stringstream ss;
  string name;

  ss << "preprocess-" << thread_id;
  ss >> name;
#ifdef __linux__
  pthread_setname_np(pthread_self(), name.c_str());
#endif

  ImageData *d;
  TimeStamp t;

  while (!preproc_last) {
    unique_lock<mutex> lck(preproc_mtx);
    if (!preproc_cons.wait_for(lck, 10ms, [&d]{ return preproc_queue.pop(d); }))
      continue;
    lck.unlock();
    preproc_prod.notify_all();

    preprocess(d, t);
  }

  while (preproc_queue.pop(d))
    preprocess(d, t);

#ifdef PER_THREAD_PROFILING
  t.show_summary("pre-process thread - " + to_string(thread_id));
#endif
}

#define MAX_INFR_THREADS 4
vector<ImageData *> batch_cache[MAX_INFR_THREADS];
int cache_count[MAX_INFR_THREADS] = {0};
vector<bm_image> batched_images[MAX_INFR_THREADS];

void net_inference(RESNET *net, int tid, ImageData *d, TimeStamp &t) {

  ++infer_count;

  batch_cache[tid].push_back(d);
  batched_images[tid].push_back(d->bmi);
  ++cache_count[tid];

  if (cache_count[tid] < 4)
      return;

  net->preForward(batched_images[tid]);

  t.save("do inference");
  net->forward();
  t.save("do inference");

  vector<vector<ObjRect>> batch_detects;
  t.save("do postprocess");
  net->postForward(batched_images[tid], batch_detects);
  t.save("do postprocess");
  if (batch_detects.size() != 4) {
    cout << "** results doesn't match batch size" << endl;
  }

  for (size_t i = 0; i < batch_cache[tid].size(); i++) {
    ImageData *d = batch_cache[tid][i];
    d->objects.swap(batch_detects[i]);
    while (!output_queue.push(batch_cache[tid][i]));
  }

  batch_cache[tid].clear();
  cache_count[tid] = 0;

  for (size_t i = 0; i < batched_images[tid].size(); i++)
    bm_image_destroy(batched_images[tid][i]);
  batched_images[tid].clear();
}

void move_unbatched(int tid) {

  for (size_t i = 0; i < batch_cache[tid].size(); i++)
    while (!infer_queue.push(batch_cache[tid][i]));

  infer_cons.notify_all();
}

RESNET *g_nets[MAX_INFR_THREADS] = {0};
boost::atomic_int thread_mask(0);

void net_infer(int thread_id) {

  stringstream ss;
  string name;

  ss << "inference-" << thread_id;
  ss >> name;
#ifdef __linux__
  pthread_setname_np(pthread_self(), name.c_str());
#endif

  RESNET *net = g_nets[thread_id];

  TimeStamp t;
  net->enableProfile(&t);

  thread_mask |= 1 << thread_id;

  ImageData *d;

  while (!infer_last) {
    unique_lock<mutex> lck(infer_mtx);
    if (!infer_cons.wait_for(lck, 10ms, [&d]{ return infer_queue.pop(d); }))
      continue;
    lck.unlock();
    infer_prod.notify_all();
 
    net_inference(net, thread_id, d, t);
  }

  if (thread_id == 0) {
    while (thread_mask != 0x1 || cache_count[0] != 0 || image_count != infer_count)
      while (infer_queue.pop(d))
        net_inference(net, thread_id, d, t);
  } else {
    if (cache_count[thread_id] != 0) {
      infer_count -= cache_count[thread_id];
      move_unbatched(thread_id);
    }
  }

  thread_mask &= ~(1 << thread_id);

#ifdef PER_THREAD_PROFILING
  t.show_summary("net-infer thread - " + to_string(thread_id));
#endif
}

void output_collector(void) {

  ImageData *d;

  while (!output_last)
    if (output_queue.pop(d))
      sorted_outputs.push_back(d);
    else
      this_thread::sleep_for(chrono::milliseconds(2));

  while (output_queue.pop(d))
    sorted_outputs.push_back(d);
}

static bool compare_name(const ImageData* first, const ImageData* second) {
  return first->imageId < second->imageId;
}

void dump_output() {
#ifdef VERBOSE_PRINT
  vector<string> labels;
  ifstream f;
  string filename = "synset_words.txt";

  f.open(filename, ios::in);
  string label;
  while (std::getline(f, label))
    labels.push_back(label);
  f.close();
#endif

#ifdef VERBOSE_PRINT
  cout << endl << "------------------" << endl;
  cout << "Output Dumps" << endl;
  cout << "  total labels: "<< labels.size() << endl;
  cout << "------------------" << endl;
#endif

  sorted_outputs.sort(compare_name);

  list<ImageData *>::iterator iter;
#ifndef VERBOSE_PRINT
  fstream f("result.txt", ios::out);
#endif
  for (iter = sorted_outputs.begin(); iter != sorted_outputs.end(); iter++) {
    ImageData *d = *iter;
    if (d->objects.size() > 0) {
#ifdef VERBOSE_PRINT
      cout << d->basename << ": class = " << labels[d->objects[0].class_id]
           << ", score = " << d->objects[0].score << endl;
#else
      //f << d->basename << " ";
      for (size_t i = 0; i < d->objects.size(); ++i)
        f << d->objects[i].class_id << " ";
      f << endl;
#endif
    } else
      cout << d->basename << ": unknown class!" << endl;
  }
#ifndef VERBOSE_PRINT
  f.close();
#endif
}

void init(int thread_num, int device_id) {

  bm_dev_request(&g_bm_handle, device_id);
  for (int i = 0; i < thread_num; ++i)
    g_nets[i] = new RESNET(g_bm_handle, g_bmodel_file);
}

void cleanup(int thread_num) {

  while (!sorted_outputs.empty()) {
    ImageData *d = sorted_outputs.front();
    delete d;
    sorted_outputs.pop_front();
  }

  for (int i = 0; i < thread_num; ++i)
    delete g_nets[i];
  bm_dev_free(g_bm_handle); 
}

void process_monitor() {
  while (!output_last) {
    cout << "\r" << "input/output: " << image_count << "/" << sorted_outputs.size() << std::flush;
    this_thread::sleep_for(chrono::milliseconds(200));
  }
  cout << endl;
}



int main(int argc, char **argv) {

  cout.setf(ios::fixed);

  if (argc < 3) {
    cout << "USAGE:" << endl;
    cout << "  " << argv[0] << " <image path> <bmodel file> [preprocess-threads num]" << endl;
    exit(1);
  }

  int device_id = 0;
  if(argc >= 5){
    device_id = atoi(argv[4]);
  }

  thread input_thread = thread(input_collector, argv[1]);
  thread output_thread = thread(output_collector);
  thread monitor_thread = thread(process_monitor);

  g_bmodel_file = argv[2];
  cout << "bmodel file: " << g_bmodel_file << endl;
  if (!fs::exists(g_bmodel_file)) {
    cerr << "**  bmodel file doesn't exist!" << endl;
    exit(1);
  }
  int threads_num = 1;
  if (argc == 4) {
     threads_num = atoi(argv[3]);
     cout << "Use " << threads_num << " for pre-processing" << endl;
     if (threads_num == 0)
       threads_num = 1;
  }

  init(infer_threads_num,device_id);

  TimeStamp t;

  boost::thread_group decode_threads, preproc_threads, infer_threads;

  t.save("total");

  for (int i = 0; i != decode_threads_num; ++i)
    decode_threads.create_thread([i]{image_decoder(i);});

  for (int i = 0; i != threads_num; ++i)
    preproc_threads.create_thread([i]{image_preproc(i);});

  for (int i = 0; i != infer_threads_num; ++i)
    infer_threads.create_thread([i]{net_infer(i);});

  input_thread.join();

  decode_last = true;
  decode_threads.interrupt_all();
  decode_threads.join_all();
#ifdef DEBUG_EN
  cout << "decode jobs completed" << endl;
#endif

  preproc_last = true;
  preproc_threads.interrupt_all();
  preproc_threads.join_all();
#ifdef DEBUG_EN
  cout << "preprocess jobs completed" << endl;
#endif

  infer_last = true;
  infer_threads.interrupt_all();
  infer_threads.join_all();
#ifdef DEBUG_EN
  cout << "inference jobs completed" << endl;
#endif

  output_last = true;
  output_thread.join();

  t.save("total");

  monitor_thread.join();

  t.show_summary("RESNET test");

  cout << endl;
  cout << "input total = " << image_count
       << ", decoded = " << decode_count
       << ", preprocssed = " << preproc_count
       << ", inferenced = " << infer_count
       << endl;
  cout << "output total = " << sorted_outputs.size() << endl;

#ifdef DUMP_RESULTS
  dump_output();
#endif
  cleanup(infer_threads_num);

  return 0;
}