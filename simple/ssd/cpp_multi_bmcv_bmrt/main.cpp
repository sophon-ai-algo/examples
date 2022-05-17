/*
* all data in DataQueue struct
* video->opencv decode->video_queue->detect_thread->l1_out_queue->
  feature_extraction_thread->l2_out_queue->convert_jpg_thread
*/
#define __STDC_CONSTANT_MACROS
extern "C"
{
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
}
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/atomic.hpp>
#include <condition_variable>
#include <chrono>
#include <mutex>
#include <thread>
#include "common.hpp"
#include "ssd.hpp"

namespace fs = boost::filesystem;
using namespace std;
bm_handle_t g_bm_handle; //device handle
void* g_bmrt;            //runtime handle

static queue<DataQueue *> video_queue;  //data queue for L1 inference
static queue<DataQueue *> l1_out_queue; //L1 inference out data
static queue<DataQueue *> l2_out_queue; //L2 inference out data
std::mutex stream_lock;                //mutex lock for stream data queue
std::mutex l1_lock;                //mutex lock for l1 out data queue
std::mutex l2_lock;                //mutex lock for l2 out data queue

static string src_file;                 //test video url
static string bmodel_file;               //bmodel for inference
int g_video_thread_num = 1;               //video thread quantity
int g_detect_thread_num = 1;               //detect thread quantity
int g_feature_thread_num = 0;               //feature extraction thread quantity
int g_jpg_thread_num = 0;               //feature extraction thread quantity
unsigned int g_batch_num = 0;            //for inference batch quantity
unsigned int g_test_loops = 1;           //for inference times 
bool g_detect_finished = false;          //detect inference stop flag
bool g_feature_finished = false;          //feature inference stop flag
bool g_convert_finished = false;          //convert jpg  stop flag
bool g_debug_detect = false;              //for debug detect thread
bool g_yuv = false;               //false:BGR,true:yuv for inference input data format
bool g_only_for_detect = true;              

int g_image_queue_max = MAX_IMAGE_QUEUE;
int g_l1out_queue_max = MAX_L1OUT_QUEUE;
int g_l2out_queue_max = MAX_L2OUT_QUEUE;
int g_frame_rate = READY_FRAME_RATE;
int g_decode_flag = 0;             //when value is 0 using opencv decode else using ffmpeg     
//for stat
boost::atomic<uint64_t> g_invalid_img{0};
boost::atomic<uint64_t> g_total_idx{0};
boost::atomic<uint64_t> g_detect_cnt{0};
boost::atomic<uint64_t> g_extraction_cnt{0};
boost::atomic<uint64_t> g_convert_cnt{0};
boost::atomic<uint64_t> g_image_drop_cnt{0};
boost::atomic<uint64_t> g_l1out_drop_cnt{0};
boost::atomic<uint64_t> g_l2out_drop_cnt{0};
boost::atomic<uint64_t> g_null_obj{0};
boost::atomic<uint64_t> g_detect_totals{0};


/*
*fetch data from queue
*return struct DataQueue pointer 
*/
DataQueue* pop_data(std::mutex &queue_lock, queue<DataQueue *> &data_queue) {
  queue_lock.lock();
  DataQueue* read = nullptr;
  if (data_queue.size() > 0) {
    read = data_queue.front();
    data_queue.pop();
  }
  queue_lock.unlock();
  return read;
}

/*
*push data to  queue
*if error return true
*/
bool push_data(std::mutex &queue_lock, queue<DataQueue *> &data_queue, 
               DataQueue * element, uint32_t max_value, boost::atomic<uint64_t> &count) {
  queue_lock.lock();
  if (element == nullptr) {
    cout << "error! push element pointer is null" << endl;
    queue_lock.unlock();
    return true;
  }
  if (data_queue.size() >= max_value) {
    //cout << "queue is full" << endl;
    DataQueue *tmp = data_queue.front();
    data_queue.pop();
    if(tmp->bmimage) {
      bm_image_destroy(*tmp->bmimage);
      free(tmp->bmimage);
    }

    if(tmp->mat) delete tmp->mat;
    delete tmp;
    count++;
    data_queue.push(element);
  } else {
    data_queue.push(element);
  }
  queue_lock.unlock();
  return false;
}

void release_queue(std::mutex &queue_lock, queue<DataQueue *> &data_queue) {
  queue_lock.lock();
  int size = data_queue.size();

  for(int i = 0; i < size; i++) {
    DataQueue *tmp = data_queue.front();
    data_queue.pop();
    if(tmp->bmimage) {
      bm_image_destroy(*tmp->bmimage);
      free(tmp->bmimage);
    }
    if(tmp->mat) delete tmp->mat;
    delete tmp;
  }
  queue_lock.unlock();
}

/*
*opencv decode video thread
*/
void opencv_thread(int thread_id) {
  cv::VideoCapture cap;
  uint64_t num = 0;
  uint8_t rate = 0;
  cout << "opencv decode thread [" << thread_id << "] started: " << gettid() << endl;
  
  if (!cap.isOpened()) {
    cap.open(src_file);
  }

  if (g_yuv) cap.set(cv::CAP_PROP_OUTPUT_YUV, 1.0);  

  while (true) {
    if (g_detect_finished) {
      if(cap.isOpened()) cap.release();
      cout << "Video thread [" << thread_id << "] end: " << gettid() << endl;
      break;
    }

    if(!cap.isOpened()) {
      cout << "cap is not open! id:" << thread_id << endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }

    cv::Mat *img = new cv::Mat;
    cap.read(*img);

    rate++;
    if (rate%READY_FRAME_RATE > 0) {
      if (img) delete img;
      continue;
    }
    rate = 0;

    num++;
    g_total_idx++;

    bm_image *bmimg = (bm_image *) malloc(sizeof(bm_image));
    bm_image_from_mat (g_bm_handle, *img, *bmimg);
    DataQueue *img_task = new DataQueue();
    img_task->id = thread_id;
    img_task->mat = img;
    img_task->bmimage = bmimg;
    img_task->num = num;
    bool ret = push_data(stream_lock, video_queue, img_task,
                         MAX_IMAGE_QUEUE, g_image_drop_cnt);
    if(ret) {
      cout << "** push image to queue error! id:" << thread_id << endl;
    }
    //std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  cout << "** opencv decode id:" << thread_id << "exit" << endl;
}

static int decode_packet(int *got_frame, AVFrame *frame, AVCodecContext *video_dec_ctx, AVPacket pkt, int video_stream_idx)
{
  if (!video_dec_ctx) {
    cout<< "video_dec_ctx is null";
    exit(1);
  }

  int ret = 0;
  int decoded = pkt.size;
  int width, height;
  enum AVPixelFormat pix_fmt;

  *got_frame = 0;

  if (pkt.stream_index == video_stream_idx) {

    if (!frame) {
      fprintf(stderr, "Could not allocate frame\n");
      return -1;
    }
    /* decode video frame */
    ret = avcodec_decode_video2(video_dec_ctx, frame, got_frame, &pkt);
    if (ret < 0) {
      fprintf(stderr, "Error decoding video frame (%d)\n", ret);
      return ret;
    }

    if (*got_frame) {
      //printf("got frame...\n");
      width = video_dec_ctx->width;
      height = video_dec_ctx->height;
      pix_fmt = video_dec_ctx->pix_fmt;

      if (frame->width != width || frame->height != height ||
	  frame->format != pix_fmt) {
	/* To handle this change, one could call av_image_alloc again and
	 * decode the following frames into another rawvideo file. */
        fprintf(stderr, "Error: Width, height and pixel format have to be "
		  "constant in a rawvideo file, but the width, height or "
		  "pixel format of the input video changed:\n"
		  "old: width = %d, height = %d, format = %s\n"
		  "new: width = %d, height = %d, format = %s\n",
		  width, height, av_get_pix_fmt_name(pix_fmt),
		  frame->width, frame->height,
		  av_get_pix_fmt_name((AVPixelFormat)frame->format));

        return -1;
      }
      //cout << "** decode w:" << frame->width << endl;
    }
  } else {
    cout << "packet contain audio data ,drop this packet" << endl;
    decoded = pkt.size;
  }

  return decoded;
}

static int open_codec_context(int *stream_idx,
                              AVCodecContext **dec_ctx,
                              AVFormatContext *fmt_ctx,
                              enum AVMediaType type)
{
  int ret, stream_index;
  AVStream *st;
  AVCodec *dec = NULL;
  AVDictionary *opts = NULL;

  ret = av_find_best_stream(fmt_ctx, type, -1, -1, NULL, 0);
  if (ret < 0) {
    fprintf(stderr, "Could not find %s stream in input file '%s'\n",
	    av_get_media_type_string(type), src_file.c_str());
    return ret;
  } else {
    stream_index = ret;
    st = fmt_ctx->streams[stream_index];

    /* find decoder for the stream */
    dec = avcodec_find_decoder(st->codecpar->codec_id);
    if (!dec) {
      fprintf(stderr, "Failed to find %s codec\n",
	      av_get_media_type_string(type));
      return AVERROR(EINVAL);
    }

    /* Allocate a codec context for the decoder */
    *dec_ctx = avcodec_alloc_context3(dec);
    if (!*dec_ctx) {
	fprintf(stderr, "Failed to allocate the %s codec context\n",
		av_get_media_type_string(type));
	return AVERROR(ENOMEM);
    }

    /* Copy codec parameters from input stream to output codec context */
    if ((ret = avcodec_parameters_to_context(*dec_ctx, st->codecpar)) < 0) {
      fprintf(stderr, "Failed to copy %s codec parameters to decoder context\n",
	      av_get_media_type_string(type));
      return ret;
    }

// pcie model
#ifdef PCIE_MODE
    av_dict_set_int(&opts, "pcie_board_id", 0, 0);
    av_dict_set_int(&opts, "pcie_no_copyback", 1, 0);
#endif

    /*set compressed output*/
    av_dict_set(&opts, "output_format", "101", 0);

    if ((ret = avcodec_open2(*dec_ctx, dec, &opts)) < 0) {
      fprintf(stderr, "Failed to open %s codec\n",
	       av_get_media_type_string(type));
      return ret;
    }

    *stream_idx = stream_index;
  }

  return 0;
}

/*
*ffmpeg decode video thread
*/
void ffmpeg_thread(int thread_id) {
  cout << "** ffmpeg decode thread [" << thread_id << "] started: "
       << gettid() << " yuv/bgr:" << g_yuv << endl;
  
  AVCodecContext *video_dec_ctx = NULL;
  AVStream *video_stream = NULL;
  int video_stream_idx = -1;
  AVFrame *frame = NULL;
  AVPacket pkt;
  AVFormatContext *fmt_ctx = NULL;
  int ret = 0, got_frame;

  /*decode loop*/
  while (1) {
    /* open input file, and allocate format context */
    if (avformat_open_input(&fmt_ctx, src_file.c_str(), NULL, NULL) < 0) {
      cout << "Could not open source file " << src_file << endl;
      exit(1);
    }

    /* retrieve stream information */
    if (avformat_find_stream_info(fmt_ctx, NULL) < 0) {
      fprintf(stderr, "Could not find stream information\n");
      exit(1);
    }

    if (open_codec_context(&video_stream_idx, &video_dec_ctx, fmt_ctx, AVMEDIA_TYPE_VIDEO) >= 0) {
      video_stream = fmt_ctx->streams[video_stream_idx];
    }

    if (video_stream) {
      cout << "Demuxing video from file " << src_file << endl;
    } else {
      fprintf(stderr, "Could not find video stream in the input, aborting\n");
      exit(1); 
    }

    cout << " ** ffmpeg decode thread:" << thread_id
         << " ** ffmpeg video width:" << video_dec_ctx->width
         << " height:" << video_dec_ctx->height << endl;
    /* dump input information to stderr */
    av_dump_format(fmt_ctx, 0, src_file.c_str(), 0);

    /* initialize packet, set data to NULL, let the demuxer fill it */
    av_init_packet(&pkt);
    pkt.data = NULL;
    pkt.size = 0;

    uint64_t num=0;
    uint8_t rate = 0;

    frame = av_frame_alloc();
    if (!frame) {
      fprintf(stderr, "Could not allocate frame\n");
      return;
    }

    /* read frames from the file */
    while (av_read_frame(fmt_ctx, &pkt) >= 0) {
      AVPacket orig_pkt = pkt;
      if (g_detect_finished) {
        break;
      }
      //cout << "** read pkt" << endl;
      do {
        ret = decode_packet(&got_frame, frame, video_dec_ctx, pkt,
                video_stream_idx);
        if (ret < 0) {
          cout << "** error!! decode_packet error" << endl;
          break;
        }
        pkt.data += ret;
        pkt.size -= ret;

        if ( got_frame == 0) {
          //cout << "** No data obtained in the ffmpeg decoder" << endl;
          break; 
        }
   
        rate++;
        if (rate%READY_FRAME_RATE > 0) {
          break;
        }
        rate = 0;
        num++;
        g_total_idx++;
        bm_image *uncmp_bmimg = (bm_image *) malloc(sizeof(bm_image));
        bm_image_from_frame(g_bm_handle, *frame, *uncmp_bmimg);
        DataQueue *img_task = new DataQueue();
        memset(img_task, 0, sizeof(DataQueue));
        img_task->id = thread_id;
        img_task->bmimage = uncmp_bmimg;
        img_task->num = num;
        bool ret = push_data(stream_lock, video_queue, img_task,
                 MAX_IMAGE_QUEUE, g_image_drop_cnt);
        if(ret) {
          cout << "** push image to queue error! id:" << thread_id << endl;
        }
      } while (pkt.size > 0);
      av_packet_unref(&orig_pkt);
      //std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    /* flush cached frames */
    pkt.data = NULL;
    pkt.size = 0;
    do {
      decode_packet(&got_frame, frame, video_dec_ctx, pkt,video_stream_idx);
    } while (got_frame);

    av_frame_free(&frame);
    avformat_close_input(&fmt_ctx);
    avcodec_free_context(&video_dec_ctx);
    if (g_detect_finished) break;
    cout << "** ffmpeg decode id: " << thread_id << " begin to next loop" << endl;
  }
  cout << "** ffmpeg_decode exited !! thread id:" << thread_id << endl;
}

/*
*detect inference thread
*/
void detect_thread(int thread_id) {
  bool data_ready = false;
  unsigned int read_count = 0;
  unsigned int loop_count = 0;
  vector<DataQueue*> stream_cache;
  vector<bm_image> input_img_bmcv;
  cout << "** detect thread [" << thread_id << "] started: " 
       << gettid() << endl;
  
  SSD net(g_bm_handle, bmodel_file);
  /*in 1 second can not ready image for inference,it will print waring info*/
  uint32_t timeout_cnt = 0;

  while(true) {
    if(loop_count == g_test_loops) {
      cout << "** detect thread:" << thread_id << " end" << endl;
      break;
    }
    //DataQueue * stream_data = pop_stream_data();
    DataQueue * stream_data = pop_data(stream_lock, video_queue);
    if (!stream_data) {
      ++timeout_cnt;
      if (timeout_cnt > 1000) {
        cout << "** waring!! fetch image data timeout" << endl;
        timeout_cnt = 0;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1)); 
      continue;
    } else if(!data_ready) {
      stream_cache.push_back(stream_data);
      input_img_bmcv.push_back(*(stream_data->bmimage));
      read_count ++;
      if(read_count == g_batch_num) {
        data_ready = true;
        read_count = 0;
        //cout << "** thread:" << thread_id << " data is ready" << endl;
      }
    }
    
    if(data_ready) {
      g_detect_cnt += g_batch_num;
      timeout_cnt = 0;
      // do detect
      vector<vector<ObjRect>> detections; 
      
      net.preForward(input_img_bmcv);
      
      // do inference
      net.forward();

      net.postForward(input_img_bmcv, detections);
      if (detections.size() != g_batch_num) {
        cout << "** result not match batch number" << endl;
      }

      for (unsigned int i = 0; i < g_batch_num; i++) {
        if (detections[i].size() == 0) {
          cout << "** Nothing was found in the image:" << i+1 <<endl;
          g_null_obj++;
          bm_image_destroy(*stream_cache[i]->bmimage);
          free(stream_cache[i]->bmimage);
          if(stream_cache[i]->mat) delete stream_cache[i]->mat;
          delete stream_cache[i];
          continue;
        }

        g_detect_totals += detections[i].size();
 
        if(g_debug_detect){
          bmcv_rect_t bmcv_rect;
          for (size_t j = 0; j < detections[i].size(); j++) {
            ObjRect rect = detections[i][j];
            bmcv_rect.start_x = rect.x1;
            bmcv_rect.start_y = rect.y1;
            bmcv_rect.crop_w = rect.x2 - rect.x1 + 1;
            bmcv_rect.crop_h = rect.y2 - rect.y1 + 1;
            bmcv_image_draw_rectangle(g_bm_handle, input_img_bmcv[i], 1, &bmcv_rect, 3, 0, 0, 255);
            stream_cache[i]->rect_out.push_back(rect);
          }
          string image_name = to_string(thread_id) + "d_";
          image_name += to_string(stream_cache[i]->id)+"v_img_";
          image_name += to_string(stream_cache[i]->num) + ".bmp";
          bm_image_write_to_bmp(input_img_bmcv[i], image_name.c_str());
        }

        for (size_t j = 0; j < detections[i].size(); j++) {
          ObjRect rect = detections[i][j];
          stream_cache[i]->rect_out.push_back(rect);
        }
        
        if (g_only_for_detect) {
          bm_image_destroy(*stream_cache[i]->bmimage);
          free(stream_cache[i]->bmimage);
          if(stream_cache[i]->mat) delete stream_cache[i]->mat;
          delete stream_cache[i];
        } else {
          /*save data to L1_out_queue*/
          bool ret = push_data(l1_lock, l1_out_queue, stream_cache[i],
                             MAX_L1OUT_QUEUE, g_l1out_drop_cnt);
          if(ret) cout << "** push l1 out to queue error! id:" << thread_id << endl;
        }
      }
 
      stream_cache.clear();
      input_img_bmcv.clear();
      loop_count++;
      data_ready = false;
    }
  }
}

/*L2 inference for feature extraction
* on the same time,convert_jpg_thread  convert data to jpg.
* 
* convert_jpg_thread and feature_extraction_thread use same data,
* special care when free memory.
* when detect thread end and l1out data queue is null feature 
  extraction thread stop.
*/
void feature_extraction_thread(int thread_id) {
  vector<DataQueue*> data_cache;
  vector<ObjRect> rect;
  uint32_t timeout_cnt = 0;
  cout << "** feature extraction thread [" << thread_id << "] started: " 
       << gettid() << endl;
  while(true) {
    DataQueue * l1_data = pop_data(l1_lock, l1_out_queue);

    if (g_detect_finished && !l1_data) break;

    if (!l1_data) {
      ++timeout_cnt;
      if (timeout_cnt > 10000) {
        cout << "** waring!! fetch L1 out data timeout" << endl;
        timeout_cnt = 0;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5)); 
      continue;
    } else {
      timeout_cnt = 0;
      g_extraction_cnt++;
      int rect_size = l1_data->rect_out.size();
      for (int i=0; i < rect_size; i++) {
        //TODO crop
        rect.push_back(l1_data->rect_out[i]);

      }
      //TODO L2 inference
      std::this_thread::sleep_for(std::chrono::milliseconds(10)); 
      //TODO save result to l2_out_queue
      bool ret = push_data(l2_lock, l2_out_queue, l1_data,
                           MAX_L2OUT_QUEUE, g_l2out_drop_cnt);
      if(ret) {
        cout << "** push l2 out to queue error! id:" << thread_id << endl;
      }
      rect.clear();
    }
  }
}

/*
* convert data to jpg picture
* when feature extraction thread end and data queue null convert jpg thread stop.
*/
void convert_jpg_thread(int thread_id) {
  cout << "** convert jpg thread [" << thread_id << "] started: " 
       << gettid() << endl;
  uint32_t timeout_cnt = 0;
  while(true) {
    DataQueue * l2_data = pop_data(l2_lock, l2_out_queue);
    if(!l2_data && g_feature_finished) break;

    if (!l2_data) {
      ++timeout_cnt;
      if (timeout_cnt > 10000) {
        //cout << "** waring!! fetch L2 out data timeout" << endl;
        timeout_cnt = 0;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    } else {
      //TODO jpeg encode
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
}

/*
*print stat thread
*/
void stat_thread() {
  uint64_t last_idx = 0, cur_idx = 0;
  uint64_t last_detect_idx = 0, cur_detect_idx = 0;
  uint64_t last_extraction_idx = 0, cur_extraction_idx = 0;
  uint64_t last_convert_idx = 0, cur_convert_idx = 0;
  float cur_input_fps = 0;
  float max_input_fps = 0;
  float cur_detect_fps = 0;
  float max_detect_fps = 0;
  float cur_extraction_fps = 0;
  float max_extraction_fps = 0;
  float cur_convert_fps = 0;
  float max_convert_fps = 0;

  boost::posix_time::ptime time_now, time_start;
  boost::posix_time::millisec_posix_time_system_config::time_duration_type time_duration;
  time_start = boost::posix_time::second_clock::universal_time();
  while (true) {

    int c = 0;
    while (c < STAT_INTERVAL*10){ 
      if (g_convert_finished) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      c++;
    }

    time_now = boost::posix_time::second_clock::universal_time();
    time_duration = time_now - time_start;
    cur_idx = g_total_idx;
    cur_detect_idx = g_detect_cnt;
    cur_extraction_idx = g_extraction_cnt;
    cur_convert_idx = g_convert_cnt;
    cur_input_fps = 1.0 * (cur_idx - last_idx) / STAT_INTERVAL;
    cur_detect_fps = 1.0 * (cur_detect_idx - last_detect_idx) / STAT_INTERVAL;
    cur_extraction_fps = 1.0 * (cur_extraction_idx - last_extraction_idx) / STAT_INTERVAL;
    cur_convert_fps = 1.0 * (cur_convert_idx - last_convert_idx) / STAT_INTERVAL;
    last_idx = cur_idx;
    last_detect_idx = cur_detect_idx;
    last_extraction_idx = cur_extraction_idx;
    last_convert_idx = cur_convert_idx;
    if (cur_input_fps > max_input_fps) max_input_fps = cur_input_fps;
    if (cur_detect_fps > max_detect_fps) max_detect_fps = cur_detect_fps;
    if (cur_extraction_fps > max_extraction_fps) max_extraction_fps = cur_extraction_fps;
    if (cur_convert_fps > max_convert_fps) max_convert_fps = cur_convert_fps;

    cout << "Stat of service: " << endl;
    cout << "   Service started(seconds): " << time_duration.total_seconds() << endl;
    cout << "   input YUV(1) or BGR(0): " << g_yuv << endl;
    cout << "   Video thread counts: " << g_video_thread_num << endl;
    cout << "   Detect thread counts: " << g_detect_thread_num << endl;
    cout << "   Feature extraction thread counts: " << g_feature_thread_num << endl;
    cout << "   Convert jpg thread counts: " << g_jpg_thread_num << endl;
    cout << "   Detect batch: " << g_batch_num << endl;
    cout << "   Every Detect thread loops: " << g_test_loops << endl;
    cout << "   Every Video decode rate:1/" << READY_FRAME_RATE << endl;
    cout << "   Detect total results:" << g_detect_totals << endl;
    cout << "   Image drop: " << g_image_drop_cnt << endl;
    cout << "   L1 out drop: " << g_l1out_drop_cnt << endl;
    cout << "   L2 out drop: " << g_l2out_drop_cnt << endl;
    cout << "   Detect Null object: " << g_null_obj << endl;
    cout << "   Invalid imgage: " << g_invalid_img << endl;
    cout << "   Total frames received: " << cur_idx << endl;
    cout << "   Total frames detected: " << cur_detect_idx << endl;
    cout << "   Total frames feature ex: " << cur_extraction_idx << endl;
    cout << "   Total frames convert jpg: " << cur_convert_idx << endl;
    cout << "   Input Speed (fps) cur/max:" << cur_input_fps << "/"
                                             << max_input_fps  << endl;
    cout << "   Detect Speed (fps) cur/max:" << cur_detect_fps << "/"
                                              << max_detect_fps << endl;

    if (g_convert_finished) break;
  }
}

int main(int argc, char **argv) {

  if(argc != 11) {
    cout << "USAGE:" << endl;
    cout << "  " << argv[0] << " video <video url>  <bmodel path>"
         << " <video thread> <detect thread> <batch> <test loops>"
         << " <yuv:1,bgr:0> <decode opencv:0,ffmpeg:1> <debug:0/1>" << endl;
    exit(1);
  }

  thread h_video_thread[MAX_VIDEO_THREAD]; //video decode thread handle 
  thread h_detect_thread[MAX_DETECT_THREAD]; //detect thread handle
  thread h_feature_thread[MAX_FEATURE_EX_THREAD]; //feature extraction thread handle
  thread h_jpg_thread[MAX_JPG_THREAD]; //convert jpg thread handle
  thread h_stat;                            // stat info thread handle

  bool is_video = false;
  if (strcmp(argv[1], "video") == 0)
    is_video = true;

  g_batch_num = stoull(string(argv[6]), nullptr, 0);
  if ((g_batch_num != 1) && (g_batch_num != 4)) {
    cout << "** this demo only suport 1 or 4 batch:" << endl;
    exit(1);
  }
  cout << "batch counts: " << g_batch_num << endl;

  src_file = argv[2];
  if(!is_video && !fs::exists(src_file)) {
    cout << "Cannot find input image file." << endl;
    exit(1);
  }
  cout << "** input url:" << src_file << endl;
  
  bmodel_file = argv[3];
  if (!fs::exists(bmodel_file)) {
    cout << "** Cannot find valid model file." << endl;
    exit(1);
  }
  cout << "** bmodel file:" << bmodel_file << endl;

  g_video_thread_num = stoull(string(argv[4]), nullptr, 0);
  if((g_video_thread_num > MAX_VIDEO_THREAD) || (g_video_thread_num < 1)) {
    cout << "** your error input video thread number is :"
         << g_video_thread_num << endl;
    cout << "** input value in  [1 to " << MAX_VIDEO_THREAD << "]" << endl;
    exit(1);
  }
  cout << "video thread counts: " << g_video_thread_num << endl;
 
  g_detect_thread_num = stoull(string(argv[5]), nullptr, 0);
  if((g_detect_thread_num > MAX_DETECT_THREAD) || (g_detect_thread_num < 1)){
    cout << "** your error input detect thread number is :"
         << g_detect_thread_num << endl;
    cout << "** input value in  [1 to " << MAX_DETECT_THREAD << "]" << endl;
    exit(1);
  }
  cout << "detect thread counts: " << g_detect_thread_num << endl;
 
  g_test_loops = stoull(string(argv[7]), nullptr, 0);
  if (g_test_loops < 1) {
    cout << "** test loops must large 1" << endl;
    exit(1);
  }
  cout << "inference loops : " << g_test_loops << endl;

  bm_dev_request(&g_bm_handle,0);   
  
  int yuv_bgr_flag = stoull(string(argv[8]), nullptr, 0);
  if (yuv_bgr_flag) {
    g_yuv = true;
    cout << "video decode out yuv" << endl;
  }

  g_decode_flag  = stoull(string(argv[9]), nullptr, 0);

  int debug_flag  = stoull(string(argv[10]), nullptr, 0);
  if (debug_flag) {
    g_debug_detect = true;
    cout << "debug open" << endl;
  }

  if(is_video) {

    for(int i = 0; i < g_detect_thread_num; i++) {
      h_detect_thread[i] = thread(detect_thread, i);
    }

    if (!g_only_for_detect) {
      for(int i = 0; i < g_feature_thread_num; i++) {
        h_feature_thread[i] = thread(feature_extraction_thread, i);
      }

      for(int i = 0; i < g_jpg_thread_num; i++) {
        h_jpg_thread[i] = thread(convert_jpg_thread, i);
      }
    }

    for(int i = 0; i < g_video_thread_num; i++) {
      if (g_decode_flag > 0) {
        h_video_thread[i] = thread(ffmpeg_thread, i);
      } else {
        h_video_thread[i] = thread(opencv_thread, i);
      }
    }
    
    h_stat = thread(stat_thread);    

    for(int i = 0; i < g_detect_thread_num; i++) {
      h_detect_thread[i].join();
      cout << "detect thread:" << i << " exit" << endl;
    }
    g_detect_finished = true;

    for(int i = 0; i < g_video_thread_num; i++) {
      h_video_thread[i].join();
      cout << "video thread:" << i << " exit" << endl;
    }

    if (!g_only_for_detect) {
      for(int i = 0; i < g_feature_thread_num; i++) {
        h_feature_thread[i].join();
        cout << "feature extraction thread:" << i << " exit" << endl;
      }
      g_feature_finished = true;

      for(int i = 0; i < g_jpg_thread_num; i++) {
        h_jpg_thread[i].join();
        cout << "convert jpg thread:" << i << " exit" << endl;
      }
      g_convert_finished = true;
    } else {
      g_convert_finished = true;
    }
 
    bm_dev_free(g_bm_handle); 
    release_queue(stream_lock, video_queue);

    if (!g_only_for_detect) {
      release_queue(l1_lock, l1_out_queue);
      release_queue(l2_lock, l2_out_queue);
    }
    h_stat.join();
    cout << "stat thread exit" << endl;
  }
  cout << "main function end " << endl;
  return 0;
}

