/*
* all data in DataQueue struct
* video->opencv decode->video_queue->detect_thread->l1_out_queue->
  feature_extraction_thread->l2_out_queue->convert_jpg_thread
*/
#define __STDC_CONSTANT_MACROS
extern "C"
{
#include "libavformat/avformat.h"
#include "libavutil/imgutils.h"
#include "libavcodec/avcodec.h"
}
#include <iostream>
#include <sstream>
#include <boost/filesystem.hpp>
#include <condition_variable>
#include <chrono>
#include <queue>
#include <mutex>
#include <thread>
#include "ssd.hpp"

namespace fs = boost::filesystem;
using namespace std;
bm_handle_t g_bm_handle; //device handle
int g_dev_id;            //device id
void* g_bmrt;            //runtime handle

static queue<bm_image *> bmimg_queue;  //bm_image queue
std::mutex queue_lock;                //mutex lock for queue

static string src_file;                 //test video url
static string bmodel_file;               //bmodel for inference
static unsigned long loops;                   //test loops
struct DataQueue {
  bm_image *bmimage;             //bm_image data
};

/* fetch data from queue */
bm_image* pop_data() {
  queue_lock.lock();
  bm_image* read = nullptr;
  if (bmimg_queue.size() > 0) {
    read = bmimg_queue.front();
    bmimg_queue.pop();
  }
  queue_lock.unlock();
  return read;
}

/* push data to  queue */
bool push_data(bm_image *element) {
  queue_lock.lock();
  if (element == nullptr) {
    cout << "error! push element pointer is null" << endl;
    queue_lock.unlock();
    return 1;
  }
  bmimg_queue.push(element);
  queue_lock.unlock();
  return 0;
}

void release_queue() {
  queue_lock.lock();
  int size = bmimg_queue.size();

  for(int i = 0; i < size; i++) {
    bm_image *tmp = bmimg_queue.front();
    bmimg_queue.pop();
    if(tmp) {
      bm_image_destroy(*tmp);
      free(tmp);
    }
  }
  queue_lock.unlock();
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

    }
  } else {
    cout << "packet contain audio data ,drop this packet" << endl;
    return -1;
    //decoded = pkt.size;
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

    /* Init the decoders, with reference counting */
    //av_dict_set(&opts, "refcounted_frames", "1", 0);

// pcie model
#ifndef SOC_MODE
    av_dict_set_int(&opts, "sophon_idx", g_dev_id, 0);
#endif

    /*frame buffer set*/
    //av_dict_set(&opts, "extra_frame_buffer_num", "20", 0);

    /*set tcp*/
    //av_dict_set(&opts, "rtsp_transport", "tcp", 0);

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
*ffmpeg decode
*/
void decode_thread(int thread_id) {
  cout << "** ffmpeg decode thread started **" << endl;

  AVCodecContext *video_dec_ctx = NULL;
  AVStream *video_stream = NULL;
  int video_stream_idx = -1;
  AVFrame *frame = NULL;
  AVPacket pkt;
  AVFormatContext *fmt_ctx = NULL;
  int ret = 0, got_frame;
  av_register_all( );
  unsigned long count = 0;
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

  cout << " ** ffmpeg decode thread:"
       << " ** ffmpeg video width:" << video_dec_ctx->width
       << " height:" << video_dec_ctx->height << endl;
  /* dump input information to stderr */
  av_dump_format(fmt_ctx, 0, src_file.c_str(), 0);

  /* initialize packet, set data to NULL, let the demuxer fill it */
  av_init_packet(&pkt);
  pkt.data = NULL;
  pkt.size = 0;

  frame = av_frame_alloc();
  if (!frame) {
    fprintf(stderr, "Could not allocate frame\n");
    return;
  }

  /* read frames from the file */
  while (av_read_frame(fmt_ctx, &pkt) >= 0) {
    AVPacket orig_pkt = pkt;

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

      bm_image *uncmp_bmimg = (bm_image *) malloc(sizeof(bm_image));
      bm_image_from_frame(g_bm_handle, *frame, *uncmp_bmimg);
      push_data(uncmp_bmimg);
      count++;
    } while (pkt.size > 0);
    av_packet_unref(&orig_pkt);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (count >= loops) break;
  }

  //cout << "** ffmpeg_decode exited !!" << endl;
  avformat_close_input(&fmt_ctx);
  avcodec_free_context(&video_dec_ctx);
}

/*
*detect thread
*/
void detect_thread(int thread_id) {

  vector<bm_image> input_img_bmcv;
  vector<vector<ObjRect>> detections;
  unsigned long count = 0;
  //init net
  SSD net(g_bm_handle, g_dev_id, bmodel_file);
  bm_image *uncmp_bmimg =  nullptr;
  while(1) {
    uncmp_bmimg = pop_data();
    if (uncmp_bmimg == nullptr) {
      //cout << "read bm_image is null from queue" << endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    input_img_bmcv.clear();
    detections.clear();
    //inference
    input_img_bmcv.push_back(*uncmp_bmimg);
    net.preForward(input_img_bmcv);

    // do inference
    net.forward();

    net.postForward(input_img_bmcv, detections);
    count++;

    net.dumpImg(input_img_bmcv[0], detections[0]);

    // this is necessary,when avframe is compressed format
    bm_image_destroy(*uncmp_bmimg);
    free(uncmp_bmimg);
    //end of inference
    if (count >= loops) break;
  }
}

int main(int argc, char **argv) {

  if(argc != 5) {
    cout << "USAGE:" << endl;
    cout << "  " << argv[0] << " <video url> <bmodel path> <frames> <dev id>" << endl;
    exit(1);
  }

  thread h_decode_thread;
  thread h_detect_thread;

  src_file = argv[1];
  if (!fs::exists(src_file)) {
      cout << "** Cannot find valid video file." << endl;
      exit(1);
  }
   cout << "** video file:" << src_file << endl;

  bmodel_file = argv[2];
  if (!fs::exists(bmodel_file)) {
    cout << "** Cannot find valid model file." << endl;
    exit(1);
  }
  // cout << "** bmodel file:" << bmodel_file << endl;

  loops = stoul(string(argv[3]), nullptr, 0);
  if (loops < 1) {
    std::cout << "test loop must large 0" << std::endl; 
    exit(1);
  }

  // dev id anomaly detection
  std::string dev_str = argv[4];
  std::stringstream checkdevid(dev_str);
  double t;
  if (!(checkdevid >> t)) {
    std::cout << "Is not a valid dev ID: " << dev_str << std::endl;
    exit(1);
  }
  g_dev_id = std::stoi(dev_str);
  cout << "** chip id:" << g_dev_id << endl; int max_dev_id = 0;
    bm_dev_getcount(&max_dev_id);
    if (g_dev_id >= max_dev_id) {
        std::cout << "ERROR: Input device id="<< g_dev_id
                  << " exceeds the maximum number " << max_dev_id << std::endl;
        exit(-1);
    }

  int ret = bm_dev_request(&g_bm_handle, g_dev_id);
  if (ret) {
    cout << "** failed to request device [" << g_dev_id << "] - " << ret << endl;
    exit(1);
  }

  h_decode_thread = thread(decode_thread, 0);
  h_detect_thread = thread(detect_thread, 0);
  h_decode_thread.join();
  h_detect_thread.join();

  bm_dev_free(g_bm_handle);
  cout << "main function end " << endl;
  return 0;
}

