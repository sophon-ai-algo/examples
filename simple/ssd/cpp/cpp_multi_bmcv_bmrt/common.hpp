/************************************************
* video decode -> image queue -> inference
************************************************/
#include <string>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <queue>
#include <vector>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/thread.hpp>

#ifdef __linux__
#include <sys/syscall.h>
#endif 

#include <libavformat/avformat.h>
#include "ssd.hpp"

#define MAX_SUPPORT_IMG_HEIGHT    (2160)
#define MIN_SUPPORT_IMG_HEIGHT    (90)
#define MAX_SUPPORT_IMG_WIDTH     (4096)
#define MIN_SUPPORT_IMG_WIDTH     (120)

#define MAX_IMAGE_QUEUE           (8)  //image data queue threshold value
#define MAX_L1OUT_QUEUE           (8)  //detect inference out data queue max value
#define MAX_L2OUT_QUEUE           (8)  //feature extraction inference out data queue max value

#define MAX_VIDEO_THREAD          (16) //video decode thread 
#define MAX_DETECT_THREAD         (16) //detect inference thread
#define MAX_FEATURE_EX_THREAD     (16) //feature extraction thread
#define MAX_JPG_THREAD            (16) //convert jpg thread

#define STAT_INTERVAL             (10) //stat display interval value,unit is second
#define READY_FRAME_RATE          (2)  //video decode every 5 frame save one

#ifdef __linux__
#define gettid() syscall(__NR_gettid)
#else
#define gettid() GetCurrentThreadId()
#endif

struct DataQueue {
  int id;                       //record thread id
  bm_image *bmimage;              //uncompressed bm_image
  AVFrame *avframe;              //ffmpeg avframe
  cv::Mat * mat;                 //opencv mat
  uint64_t num;                  //
  std::vector<ObjRect> rect_out;     //L1 out result
  bool done;                    //flag
  void *reserved;                //reserved for expansion data
};
