/**
  @file videocapture_basic.cpp
  @brief A very basic sample for using VideoCapture and VideoWriter
  @author PkLab.net
  @date Aug 24, 2016
*/
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <stdio.h>
#include <queue>
#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#include <signal.h>
#endif
#define IMAGE_MATQUEUE_NUM 25

using namespace cv;
using namespace std;

typedef struct  threadArg{
    const char  *inputUrl;
    const char  *codecType;
    int         frameNum;
    const char  *outputName;
    int         yuvEnable;
    int         deviceId;
    const char  *encodeParams;
    int         startWrite;
    int         fps;
    int         imageCols;
    int         imageRows;
    queue<Mat*> *imageQueue;
}THREAD_ARG;


//std::queue<Mat> g_image_queue;
std::mutex g_video_lock;

int exit_flag = 0;
#ifdef __linux__
void signal_handler(int signum){
    exit_flag = 1;
}
#elif _WIN32
static BOOL CtrlHandler(DWORD fdwCtrlType)
{
    switch (fdwCtrlType)
    {
    case CTRL_C_EVENT:
        exit_flag = 1;
        return(TRUE);
    default:
        return FALSE;
    }
}
#endif

#ifdef __linux__
void *videoWriteThread(void *arg){
#elif _WIN32
DWORD WINAPI videoWriteThread(void* arg){
#endif
    THREAD_ARG *threadPara = (THREAD_ARG *)(arg);
    FILE *fp_out           = NULL;
    char *out_buf          = NULL;
    int is_stream          = 0;
    int quit_times         = 0;
    VideoWriter            writer;
    Mat                    image;
    string outfile         = "";
    string encodeparms     = "";
    int64_t curframe_start = 0;
    int64_t curframe_end   = 0;
    Mat *toEncImage;
#ifdef __linux__
    struct timeval         tv;
#endif
    if(threadPara->encodeParams)
        encodeparms = threadPara->encodeParams;

    if((strcmp(threadPara->outputName,"NULL") != 0) && (strcmp(threadPara->outputName,"null") != 0))
        outfile = threadPara->outputName;

    if(strstr(threadPara->outputName,"rtmp://") || strstr(threadPara->outputName,"rtsp://"))
        is_stream = 1;

    if(strcmp(threadPara->codecType,"H264enc") ==0)
    {
        writer.open(outfile, VideoWriter::fourcc('H', '2', '6', '4'),
        threadPara->fps,
        Size(threadPara->imageCols, threadPara->imageRows),
        encodeparms,
        true,
        threadPara->deviceId);
    }
    else if(strcmp(threadPara->codecType,"H265enc") ==0)
    {
       writer.open(outfile, VideoWriter::fourcc('h', 'v', 'c', '1'),
        threadPara->fps,
        Size(threadPara->imageCols, threadPara->imageRows),
        encodeparms,
        true,
        threadPara->deviceId);
    }
    else if(strcmp(threadPara->codecType,"MPEG2enc") ==0)
    {
       writer.open(outfile, VideoWriter::fourcc('M', 'P', 'G', '2'),
        threadPara->fps,
        Size(threadPara->imageCols, threadPara->imageRows),
        true,
        threadPara->deviceId);
    }

    if(!writer.isOpened())
    {
#ifdef __linux__
        return (void *)-1;
#elif _WIN32
        return -1;
#endif
    }

    while(1){
        if(is_stream){
#ifdef __linux__
            gettimeofday(&tv, NULL);
            curframe_start= (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
#elif _WIN32
            FILETIME ft;
            GetSystemTimeAsFileTime(&ft);
            curframe_start = (int64_t)ft.dwHighDateTime << 32 | ft.dwLowDateTime;// unit is 100 us
            curframe_start = curframe_start / 10;
#endif
        }

        //if(threadPara->startWrite && !g_image_queue.empty()) {
        if(threadPara->startWrite && !threadPara->imageQueue->empty()) {
            if((strcmp(threadPara->outputName,"NULL") != 0) && (strcmp(threadPara->outputName,"null") != 0)){
                g_video_lock.lock();
                //writer.write(g_image_queue.front());
                toEncImage = threadPara->imageQueue->front();
                g_video_lock.unlock();
                writer.write(*toEncImage);

            }else{
                if(fp_out == NULL){
                        fp_out = fopen("pkt.dump","wb+");
                }
                if(out_buf == NULL){
                        out_buf = (char*)malloc(threadPara->imageCols * threadPara->imageRows * 4);
                }
                int out_buf_len = 0;
                //writer.write(g_image_queue.front(),out_buf,&out_buf_len);
                g_video_lock.lock();
                toEncImage = threadPara->imageQueue->front();
                g_video_lock.unlock();
                writer.write(*toEncImage,out_buf,&out_buf_len);
                if(out_buf_len > 0){
                    fwrite(out_buf,1,out_buf_len,fp_out);
                }
            }

            g_video_lock.lock();
            //g_image_queue.pop();
            threadPara->imageQueue->pop();
            delete toEncImage;
            g_video_lock.unlock();
            quit_times = 0;
        }else{
#ifdef __linux__
            usleep(2000);
#elif _WIN32
            Sleep(2);
#endif
            quit_times++;
        }
        //only Push video stream
        if(is_stream){
#ifdef __linux__
            gettimeofday(&tv, NULL);
            curframe_end= (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
#elif _WIN32
            FILETIME ft;
            GetSystemTimeAsFileTime(&ft);
            curframe_end = (int64_t)ft.dwHighDateTime << 32 | ft.dwLowDateTime;
            curframe_end = curframe_end / 10;
#endif
            if(curframe_end - curframe_start > 1000000 / threadPara->fps)
                continue;
#ifdef __linux__
            usleep((1000000 / threadPara->fps) - (curframe_end - curframe_start));
#elif _WIN32
            Sleep((1000 / threadPara->fps) - (curframe_end - curframe_start)/1000 - 1);
#endif
        }
        //if((exit_flag && g_image_queue.size() == 0) || quit_times >= 1000){//No bitstream exits after a delay of three seconds
        if((exit_flag && threadPara->imageQueue->size() == 0) || quit_times >= 300){//No bitstream exits after a delay of three seconds
            break;
        }
    }
    writer.release();
    if(fp_out != NULL){
        fclose(fp_out);
        fp_out = NULL;
    }

    if(out_buf != NULL){
        free(out_buf);
        out_buf = NULL;
    }
#ifdef __linux__
    return (void *)0;
#elif _WIN32
    return -1;
#endif
}

int main(int argc, char* argv[])
{
    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap;

#ifdef WIN32
    LARGE_INTEGER liPerfFreq={0};
    QueryPerformanceFrequency(&liPerfFreq);
    LARGE_INTEGER tv1 = {0};
    LARGE_INTEGER tv2 = {0};
    HANDLE threadId;
#else
    struct timeval tv1, tv2;
    pthread_t threadId;
#endif

    if (argc < 6){
        cout << "usage:  test encoder by HW(h.264/h.265) with difference contianer !!" << endl;
        cout << "\t" << argv[0] << " input code_type frame_num outputname yuv_enable [device_id] [encodeparams]" <<endl;
        cout << "\t" << "eg: " << argv[0] << " rtsp://admin:bitmain.com@192.168.1.14:554  H265enc  30 encoder_test265.ts 1 0 bitrate=1000" <<endl;
        cout << "params:" << endl;
        cout << "\t" << "<code_type>: H264enc is h264; H265enc is h265." << endl;
        cout << "\t" << "<outputname>: null or NULL output pkt.dump." << endl;
        cout << "\t" << "<yuv_enable>: 0 decode output bgr; 1 decode output yuv420." << endl;
        cout << "\t" << "<encodeparams>: gop=30:bitrate=800:gop_preset=2:mb_rc=1:delta_qp=3:min_qp=20:max_qp=40:push_stream=rtmp/rtsp." << endl;
        return -1;
    }
    THREAD_ARG *threadPara = (THREAD_ARG *)malloc(sizeof(THREAD_ARG));
    memset(threadPara,0,sizeof(THREAD_ARG));
#ifdef __linux__
    signal(SIGINT, signal_handler);
#elif _WIN32
    SetConsoleCtrlHandler((PHANDLER_ROUTINE)CtrlHandler, TRUE);
#endif
    if(!threadPara){
        return -1;
    }
    threadPara->imageQueue   = new queue<Mat*>;
    threadPara->outputName = argv[4];
    threadPara->codecType  = argv[2];
    threadPara->frameNum   = atoi(argv[3]);
    if ( strcmp(threadPara->codecType,"H264enc") ==0
      || strcmp(threadPara->codecType,"H265enc") == 0
      || strcmp(threadPara->codecType,"MPEG2enc") == 0)
    {
        threadPara->startWrite = 1;
    } else {
        if(threadPara->imageQueue){
            delete threadPara->imageQueue;
            threadPara->imageQueue = NULL;
        }
        if(threadPara){
            free(threadPara);
            threadPara = NULL;
        }
        return 0;
    }
    if (argc == 7) {threadPara->deviceId = atoi(argv[6]);}
    if (argc == 8) {threadPara->encodeParams = argv[7];}

    threadPara->yuvEnable = atoi(argv[5]);
    if ((threadPara->yuvEnable != 0) && (threadPara->yuvEnable != 1)) {
        cout << "yuv_enable param err." << endl;
        return -1;
    }
    // open the default camera using default API
    threadPara->inputUrl = argv[1];
    cap.open(threadPara->inputUrl, CAP_FFMPEG, threadPara->deviceId);
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    // Set Resamper
    cap.set(CAP_PROP_OUTPUT_SRC, 1.0);
    double out_sar = cap.get(CAP_PROP_OUTPUT_SRC);
    cout << "CAP_PROP_OUTPUT_SAR: " << out_sar << endl;

    if(threadPara->yuvEnable == 1){
        cap.set(cv::CAP_PROP_OUTPUT_YUV, PROP_TRUE);
    }

    // Set scalar size
    //int height = (int) cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    //int width  = (int) cap.get(cv::CAP_PROP_FRAME_WIDTH);
    cout << "orig CAP_PROP_FRAME_HEIGHT: " << (int) cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
    cout << "orig CAP_PROP_FRAME_WIDTH: " << (int) cap.get(CAP_PROP_FRAME_WIDTH) << endl;

    if(threadPara->startWrite)
    {
        Mat image;
        cap.read(image);
        //threadPara->imageQueue->push(image);
        threadPara->fps = cap.get(CAP_PROP_FPS);
        threadPara->imageCols = image.cols;
        threadPara->imageRows = image.rows;
#ifdef WIN32
        threadId = CreateThread(NULL, 0, videoWriteThread, threadPara, 0, NULL);
#else
        pthread_create(&threadId, NULL, videoWriteThread, threadPara);
#endif
    }

    //--- GRAB AND WRITE LOOP
#ifdef WIN32
    QueryPerformanceCounter(&tv1);
#else
    gettimeofday(&tv1, NULL);
#endif
    for (int i=0; i < threadPara->frameNum; i++)
    {
        if(exit_flag){
            break;
        }
        //Mat image;
        Mat *image = new Mat;
        // wait for a new frame from camera and store it into 'frame'
        cap.read(*image);

        // check if we succeeded
        //if (image.empty()) {
        if (image->empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            if ((int)cap.get(CAP_PROP_STATUS) == 2) {     // eof
                cout << "file ends!" << endl;
                cap.release();
                cap.open(threadPara->inputUrl, CAP_FFMPEG, threadPara->deviceId);
                if(threadPara->yuvEnable == 1){
                    cap.set(cv::CAP_PROP_OUTPUT_YUV, PROP_TRUE);
                }
                cout << "loop again " << endl;
            }
            continue;
        }
        g_video_lock.lock();
        //g_image_queue.push(image);
        threadPara->imageQueue->push(image);
        g_video_lock.unlock();
        if(threadPara->startWrite){
            //while(g_image_queue.size() >= IMAGE_MATQUEUE_NUM){
            while(threadPara->imageQueue->size() >= IMAGE_MATQUEUE_NUM){
#ifdef __linux__
                usleep(2000);
#elif _WIN32
                Sleep(2);
#endif
                if(exit_flag){
                    break;
                }
            }
        }
        if ((i+1) % 300 == 0)
        {
            unsigned int time;
#ifdef WIN32
            QueryPerformanceCounter(&tv2);
            time = ( ((tv2.QuadPart - tv1.QuadPart) * 1000)/liPerfFreq.QuadPart);
#else
            gettimeofday(&tv2, NULL);
            time = (tv2.tv_sec - tv1.tv_sec)*1000 + (tv2.tv_usec - tv1.tv_usec)/1000;
#endif
            printf("current process is %f fps!\n", (i * 1000.0) / (float)time);
        }
    }

#ifdef WIN32
    WaitForSingleObject(threadId, INFINITE);
#else
    pthread_join(threadId, NULL);
#endif
    cap.release();
    if(threadPara->imageQueue){
        delete threadPara->imageQueue;
        threadPara->imageQueue = NULL;
    }
    if(threadPara){
        free(threadPara);
        threadPara = NULL;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
