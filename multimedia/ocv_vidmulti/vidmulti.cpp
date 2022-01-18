#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/vpp.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <thread>
#include <random>
#include <stdlib.h>
#include <signal.h>
#include <queue>
#include <mutex>
#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <WinSock2.h>
#include <signal.h>
#include <conio.h>
#else
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>
#include <execinfo.h>
#endif

using std::vector;
using std::string;
using std::cout;
using std::cin;
using std::endl;
using namespace cv;

#define INTERVAL (1)
#define MAX_THREAD_NUM (512)
#define XUN_DEBUG 0
#define ONE_FRAME_CLOCK 30000
#define MAX_QUEUE_FRAME 4
//#define DSIPLAY_THREAD_FRAMERATE
//#define ENABLE_VPP_RESIZE
//#define ENABLE_VPP_CSC
std::default_random_engine generator;
std::uniform_int_distribution<int> distribution(0,2*ONE_FRAME_CLOCK);
string g_filename[MAX_THREAD_NUM];
int g_card[MAX_THREAD_NUM];
cv::VideoCapture g_cap[MAX_THREAD_NUM];
int g_thread_running[MAX_THREAD_NUM];

uint64_t count[MAX_THREAD_NUM];
uint64_t g_interval_80[MAX_THREAD_NUM];
uint64_t g_interval_160[MAX_THREAD_NUM];
uint64_t g_interval_400[MAX_THREAD_NUM];
uint64_t g_interval_1s[MAX_THREAD_NUM];
double fps[MAX_THREAD_NUM];
int g_exit_flag = 0;
int g_dump_flag[MAX_THREAD_NUM];

std::queue<Mat> g_image_queue[MAX_THREAD_NUM];
std::mutex g_video_lock[MAX_THREAD_NUM];

#ifdef WIN32
DWORD stat_pthread(void *arg);
DWORD video_download_pthread(void* arg);
#else
void *stat_pthread(void *arg);
void *video_download_pthread(void * arg);
int  kbhit(void);
#endif
void signal_handler(int signum);
void register_signal_handler(void);

#ifndef WIN32
int kbhit (void)
{
    struct timeval tv;
    fd_set rdfs;

    tv.tv_sec = 0;
    tv.tv_usec = 0;

    FD_ZERO(&rdfs);
    FD_SET (STDIN_FILENO, &rdfs);

    select(STDIN_FILENO+1, &rdfs, NULL, NULL, &tv);
    return FD_ISSET(STDIN_FILENO, &rdfs);
}
#endif

/* Obtain a backtrace and print it to stdout. */
void
print_trace (void)
{
#ifdef WIN32
#else
  void *array[10];
  char **strings;
  int size, i;

  size = backtrace (array, 10);
  strings = backtrace_symbols (array, size);
  if (strings != NULL)
  {
      printf ("Obtained %d stack frames.\n", size);
      for (i = 0; i < size; i++)
          printf ("%s\n", strings[i]);
  }

  free (strings);
#endif
}

void signal_handler(int signum) {
   // Release handle before crash in case we cannot reopen it again.
   g_exit_flag = 1;     // exit all threads
   int try_count = 100;
   cout << "signal " << signum << endl;

   signal(signum, SIG_IGN);

   print_trace();
   /* wait thread quit for 1s */
   while (try_count--){
     bool exit_all = true;
     for (int i = 0; i < MAX_THREAD_NUM; i++) {
         if(g_thread_running[i]){
             exit_all = false;
         }
     }
     if(exit_all){
         break;
     }

#ifdef WIN32
     Sleep(10);
#else
     usleep(10000);
#endif
   }

   for (int i = 0; i < MAX_THREAD_NUM; i++) {
      if (g_cap[i].isOpened()){
          cout << "thread " << i << "timed out! Force to release video capture!" << endl;
          g_cap[i].release();
      }
   }
   // Reset the signal handler as default
   signal(signum, SIG_DFL);

   // flush IO before exiting.
   cout.flush();

   _exit(signum);
}

void register_signal_handler(void) {
#ifdef WIN32
    typedef void (*SignalHandlerPoiner)(int);
    SignalHandlerPoiner _handler;
    _handler = signal(SIGABRT, signal_handler);
    _handler = signal(SIGFPE, signal_handler);
    _handler = signal(SIGILL, signal_handler);
    _handler = signal(SIGSEGV, signal_handler);
    _handler = signal(SIGINT, signal_handler);
#else
  struct sigaction action;
  memset(&action, 0, sizeof(action));
  sigfillset(&action.sa_mask);
  action.sa_handler = signal_handler;
  action.sa_flags = SA_RESTART;

  sigaction(SIGABRT, &action, nullptr);   // abort instruction
  sigaction(SIGBUS, &action, nullptr);    // Bus error
  sigaction(SIGFPE, &action, nullptr);    // float exception
  sigaction(SIGILL, &action, nullptr);    // illegal instruction
  sigaction(SIGSEGV, &action, nullptr);   // segement falut
  sigaction(SIGINT, &action, nullptr);   // ctrl_c signal
#endif
}

#ifdef WIN32
DWORD WINAPI stat_pthread(void* arg)
#else
void* stat_pthread(void *arg)
#endif
{
    int thread_num = *(int *)arg;
    uint64_t last_count[MAX_THREAD_NUM] = {0};
    uint64_t last_count_sum = 0;

    while(!g_exit_flag) {
#ifdef WIN32
        Sleep(INTERVAL * 1000);
#else
        sleep(INTERVAL);
#endif
#ifdef DSIPLAY_THREAD_FRAMERATE
        for (int i = 0; i < thread_num; i++) {
            if (i == 0)
                printf("ID[%d], FRM[%10lld], FPS[%2.2f]/[%2.2f] | ",
                    i, (long long)count[i],((double)(count[i]-last_count[i]))/INTERVAL, fps[i]);
            else
                printf("[%d] ,[%10lld], [%2.2f],[%2.2f] | ",
                    i, (long long)count[i], ((double)(count[i]-last_count[i]))/INTERVAL, fps[i]);
            last_count[i] = count[i];
        }
#else
        uint64_t count_sum = 0;
        for (int i = 0; i < thread_num; i++)
          count_sum += count[i];
        printf("thread %d, frame %lld, fps %2.2f", thread_num, count_sum, ((double)(count_sum-last_count_sum))/INTERVAL);
        last_count_sum = count_sum;
#endif
        printf("\r");
        fflush(stdout);
    }
    printf("\n[ID],[frame nums], [rate],  [invl>80ms],  [invl>160ms], [invl>400ms], [invl>1s]\n");
    for (int i = 0; i < thread_num; i++) {
        printf("[%d], [%10lld], [%2.2f], [%10llu], [%10llu], [%10llu], [%10llu]\n",
            i, (long long)count[i], fps[i], g_interval_80[i], g_interval_160[i], g_interval_400[i], g_interval_1s[i]);
        last_count[i] = count[i];
    }
    fflush(stdout);

    cout << "Stat thread exit!" << endl;
    return NULL;
}

#ifdef WIN32
DWORD WINAPI image_process_pthread(void * arg)
#else
void *image_process_pthread(void * arg)
#endif
{
    int id = *(int *)arg;
#ifdef WIN32
    clock_t tv1, tv2;
#else
    struct timeval tv1, tv2;
#endif
    Mat m;
    Mat rgb_m, resize_m, resize_m1, resize_m2;
    AVFrame *ff = av::create(600, 800, g_card[id]);
    resize_m.create(ff, g_card[id]);

    resize_m1.create(600, 800, CV_8UC3, g_card[id]);
    resize_m2.create(360, 640, CV_8UC3, g_card[id]);

    g_video_lock[id].lock();
    g_thread_running[id] = g_thread_running[id] | 0x2;
    g_video_lock[id].unlock();

#ifdef WIN32
    tv2 = clock();
#else
    gettimeofday(&tv2, NULL);
#endif

    while(!g_exit_flag){
        while(g_image_queue[id].empty()){
#ifdef WIN32
            Sleep(1);
#else
            usleep(1000);
#endif
            if (g_exit_flag) break;
        }
        if (g_exit_flag) break;

        g_video_lock[id].lock();
        m = g_image_queue[id].front();
        g_image_queue[id].pop();
        g_video_lock[id].unlock();

        /* resize & csc test */
#ifdef ENABLE_VPP_RESIZE
        bmcv::resize(m, resize_m, false);
        //bmcv::resize(m, resize_m1, false);
        //bmcv::resize(m, resize_m2, false);
#endif
#ifdef ENABLE_VPP_CSC
        {
            std::vector<Rect> vrt;
            std::vector<Size> vsz;
            std::vector<Mat> out;
            Rect rt;
            Size sz;

            rt.x = rt.y = 0;
            rt.width = sz.width = m.cols;
            rt.height = sz.height = m.rows;
            vrt.push_back(rt);
            vsz.push_back(sz);
            out.push_back(rgb_m);

            CV_Assert(BM_SUCCESS == cv::bmcv::convert(m, vrt, vsz, out, false));

            vrt.pop_back();
            vsz.pop_back();
            out.pop_back();
        }
#endif

        count[id]++;
#ifdef WIN32
        tv1 = clock();
#else
        gettimeofday(&tv1, NULL);
#endif

        if (count[id] % 100 == 0)
        {

#ifdef WIN32
            int t = (int)(tv1 - tv2);
            fps[id] = (double)count[id] * 1000.0 / t;
#else
            fps[id] = (double)count[id] * 1000.0 / ((tv1.tv_sec - tv2.tv_sec) * 1000.0 + (tv1.tv_usec - tv2.tv_usec) / 1000.0);
#endif
        }

        if (fps[id] > 1000.0)
        {
            cout << "illegal fps happens";
            break;
        }
    }

    g_video_lock[id].lock();
    g_thread_running[id] = g_thread_running[id] & ~0x2;
    g_video_lock[id].unlock();
    printf("image thread %d exit!", id);
    return NULL;
}

#ifdef WIN32
DWORD WINAPI video_download_pthread(void* arg)
#else
void *video_download_pthread(void * arg)
#endif
{
    int id = *(int *)arg;
#ifdef WIN32
    clock_t tv1, tv2, tv0;
#else
    struct timeval tv1, tv2, tv0;
#endif

    cv::VideoCapture *cap = &g_cap[id];
    g_video_lock[id].lock();
    g_thread_running[id] = g_thread_running[id] | 0x1;
    g_video_lock[id].unlock();

    if (!cap->open(g_filename[id], cv::CAP_ANY, g_card[id])) {
        cout << "open " << g_filename[id] << " failed!" << endl;
        return NULL;
    }

    if (!cap->isOpened()) {
        cout << "Failed to open camera!" << endl;
        return NULL;
    }

    cap->set(cv::CAP_PROP_OUTPUT_YUV, PROP_TRUE);

    int height = (int) cap->get(cv::CAP_PROP_FRAME_HEIGHT);
    int width  = (int) cap->get(cv::CAP_PROP_FRAME_WIDTH);

#ifndef HAVE_BMCV
    cv::Mat image(height, width, CV_8UC3);
#endif

#ifdef WIN32
    tv0 = clock();
#else
    gettimeofday(&tv0, NULL);
#endif
    tv2 = tv0;

    while (!g_exit_flag) {
		cv::Mat frame;

        cap->read(frame);

#ifndef HAVE_BMCV
        cv::vpp::toMat(frame, image);
#endif

        int64_t pts = (int64_t)cap->get(cv::CAP_PROP_TIMESTAMP);
        if (g_dump_flag[id]) {
            g_dump_flag[id] = 0;
#ifdef HAVE_BMCV
            imwrite("out-" + std::to_string(id)  + "-" + std::to_string(pts) + ".jpg", frame);
#else
            imwrite("out-" + std::to_string(id)  + "-" + std::to_string(pts) + ".jpg", image);
#endif
        }

#ifdef HAVE_BMCV
        if(frame.empty()) {
#else
        if(image.empty()) {
#endif
            if ((int)cap->get(cv::CAP_PROP_STATUS) == 2) { // eof
                cout << "file ends!" << endl;
                break;
            }

#ifdef WIN32
            Sleep(20);
#else
            usleep(20000);
#endif
            cout << "empty image got.." << endl;

            continue;
        }

        /* push image to queue */
        while ((g_image_queue[id].size() == MAX_QUEUE_FRAME) && (!g_exit_flag)){
#ifdef WIN32
            Sleep(1);
#else
            usleep(1000);
#endif
        }

        g_video_lock[id].lock();
        g_image_queue[id].push(frame);
        g_video_lock[id].unlock();

#ifdef WIN32
        tv1 = clock();
        int t = (int)(tv1 - tv0);
#else
        gettimeofday(&tv1, NULL);
        int t = (tv1.tv_sec - tv0.tv_sec) * 1000 + (tv1.tv_usec - tv0.tv_usec) / 1000;
#endif
        tv0 = tv1;

        if(t>1000) {
            g_interval_1s[id]++;
        }
        else if(t>400) {
            g_interval_400[id]++;
        }
        else if(t>160) {
            g_interval_160[id]++;
        }
        else if(t>80) {
            g_interval_80[id]++;
        }
   }

    if (cap->isOpened())
        cap->release();

    g_video_lock[id].lock();
    g_thread_running[id] = g_thread_running[id] & ~0x1;
    g_video_lock[id].unlock();

    cout << "Download thread " << id << " exit!" << endl;
    return NULL;
}

void usages(char **argv)
{
#ifdef USING_SOC
    cout << "Usage: " << argv[0] << " thread_num input_video input_video ..." << endl;
    cout << "For Examples: " << argv[0] << " 3 a.264 b.264 c.264" << endl;
#else
    cout << "Usage: " << argv[0] << " thread_num input_video [card] input_video [card] ..." << endl;
    cout << "For Examples: " << argv[0] << " 3 a.264 0 b.264 1 c.264 0" << endl;
#endif
}


int main(int argc, char **argv)
{
    memset(g_card, 0, sizeof(g_card));
    memset(count, 0, sizeof(count));

    if (argc < 3) {
        usages(argv);
        return -1;
    }

    int thread_num = atoi(argv[1]);
    if (argc < 2 + thread_num) {
        cout << "invalid: video_file number is less than thread_num " << endl;
        return -1;
    }
    if (thread_num > MAX_THREAD_NUM) {
        cout << "The thread num is too big. The max thread num is " << MAX_THREAD_NUM << endl;
	return -1;
    }

    if (argc - 2 == thread_num) { // only filename
        for (int i = 0; i < thread_num; i++)
            g_filename[i] = argv[2 + i];
    }
#ifndef USING_SOC
    else if (argc - 2 == thread_num * 2) { // filename + card
        for (int i = 0; i < thread_num; i++) {
            g_filename[i] = argv[2 + 2 * i];
            g_card[i] = atoi(argv[2 + 2 * i + 1]);
        }
    }
#endif
    else {
        usages(argv);
        return -1;
    }
    int test_mode = 0;
    char *p = getenv("test_mode");
    if (p) test_mode = atoi(p);

#ifdef WIN32
    HANDLE vc_thread[MAX_THREAD_NUM];
    DWORD thread_id[MAX_THREAD_NUM];
    HANDLE image_thread[MAX_THREAD_NUM];
    HANDLE stat_h;
#else
    pthread_t vc_thread[thread_num];
    int thread_id[thread_num];
    pthread_t image_thread[thread_num];
    pthread_t stat_h;
#endif
    srand((unsigned)time(NULL));

    /** Notice:
     * 1. In order to make sure signal_handler valid, video capture object should not be
     * processed in main thread. Otherwise crash of video capture will cause main thread
     * exit immediately, then signal_handler() has no chance to release resource of video
     * capture.
     * 2. register_signal_handler should be registered before pthread_create, otherwise
     * child thread will still to process signal in the signal_handler()
     */
    if (!test_mode)
        register_signal_handler();
    else
        cout << "now we are at test mode, signal_handler is not registered" << endl;

    for (int i = 0; i < thread_num; i++) {
        thread_id[i] = i;
#ifdef WIN32
        vc_thread[i] = CreateThread(NULL, 0, video_download_pthread, (void*)(&i), 0, NULL);
        int ret = 0;
#else
        int ret = pthread_create(&vc_thread[i], NULL, video_download_pthread, thread_id + i);
#endif
        if (ret != 0) {
            cout << "pthread create failed" << endl;
            return -1;
        }

#ifdef WIN32
        image_thread[i] = CreateThread(NULL, 0, image_process_pthread, (void*)(&i), 0, NULL);
#else
        ret = pthread_create(&image_thread[i], NULL, image_process_pthread, thread_id + i);
#endif
        if (ret != 0) {
            cout << "pthread create failed" << endl;
            return -1;
        }

#ifdef WIN32
        Sleep(1000);
#else
        sleep(1);
#endif
    }
#ifdef WIN32
    stat_h = CreateThread(NULL, 0, stat_pthread, (void*)&thread_num, 0, NULL);
#else
    pthread_create(&stat_h, NULL, stat_pthread, &thread_num);
#endif

    char cmd = 0;
    while(cmd != 'q')
    {
#ifdef WIN32
        if (_kbhit())
            cmd = _getch();
#else
        if (kbhit())
            cin >> cmd;
#endif
        else
#ifdef WIN32
            Sleep(1000);
#else
            sleep(1);
#endif

         if (cmd == 'c')
            for (int i = 0; i < thread_num; i++)
               g_dump_flag[i] = 1;
    }

    g_exit_flag = 1;

    for (int i = 0; i < thread_num; i++) {
#ifdef WIN32
        WaitForSingleObject(vc_thread[i], INFINITE);
        WaitForSingleObject(image_thread[i], INFINITE);
#else
        pthread_join(vc_thread[i], NULL);
        pthread_join(image_thread[i], NULL);
#endif
    }
#ifdef WIN32
        WaitForSingleObject(stat_h, INFINITE);
#else
    pthread_join(stat_h, NULL);
#endif
    return 0;
}
