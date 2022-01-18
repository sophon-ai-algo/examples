#include <fstream>
#include <iostream>
#include <streambuf>
#include <stdio.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/syscall.h>
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
#ifdef WIN32
#define gettid GetCurrentThreadId
#define getpid GetCurrentProcessId
#else
#define gettid() syscall(SYS_gettid)
#endif
#define MAX_FILE_PATH    256
#define MAX_NUM_INSTANCE  12

typedef struct {
    void* new_mat;
    char yuvFileName[MAX_FILE_PATH];
    char bitstreamFileName[MAX_FILE_PATH];
    char refYuvFileName[MAX_FILE_PATH];
    char refBitStreamFileName[MAX_FILE_PATH];
    int  test_times;
    int  index_instance;
    int  codec_type;
} InputParam;

static int isOutJPG = 0;
static int card = 0;
static int g_flags = -1;

#ifdef WIN32
DWORD WINAPI FnEncodeTest(void * arg)
#else
static void * FnEncodeTest(void * arg)
#endif
{
    InputParam *  pParam = (InputParam *) arg;
    char * fileName = pParam->bitstreamFileName;
    int times = pParam->test_times;
    printf(" thread input : %s, test times = %d\n",fileName, times);
    vector<uchar> encoded;
    int count = 0;
    int pid = getpid();
    int tid = gettid();

#ifdef WIN32
    clock_t start;
    clock_t end;
#else
    struct timeval start;
    struct timeval end;
#endif

    ifstream in(fileName, std::ifstream::binary);
    string s((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
    in.close();
    std::vector<char> pic(s.c_str(), s.c_str() + s.length());
    cv::Mat image;

    cv::Mat save = cv::imread(fileName, g_flags, card);
    if(save.cols >=16 && save.rows >=16)
    {
#ifdef WIN32
        start = clock();
#else
        gettimeofday(&start, NULL);
#endif
        while(count++ < times )
        {
            cv::imencode(".jpg", save, encoded);
            if(count % 20 == 0)
            {
                printf("EncodeTest tid=%d (pid=%d), instance index =%d, times : %d\n", tid, pid, pParam->index_instance, count);
            }
        }
#ifdef WIN32
        end = clock();
        double t = (end - start) / 1000.0;
#else
        gettimeofday(&end, NULL);
        double t = (end.tv_sec + end.tv_usec / 1000.0 / 1000.0) - (start.tv_sec + start.tv_usec / 1000.0 / 1000.0);
#endif
        printf("Encoder%d time(second): %f\n", pParam->index_instance, t);
        if(isOutJPG){
            char str[50];
            int bufLen = encoded.size();
            if (bufLen){
                unsigned char* pYuvBuf = encoded.data();
                sprintf(str, "out%d_enc_pid%d.jpg", pParam->index_instance, pid);
                FILE * fclr = fopen(str, "wb");
                fwrite( pYuvBuf, 1, bufLen, fclr);
                fclose(fclr);
            }
        }
    }
    else{
        printf("decoding error,image size(width=%d,height=%d) is too small.\n",save.cols,save.rows);
    }
    return 0;
}


#ifdef WIN32
DWORD WINAPI FnDecodeTest(void * arg)
#else
static void * FnDecodeTest(void * arg)
#endif
{
    InputParam *  pParam = (InputParam *) arg;
    char * fileName = pParam->bitstreamFileName;
    int times = pParam->test_times;
    printf(" thread input : %s, test times = %d\n",fileName, times);
    vector<uchar> encoded;
    int count = 0;
    //int threadid = pthread_self();
    int pid = getpid();
    int tid = gettid();
#ifdef WIN32
    clock_t start;
    clock_t end;
#else
    struct timeval start;
    struct timeval end;
#endif
    ifstream in(fileName, std::ifstream::binary);
    string s((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
    in.close();
    std::vector<char> pic(s.c_str(), s.c_str() + s.length());
    cv::Mat image;

#ifdef WIN32
    start = clock();
#else
    gettimeofday(&start, NULL);
#endif
    while(count++ < times )
    {
        cv::imdecode(pic, g_flags, &image, card);
        if(count % 20 == 0)
        {
            printf("DecodeTest tid=%d (pid=%d), instance index =%d, times : %d\n", tid, pid, pParam->index_instance, count);
        }

    }
#ifdef WIN32
    end = clock();
    double t = (end - start) / 1000.0;
#else
    gettimeofday(&end, NULL);
    double t = (end.tv_sec + end.tv_usec / 1000.0 / 1000.0) - (start.tv_sec + start.tv_usec / 1000.0 / 1000.0);
#endif
    printf("Decoder%d time(second): %f\n", pParam->index_instance, t);

    if(isOutJPG){
        char str[50];
        sprintf(str, "out%d_dec_pid%d.jpg", pParam->index_instance,pid);
        cv::imwrite(str, image);
    }
    return 0;
}


int main(int argc, char *argv[])
{
    InputParam iParam;
    InputParam mulParam[MAX_NUM_INSTANCE];
#ifdef _WIN32
    DWORD thread_id[MAX_NUM_INSTANCE];
    HANDLE thread_handle[MAX_NUM_INSTANCE];
#else
    pthread_t thread_id[MAX_NUM_INSTANCE];
#endif
    void *ret[MAX_NUM_INSTANCE] = {NULL};
    int codec_type = 0;
    int num_threads = 0;

    if (argc < 6)
    {
        cout << "usage:" << endl;
        cout << "\t" << argv[0] << " <test type> <inputfile> <loop> <num_threads> <outjpg> [card]" << endl;
        cout << "params:" << endl;
        cout << "\t" << "<test type>:   1-only dec  2-only enc  3-mix" << endl;
        cout << "\t" << "<num_threads>: 1 <= num_threads <= " << MAX_NUM_INSTANCE << endl;
        cout << "\t" << "<outjpg>:      1 output jpg, 0 disable output jpg" << endl;
        exit(1);
    }

    if (argc == 7) {
        card = atoi(argv[6]);
    }

    num_threads = atoi(argv[4]);
    if( num_threads > MAX_NUM_INSTANCE )
    {
        num_threads = MAX_NUM_INSTANCE;
        printf("num_threads: %d   \n", num_threads);
    }
    if( num_threads <= 0 )
    {
        num_threads = 1;
        printf("num_threads: %d   \n", num_threads);
    }
    codec_type = atoi(argv[1]);
    printf("test type: %d   1-only dec  2-only enc  3-mix\n", codec_type);
    if( codec_type < 1 || codec_type > 3)
    {
        printf("codec type: %d error!\n", codec_type);
        return -1;
    }

    g_flags = cv::IMREAD_AVFRAME;

    memset(&iParam, 0x00, sizeof(InputParam));
    iParam.test_times = atoi(argv[3]);
    iParam.codec_type = codec_type;
    switch(codec_type)
    {
    case 1:
    case 2:
    case 3:
        printf("input bitstream file: %s\n", argv[2]);
        memcpy(iParam.bitstreamFileName, argv[2], strlen(argv[2]));
        break;
#if 0
    case 2:
        printf("input YUV file: %s file\n", argv[2]);
        memcpy(iParam.yuvFileName, argv[2], strlen(argv[2]));
#endif
        break;
    default:
        printf("error arg[1]\n");
        return -1;
        break;
    }

    isOutJPG = atoi(argv[5]);
    if (isOutJPG != 0 && isOutJPG != 1) {
        printf("\noutjpg must be 0 or 1\n");
        return -1;
    }

    for(int i = 0; i < num_threads; i++)
    {
        mulParam[i] = iParam;
        mulParam[i].index_instance = i;
        if(iParam.codec_type == 1)
        {
#ifdef WIN32
            thread_handle[i] = CreateThread(NULL, 0, FnDecodeTest, (void *)&mulParam[i], 0, &thread_id[i]);
#else
            pthread_create(&thread_id[i], NULL, FnDecodeTest, (void *)&mulParam[i]);
#endif
        }
        else if(iParam.codec_type == 2)
        {
#ifdef WIN32
            thread_handle[i] = CreateThread(NULL, 0, FnEncodeTest, (void *)&mulParam[i], 0, &thread_id[i]);
#else
            pthread_create(&thread_id[i], NULL, FnEncodeTest, (void *)&mulParam[i]);
#endif
        }
        else if(iParam.codec_type == 3)
        {
            if( i%2 == 0)
            {
#ifdef WIN32
            thread_handle[i] = CreateThread(NULL, 0, FnDecodeTest, (void *)&mulParam[i], 0, &thread_id[i]);
#else
                pthread_create(&thread_id[i], NULL, FnDecodeTest, (void *)&mulParam[i]);
#endif
            }
            else
            {
#ifdef WIN32
            thread_handle[i] = CreateThread(NULL, 0, FnEncodeTest, (void *)&mulParam[i], 0, &thread_id[i]);
#else
                pthread_create(&thread_id[i], NULL, FnEncodeTest, (void *)&mulParam[i]);
#endif
            }
        }
    }

    for(int i=0; i<num_threads; i++)
    {
#ifdef WIN32
        WaitForSingleObject(thread_handle[i], INFINITE);
#else
        pthread_join(thread_id[i], &ret[i]);
#endif
    }

    for(int i = 0; i < num_threads; i++)
    {
        if (ret[i] != 0)
            return 1;
    }
    return 0;
}
