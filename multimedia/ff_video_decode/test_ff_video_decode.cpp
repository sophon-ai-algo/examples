#include "ff_video_decode.h"

extern "C"{


#ifdef WIN32
#include <WinSock2.h>
#include <signal.h>
#include <conio.h>
#include <windows.h>
#include <time.h>
#else
#include <sys/time.h>
#include <csignal>
#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>
#endif
}

#define MAX_INST_NUM 256
#define PCIE_MODE_ARG_NUM 5
#define SOC_MODE_ARG_NUM 3
#define PCIE_CARD_NUM 1
#ifdef WIN32
HANDLE thread_id[MAX_INST_NUM];
#else
pthread_t thread_id[MAX_INST_NUM];
#endif
int quit_flag    = 0;


typedef struct MultiInstTest {
    const char *src_filename;
    const char *decoder_name;
    int sophon_idx;
    int output_format_mode;
    int codec_name_flag;
    int zero_copy;
    int pre_allocation_frame;
    int thread_index;
    unsigned int frame_nums[MAX_INST_NUM];
} THREAD_ARG;


static void usage(char *program_name)
{
#ifdef BM_PCIE_MODE
    av_log(NULL, AV_LOG_ERROR, "Usage: \n\t%s [yuv_format] [pre_allocation_frame] [codec_name] [sophon_idx] [zero_copy] [input_file/url] [input_file/url] ...\n", program_name);
    av_log(NULL, AV_LOG_ERROR, "\t[yuv_format]              0: non-compressed, 1: compressed.\n");
    av_log(NULL, AV_LOG_ERROR, "\t[pre_allocation_frame]    cache frames.\n");
    av_log(NULL, AV_LOG_ERROR, "\t[codec_name]              Unspecified: no, specified: decoder_name h264_bm/hevc_bm\n");
    av_log(NULL, AV_LOG_ERROR, "\t[sophon_idx]              sop chip idx\n");
    av_log(NULL, AV_LOG_ERROR, "\t[zero_copy]               0: Host memory available, 1: Host memory unavailable.\n");
    av_log(NULL, AV_LOG_ERROR, "\t[input_file/url] The number of input_file/url determines the number of threads.\n");
    av_log(NULL, AV_LOG_ERROR, "\t%s 0 1 no 0 1 ./example0.mp4 ./example1.mp4 ./example2.mp4\n", program_name);

#else
    av_log(NULL, AV_LOG_ERROR, "Usage: \n\t%s [yuv_format] [pre_allocation_frame] [codec_name] [input_file/url] [input_file/url] ...\n", program_name);
    av_log(NULL, AV_LOG_ERROR, "\t[yuv_format]              0: non-compressed, 1: compressed.\n");
    av_log(NULL, AV_LOG_ERROR, "\t[pre_allocation_frame]    cache frames.\n");
    av_log(NULL, AV_LOG_ERROR, "\t[codec_name]              Unspecified: no, specified: decoder_name h264_bm/hevc_bm\n");
    av_log(NULL, AV_LOG_ERROR, "\t[input_file/url] The number of input_file/url determines the number of threads.\n");
    av_log(NULL, AV_LOG_ERROR, "\t%s 0 1 no ./example0.mp4 ./example1.mp4 ./example2.mp4\n", program_name);

#endif
}




#ifdef _WIN32
static BOOL CtrlHandler(DWORD fdwCtrlType)
{
    switch (fdwCtrlType)
    {
    case CTRL_C_EVENT:
        quit_flag = 1;
        return(TRUE);
    default:
        return FALSE;
    }
}
#else
void handler(int sig)
{
    quit_flag = 1;
    printf("program will exited \n");
}
#endif

#if _WIN32
DWORD startOneInst(void* arg){
#else
void *startOneInst(void *arg){
#endif
    THREAD_ARG *thread_arg = (THREAD_ARG *)arg;
    const char *filename   = thread_arg->src_filename;
    int index              = thread_arg->thread_index;
    int sophon_index       = thread_arg->sophon_idx;
    int ret                = 0;
#ifdef WIN32
    clock_t tv1, tv2;
    long time;

#else
    struct timeval tv1, tv2;
    unsigned int time;
#endif
    unsigned int reconnet_times = 0;
    while(1){
        printf("reconnect stream[%s] times:%u.\n", filename, reconnet_times++);
        VideoDec_FFMPEG reader;
        ret = reader.openDec(filename,thread_arg->codec_name_flag,
                              thread_arg->decoder_name,thread_arg->output_format_mode,
                              thread_arg->pre_allocation_frame,sophon_index,
                              thread_arg->zero_copy);
        if(ret < 0 )
        {
            printf("open input media failed\n");
            if(quit_flag)
                break;
#ifdef WIN32
            Sleep(30*1000);
#else
            usleep(30 * 1000 * 1000);
#endif
            continue;
        }



        AVFrame * frame = NULL;
        thread_arg->frame_nums[index] = 0;
        //frame = reader.grabFrame();

#ifdef WIN32
        tv1 = clock();
#else
        gettimeofday(&tv1, NULL);
#endif
        while(!quit_flag){
            frame = reader.grabFrame();
            if(!frame)
            {
                printf("no frame ! \n");
                break;
            }
            if ((thread_arg->frame_nums[index]+1) % 300 == 0)
            {
#ifdef WIN32
                tv2 = clock();
                time = (tv2 - tv1);
#else
                gettimeofday(&tv2, NULL);
                time = (tv2.tv_sec - tv1.tv_sec) * 1000 + (tv2.tv_usec - tv1.tv_usec) / 1000;
#endif
                printf("%3dth thread process is %5.4f fps!\n", index ,(thread_arg->frame_nums[index] * 1000.0) / (float)time);
            }
            thread_arg->frame_nums[index]++;
        }
#ifdef WIN32
        tv2 = clock();
        time = (tv2 - tv1);
#else
        gettimeofday(&tv2, NULL);
        time = (tv2.tv_sec - tv1.tv_sec) * 1000 + (tv2.tv_usec - tv1.tv_usec) / 1000;
#endif
        printf("%3dth thread Decode %3d frame in total, avg: %5.4f, time: %ldms!\n", index ,thread_arg->frame_nums[index],(float)thread_arg->frame_nums[index] * 1000 / (float)time, time);

        reader.closeDec();
        thread_arg->frame_nums[index] = 0;

        if(quit_flag)
            break;
    }
    return NULL;
}





int main(int argc, char **argv)
{

    int arg_index = 0;

#ifdef BM_PCIE_MODE
    if(argc > MAX_INST_NUM + PCIE_MODE_ARG_NUM + 1){
        printf("The number of threads cannot exceed 256\n");
        return -1;
    }
    printf("This is pcie module\n");


    if(argc < 7){
        usage(argv[0]);
        return -1;
    }
#else
    if(argc > MAX_INST_NUM + SOC_MODE_ARG_NUM + 1){
        printf("The number of threads cannot exceed 256\n");
        return -1;
    }
    printf("This is soc module\n");

    if(argc < 5){
        usage(argv[0]);
        return -1;
    }
#endif


#ifdef __linux__
    signal(SIGINT, handler);
    signal(SIGTERM, handler);
#elif _WIN32
    SetConsoleCtrlHandler((PHANDLER_ROUTINE)CtrlHandler, TRUE);
#endif
    THREAD_ARG *thread_arg = (THREAD_ARG *)malloc(sizeof(THREAD_ARG));




    int is_compress = atoi(argv[ arg_index+1 ]);
    arg_index++;
    if(is_compress != 0 && is_compress != 1){
        usage(argv[0]);
        return -1;
    }
    if(is_compress == 0)
        thread_arg->output_format_mode = 100;
    else
        thread_arg->output_format_mode = 101;


    int is_pre_allocation_frame = atoi(argv[arg_index+1]);
    arg_index++;
    if(is_pre_allocation_frame < 0 && is_pre_allocation_frame > 64){
        usage(argv[0]);
        return -1;
    }
    thread_arg->pre_allocation_frame = is_pre_allocation_frame;


    if( !strcmp(argv[arg_index+1], "no") || !strcmp(argv[arg_index+1], "h264_bm") || !strcmp(argv[arg_index+1], "hevc_bm")){

        if(!strcmp(argv[arg_index+1], "no"))
                thread_arg->codec_name_flag = 0;

        if(!strcmp(argv[arg_index+1], "h264_bm") || !strcmp(argv[arg_index+1], "hevc_bm")){
            thread_arg->codec_name_flag = 1;
            thread_arg->decoder_name = argv[arg_index+1];
        }
    } else {
        usage(argv[0]);
        return -1;
    }
    arg_index++;

#ifdef BM_PCIE_MODE
    int sophon_idx = atoi(argv[++arg_index]);
    if(sophon_idx < 0 || sophon_idx > 120){
         usage(argv[0]);
         return -1;
    }
    thread_arg->sophon_idx = sophon_idx;


    int zero_copy = atoi(argv[++arg_index]);
    if(zero_copy != 0 && zero_copy != 1){
        usage(argv[0]);
        return -1;
    }
    thread_arg->zero_copy = zero_copy;
#endif


    //now arg_index value start with file parameters
    arg_index++;
    int td_index = 0;
    //Initialize multiple threads
    while( argc - arg_index ){
        thread_arg->thread_index = td_index;
        thread_arg->src_filename = argv[arg_index];
#ifdef WIN32
        thread_id[td_index] = CreateThread(NULL, 0, startOneInst, (void*)thread_arg, 0, NULL);
        Sleep(100);
#else
        pthread_create(&(thread_id[td_index]), NULL, startOneInst, thread_arg);
        usleep(100000);
#endif
        td_index++;
        arg_index++;
    }

    int idx = 0;
    for(idx = 0; idx < td_index; idx++){
#ifdef WIN32
        WaitForSingleObject(thread_id[idx], INFINITE);
#else
        pthread_join(thread_id[idx],NULL);
#endif
    }
    free(thread_arg);
    return 0;
}
