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
#include "yolox.hpp"


using namespace std;
namespace fs = boost::filesystem;

#define MAX_IMAGE_QUEUE         16
#define MAX_VIDEO_THREAD        16
#define MAX_DETECT_THREAD       16
#define READY_FRAME_RATE        2           //video decode every 2 frame save one
bool g_detect_finished = false;             //detect inference stop flag
int img_lost_count = 0;                     //丢弃帧数量
int img_process_count = 0;                  //处理帧的数量

std::mutex exit_lock;

bool get_exit_flage()
{
    bool return_flage = false;
    {
        std::lock_guard<std::mutex> thread_temp(exit_lock);
        return_flage = g_detect_finished;
    }
    return return_flage;
}

void set_exit_flage(bool value_temp)
{
    std::lock_guard<std::mutex> thread_temp(exit_lock);
    g_detect_finished = value_temp;
}

std::vector<string> getlabels(std::string label_file){
    std::vector<string> m_class_names;
    m_class_names.clear();
    std::ifstream ifs(label_file);
    if (ifs.is_open()) {
        std::string line;
        while(std::getline(ifs, line)) {
            m_class_names.push_back(line);
        }
    }
    return m_class_names;
}

DataQueue* pop_data(std::mutex *queue_lock, std::queue<DataQueue*> *data_queue)
{
    DataQueue* read = NULL;
    std::lock_guard<std::mutex> thread_temp(*queue_lock);
    if (data_queue->size() > 0){
        read = data_queue->front();
        data_queue->pop();
    }
    return read;
}

bool push_data(std::mutex *queue_lock, std::queue<DataQueue*> *data_queue, 
    DataQueue * element, uint32_t max_value, int &last_count)
{
    if(element == NULL){
        return false;
    }
    std::lock_guard<std::mutex> thread_temp(*queue_lock);
    if(data_queue->size() > max_value){
        DataQueue* read = data_queue->front();
        data_queue->pop();
        if(read->bmimage){
            bm_image_destroy(*read->bmimage);
            free(read->bmimage);
        }
        delete read;
        last_count++;
    }
    data_queue->push(element);
    return true;
}

void release_queue(std::mutex *queue_lock, std::queue<DataQueue*> *data_queue)
{
    std::lock_guard<std::mutex> thread_temp(*queue_lock);
    while(data_queue->size() > 0){
        DataQueue* read = data_queue->front();
        data_queue->pop();
        if(read->bmimage){
            bm_image_destroy(*read->bmimage);
            free(read->bmimage);
        }
        delete read;
    }
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

            if (frame->width != width || frame->height != height || frame->format != pix_fmt) {
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
        return -1;
        //decoded = pkt.size;
    }

    return decoded;
}

static int open_codec_context(int *stream_idx, AVCodecContext **dec_ctx, AVFormatContext *fmt_ctx, enum AVMediaType type,const char* video_name)
{
    int ret, stream_index;
    AVStream *st;
    AVCodec *dec = NULL;
    AVDictionary *opts = NULL;

    ret = av_find_best_stream(fmt_ctx, type, -1, -1, NULL, 0);
    if (ret < 0) {
        fprintf(stderr, "Could not find %s stream in input file '%s'\n",
            av_get_media_type_string(type), video_name);
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

        /*set compressed output*/
        av_dict_set(&opts, "output_format", "101", 0);

        av_dict_set(&opts, "stimeout", "20000000", 0);

        if ((ret = avcodec_open2(*dec_ctx, dec, &opts)) < 0) {
        fprintf(stderr, "Failed to open %s codec\n",
            av_get_media_type_string(type));
        return ret;
        }

        *stream_idx = stream_index;
    }

    return 0;
}

// bool g_yuv = false;               //false:BGR,true:yuv for inference input data format
void ffmpeg_thread(bm_handle_t g_bm_handle, int thread_id, const char* videoname, std::mutex *queue_lock, std::queue<DataQueue*> *data_queue,bool g_yuv)
{
    cout << "** ffmpeg decode thread [" << thread_id << "] started: " << " yuv/bgr:" << g_yuv << endl;
    AVCodecContext *video_dec_ctx = NULL;
    AVStream *video_stream = NULL;
    int video_stream_idx = -1;
    AVFrame *frame = NULL;
    AVPacket pkt;
    AVFormatContext *fmt_ctx = NULL;
    int ret = 0,got_frame;

    /* open input file, and allocate format context */
    if (avformat_open_input(&fmt_ctx,videoname, NULL, NULL) < 0) {
        cout << "Could not open source file " << videoname << endl;
        exit(1);
    }

    /* retrieve stream information */
    if (avformat_find_stream_info(fmt_ctx, NULL) < 0) {
        fprintf(stderr, "Could not find stream information\n");
        exit(1);
    }

    if (open_codec_context(&video_stream_idx, &video_dec_ctx, fmt_ctx, AVMEDIA_TYPE_VIDEO, videoname) >= 0) {
        video_stream = fmt_ctx->streams[video_stream_idx];
    }

    if (video_stream) {
        cout << "Demuxing video from file " << videoname << endl;
    } else {
        fprintf(stderr, "Could not find video stream in the input, aborting\n");
        exit(1); 
    }

    cout << " ** ffmpeg decode thread:" << thread_id
        << " ** ffmpeg video width:" << video_dec_ctx->width
        << " height:" << video_dec_ctx->height << endl;
    /* dump input information to stderr */
    av_dump_format(fmt_ctx, 0, videoname, 0);

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
        if (get_exit_flage()) {
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
        bm_image *uncmp_bmimg = (bm_image *) malloc(sizeof(bm_image));
        bm_image_from_frame(g_bm_handle, *frame, *uncmp_bmimg);
        DataQueue *img_task = new DataQueue();
        memset(img_task, 0, sizeof(DataQueue));
        img_task->id = thread_id;
        img_task->bmimage = uncmp_bmimg;
        img_task->num = num;
        bool ret = push_data(queue_lock, data_queue, img_task,
                MAX_IMAGE_QUEUE, img_lost_count);
        if(!ret) {
            cout << "** push image to queue error! id:" << thread_id << endl;
        }
        if (get_exit_flage()) break;
        
    } while (pkt.size > 0);
        av_packet_unref(&orig_pkt);
        //std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    av_frame_free(&frame);
    avformat_close_input(&fmt_ctx);
    avcodec_free_context(&video_dec_ctx);

    cout << "** ffmpeg_decode exited !! thread id:" << thread_id << endl;
}

void detect_thread(bm_handle_t g_bm_handle, int thread_id, const char* bmodel_file, std::mutex *queue_lock_input, std::queue<DataQueue*> *data_queue_input, 
    bool save_result,float detect_thr, float num_thr,std::vector<string> m_class_names)
{
    vector<int> strides;
    strides.push_back(8);
    strides.push_back(16);
    strides.push_back(32);
    YOLOX net(g_bm_handle, bmodel_file, strides);
    int batch_size =  net.getInputBatchSize();
    vector<DataQueue*> stream_cache;
    vector<bm_image> input_img_bmcv;
    cout << "** detect thread [" << thread_id << "] started: " << endl;
    int read_count = 0;
    int process_count = 0;
    while (true)
    {
        if (get_exit_flage()) break;
        DataQueue * stream_data = pop_data(queue_lock_input, data_queue_input);
        if (stream_data){
            stream_cache.push_back(stream_data);
            input_img_bmcv.push_back(*(stream_data->bmimage));
            read_count++;
        }else{
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); 
            continue;
        }
        vector<vector<ObjRect>> detections; 
        if(read_count == batch_size){
            read_count = 0;
            process_count++;
            double m_time_start = cv::getTickCount();
            if(net.preForward (input_img_bmcv) == 0){
                // do inference
                net.forward(detect_thr,num_thr);
                net.postForward (input_img_bmcv , detections);
            }
            printf("Detect thread[%d] Time use: %.2fms \n",thread_id,(cv::getTickCount()-m_time_start)/cv::getTickFrequency()*1000);
            if(save_result){
                for (size_t idx_resu = 0; idx_resu < detections.size(); idx_resu++)    {
                    for (size_t obj_idx = 0; obj_idx < detections[idx_resu].size(); obj_idx++)        {
                        bmcv_rect_t rect_temp;
                        rect_temp.start_x = detections[idx_resu][obj_idx].left;
                        rect_temp.start_y = detections[idx_resu][obj_idx].top;
                        rect_temp.crop_w = detections[idx_resu][obj_idx].width;
                        rect_temp.crop_h = detections[idx_resu][obj_idx].height;
                        bmcv_image_draw_rectangle(g_bm_handle,input_img_bmcv[idx_resu],1,&rect_temp,2,255,0,0);

                        bmcv_image_put_text(g_bm_handle,input_img_bmcv[idx_resu],m_class_names[detections[idx_resu][obj_idx].class_id].c_str(),
                            {rect_temp.start_x,rect_temp.start_y},{255,0,0},1.2,2);
                    }

                    if (!fs::exists("./results")) {
                        fs::create_directory("results");
                    }
                    
                    char save_name[256]={0};
                    sprintf(save_name,"results/detect-%d-batch-%d-%d.bmp",thread_id,process_count,idx_resu);
                    bm_image_write_to_bmp(input_img_bmcv[idx_resu], save_name);
                }
            }

            for (size_t i = 0; i < batch_size; i++)
            {
                DataQueue* read = stream_cache[i];
                if(read->bmimage){
                    bm_image_destroy(*read->bmimage);
                    free(read->bmimage);
                }
                delete read;
            }
            stream_cache.clear();
            input_img_bmcv.clear();
        }

    }
    
}

int main(int argc, char** argv)
{
    if (argc < 8){
        printf("  %s video <video url> <bmodel path> <name file> <detect threshold> <nms threshold> <device id> <video thread> <detect thread> <save:0/1>\n",argv[0]);
        exit(1);
    }

    bool is_video = false;
    if (strcmp(argv[1], "video") == 0)
        is_video = true;

    string image_name = argv[2];
    if (!fs::exists(image_name)) {
        printf("Cannot find input file: %s\n",image_name.c_str());
        exit(1);
    }

    fs::path image_file(image_name);
    string name_save = image_file.filename().string();

    printf("*****************************************************\n");
    printf("%s\n",image_name.c_str());
    printf("%s\n",name_save.c_str());
    printf("*****************************************************\n");

    string bmodel_file = argv[3];
    if (!fs::exists(bmodel_file)){
        printf("Can not find valid model file: %s\n",bmodel_file.c_str());
        exit(1);
    }

    string name_file = argv[4];
    if(!fs::exists(name_file)){
        printf("Can not find name file: %s\n",name_file.c_str());
        exit(1);
    }

    std::vector<string> m_class_names = getlabels(name_file);

    float threshold = atof(argv[5]);                 //0.25
    float nms_threshold = atof(argv[6]);             //0.45
    int device_id = atoi(argv[7]);
    int device_count = 0;
    bm_dev_getcount(&device_count);
    printf("device_count: %d\n",device_count);
    if(device_id >= device_count){
        printf("ERROR: Input device id=%d exceeds the maximum number %d\n",device_id,device_count);
        exit(1);
    }

    bm_handle_t bm_handle;
    bm_status_t ret = bm_dev_request(&bm_handle,device_id);
    if(ret != BM_SUCCESS){
        printf("bm_dev_request Fialed! ret = %d\n",ret);
        exit(1);
    }

    int video_thread_num = 1;
    int detect_thread_num = 1;
    bool save_image_flage = true;
    if(argc < 9){
        printf("Not set video thread number, use default 1!\n");
    }else{
        video_thread_num = atoi(argv[8]);
        if(video_thread_num > MAX_VIDEO_THREAD){
            printf("Video thread number exceeds maxinum value, use maxinum value: %d!\n",MAX_VIDEO_THREAD);
            video_thread_num = MAX_VIDEO_THREAD;
        }
    }
    if(argc < 10){
        printf("Not set detect thread number, use default 1!\n");
    }else{
        detect_thread_num = atoi(argv[9]);
        if(detect_thread_num > MAX_DETECT_THREAD){
            printf("Detect thread number exceeds maxinum value, use maxinum value: %d!\n",MAX_DETECT_THREAD);
            detect_thread_num = MAX_DETECT_THREAD;
        }
    }
    if (argc < 11)    {
        printf("Not set save result flage, use default True!\n");
    }else{
        save_image_flage = bool(atoi(argv[10]));
    }
    
    thread h_video_thread[MAX_VIDEO_THREAD]; //video decode thread handle 
    thread h_detect_thread[MAX_DETECT_THREAD]; //detect thread handle

    std::mutex mutex_image_data;
    std::queue<DataQueue*> image_data_queue;

    for (size_t i = 0; i < video_thread_num; i++)    {
        h_video_thread[i] = std::thread(ffmpeg_thread,bm_handle,i,image_name.c_str(), &mutex_image_data, &image_data_queue,0);
    }

    for (size_t i = 0; i < detect_thread_num; i++)    {
        h_detect_thread[i] = std::thread(detect_thread,bm_handle,i,bmodel_file.c_str(),&mutex_image_data, &image_data_queue,
            save_image_flage,threshold,nms_threshold,m_class_names);
    }
    

    for(int i = 0; i < video_thread_num; i++) {
      h_video_thread[i].join();
      cout << "video thread:" << i << " exit" << endl;
    }
    for(int i = 0; i < detect_thread_num; i++) {
      h_detect_thread[i].join();
      cout << "detect thread:" << i << " exit" << endl;
    }
   
    return 0;
}

