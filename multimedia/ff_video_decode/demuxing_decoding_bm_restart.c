/*
 * Copyright (c) 2018-2021, Bitmain Technologies Ltd
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file
 * Demuxing and decoding example.
 *
 * @example demuxing_decoding_bm_restart.c
 * Show how to use the libavformat and libavcodec API to demux and decode video data.
 */

#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include "libavutil/time.h"
#include "libavutil/threadmessage.h"
#include <stdatomic.h>

#include <stdlib.h>
#define HAVE_TERMIOS_H 1
#define HAVE_PTHREADS 1
#define HAVE_THREADS 1
#define HAVE_UNISTD_H 1
#if HAVE_TERMIOS_H
#include <fcntl.h>
#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <WinSock2.h>
#include <windows.h>
#pragma comment(lib,"ws2_32.lib")
#else
#include <sys/ioctl.h>
#include <sys/time.h>
#include <termios.h>
#endif
#elif HAVE_KBHIT
#include <conio.h>
#endif

#ifdef WIN32
#include <windows.h>
#else

#if HAVE_IO_H
#include <io.h>
#endif
#if HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <time.h>
#if HAVE_PTHREADS
#include <pthread.h>
#endif
#endif

#define MAX_INST_NUM 256
typedef struct MultiInstTest {
    AVFormatContext *fmt_ctx;
    AVCodecContext *video_dec_ctx, *audio_dec_ctx;
    int width, height;
    enum AVPixelFormat pix_fmt;
    AVStream *video_stream, *audio_stream;
    const char *src_filename;
    int video_stream_idx, audio_stream_idx;
    AVFrame *frame;
    AVPacket pkt;
    volatile int video_frame_count, latest_frame_count;

    /* Enable or disable frame reference counting. You are not supposed to support
     * both paths in your application but pick the one most appropriate to your
     * needs. Look for the use of refcount in this example to see what are the
     * differences of API usage between them. */
    int refcount;
    volatile int end_of;
    volatile int first_frame_flag;
    volatile int64_t start_time_dec, get_time_dec, last_time_read_pkt, short_start_time_dec;
    volatile double fps_dec, short_fps;
    int inst_idx;
    int first_pkt_flag;
    atomic_uint_fast64_t start_msg_pts, first_pkt_pts, input_pts, output_pts, ave_delay, max_delay, min_delay, total_max_delay, total_min_delay, total_delay;
    atomic_uint_fast32_t ave_num;
    int sophon_idx;
    int target_fps;
} MultiInstTest;

MultiInstTest inst[MAX_INST_NUM] = {0};
int inst_num = 0;
char codec_name[255];
int codec_name_flag = 0;
AVRational pts_to_ms = {1, 1000};
int delayms = 0;
int zero_copy = 1;

static int decode_packet(MultiInstTest *test_inst, int *got_frame, int cached)
{
    int ret = 0;
    int decoded = test_inst->pkt.size;

    *got_frame = 0;

    if (test_inst->pkt.stream_index == test_inst->video_stream_idx) {
        av_log(test_inst->video_dec_ctx, AV_LOG_TRACE, "inst: %d", test_inst->inst_idx);

        /* decode video frame */
        ret = avcodec_decode_video2(test_inst->video_dec_ctx, test_inst->frame, got_frame, &test_inst->pkt);
        if (ret < 0) {
            fprintf(stderr, "Error decoding video frame (%s)\n", av_err2str(ret));
            return ret;
        }

        if (*got_frame) {
            if(test_inst->first_frame_flag == 0) {
                test_inst->first_frame_flag = 1;
                test_inst->start_time_dec = av_gettime();
                test_inst->short_start_time_dec = test_inst->start_time_dec;
                test_inst->get_time_dec = test_inst->start_time_dec;
                test_inst->latest_frame_count = 1;
            }
            test_inst->video_frame_count++;

            if(test_inst->video_frame_count % 100 == 0) {
                test_inst->get_time_dec = av_gettime();
                if(test_inst->get_time_dec>test_inst->start_time_dec)
                    test_inst->fps_dec = (double)test_inst->video_frame_count/((double)(test_inst->get_time_dec - test_inst->start_time_dec)/(1000*1000));
            }
            if(test_inst->target_fps > 0 && test_inst->fps_dec > test_inst->target_fps + 1) {
                //the instances too fast make it slowly.
                av_usleep(2*1000);
            }
            if(delayms > 0)
                av_usleep((rand()%(2*delayms))*1000);
        }
    } else if (test_inst->pkt.stream_index == test_inst->audio_stream_idx) {
        decoded = test_inst->pkt.size;
    }

    /* If we use frame reference counting, we own the data and need
     * to de-reference it when we don't use it anymore */
    if (*got_frame/* && test_inst->refcount*/) {
        int64_t in_pts, out_pts;
#ifdef WIN32
        clock_t tv;
        tv = clock();
        if (tv != 0){
            test_inst->output_pts = (double) tv - test_inst->start_msg_pts;
        } else {
            test_inst->output_pts = 0;
        }
#else
        struct timeval tv;
        in_pts = test_inst->pkt.pts;
        out_pts = test_inst->frame->pts;
        if(gettimeofday(&tv, NULL) == 0) {
            test_inst->output_pts = (int64_t)tv.tv_sec*1000 + tv.tv_usec/1000 - test_inst->start_msg_pts;
        } else {
            test_inst->output_pts = 0;
        }
#endif

        test_inst->input_pts = av_rescale_q(out_pts - test_inst->first_pkt_pts, test_inst->video_stream->time_base, pts_to_ms);
        test_inst->total_delay = test_inst->output_pts - test_inst->input_pts;
        test_inst->ave_delay = in_pts - out_pts;
        test_inst->ave_delay = av_rescale_q(test_inst->ave_delay, test_inst->video_stream->time_base, pts_to_ms);

        av_frame_unref(test_inst->frame);
    }

    return decoded;
}
static AVCodec *find_codec_or_die(const char *name, enum AVMediaType type, int encoder)
{
    const AVCodecDescriptor *desc;
    const char *codec_string = encoder ? "encoder" : "decoder";
    AVCodec *codec;

    codec = encoder ?
        avcodec_find_encoder_by_name(name) :
        avcodec_find_decoder_by_name(name);

    if (!codec && (desc = avcodec_descriptor_get_by_name(name))) {
        codec = encoder ? avcodec_find_encoder(desc->id) :
                          avcodec_find_decoder(desc->id);
        if (codec)
            av_log(NULL, AV_LOG_VERBOSE, "Matched %s '%s' for codec '%s'.\n",
                   codec_string, codec->name, desc->name);
    }

    if (!codec) {
        av_log(NULL, AV_LOG_FATAL, "Unknown %s '%s'\n", codec_string, name);
        exit(1);
    }
    if (codec->type != type) {
        av_log(NULL, AV_LOG_FATAL, "Invalid %s type '%s'\n", codec_string, name);
        exit(1);
    }
    return codec;
}
int pic_mode = 0;
static int open_codec_context(int *stream_idx,
                              AVCodecContext **dec_ctx, AVFormatContext *fmt_ctx, enum AVMediaType type, int sophon_idx)
{
    int ret, stream_index;
    AVStream *st;
    AVCodec *dec = NULL;
    AVDictionary *opts = NULL;
    ret = av_find_best_stream(fmt_ctx, type, -1, -1, NULL, 0);
    if (ret < 0) {
        fprintf(stderr, "Could not find %s stream in input file\n",
                av_get_media_type_string(type));
        return ret;
    }
    stream_index = ret;
    st = fmt_ctx->streams[stream_index];

    /* find video decoder for the stream */
    if(codec_name_flag && type==AVMEDIA_TYPE_VIDEO)
        dec = find_codec_or_die(codec_name   , AVMEDIA_TYPE_VIDEO   , 0);
    else
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
    av_dict_set(&opts, "extra_frame_buffer_num", "5", 18);
    //av_dict_set(&opts, "frame_delay", "1", 0);
    //av_dict_set_int(&opts, "skip_non_idr", 1, 0);
    /* Init the decoders, with or without reference counting */
    if(pic_mode == 2)
        av_dict_set(&opts, "mode_bitstream", "2", 18);
    else if(pic_mode == 100)
        av_dict_set(&opts, "output_format", "100", 18);
    else if(pic_mode == 101)
        av_dict_set(&opts, "output_format", "101", 18);
    //av_dict_set(&opts, "extra_frame_buffer_num", "8", 18);
    //av_dict_set(&opts, "extra_data_flag", "8", 18);
    if(sophon_idx > 0) {
        av_dict_set_int(&opts, "sophon_idx", sophon_idx, 0);
        printf("sophon_idx: %d\n", sophon_idx);
    }
    if(zero_copy == 0) {
        av_dict_set_int(&opts, "zero_copy", zero_copy, 0);
    }
    av_dict_set_int(&opts, "perf", 1, 0);
    if ((ret = avcodec_open2(*dec_ctx, dec, &opts)) < 0) {
        fprintf(stderr, "Failed to open %s codec\n",
                av_get_media_type_string(type));
        return ret;
    }
    *stream_idx = stream_index;

    return 0;
}

static int get_input_packet(MultiInstTest *f, AVPacket *pkt)
{
    int ret = 0;
    ret = av_read_frame(f->fmt_ctx, pkt);
    return ret;
}

static int AVInterruptCallBackFun(void *param) {
    MultiInstTest *test_inst = (MultiInstTest *)param;
    unsigned int time_temp = 0;
    time_temp = av_gettime() - test_inst->last_time_read_pkt;
    if(time_temp > 3*1000*1000) {
        return 1;
    }
    return 0;
}

#ifdef WIN32
DWORD WINAPI start_one_inst(void* arg)
#else
static void *start_one_inst(void *arg)
#endif
{
    int ret = 0, got_frame;
    MultiInstTest *test_inst = (MultiInstTest *)arg;
    AVDictionary *dict = NULL;
    //av_dict_set(&dict, "buffer_size", "1024000", 0);
    //av_dict_set(&dict, "max_delay", "500000", 0);
    //av_dict_set(&dict, "stimeout", "2000000", 0);
    av_dict_set(&dict, "rtsp_flags", "prefer_tcp", 0);
    //av_dict_set(&dict, "analyzeduration", "10", 0);
    //av_dict_set(&dict, "probesize", "500", 0);
    //av_dict_set(&dict, "discardcorrupt", "0x0100", 0);
    //av_dict_set(&dict, "keep_rtsp_timestamp", "1", 0);
    while(1) {
#ifdef WIN32
        clock_t tv = clock();
        test_inst->start_msg_pts = (int64_t)tv;
#else
        struct timeval tv;
        gettimeofday(&tv, NULL);
        test_inst->start_msg_pts = (int64_t)tv.tv_sec*1000 + tv.tv_usec/1000;
#endif
        /* open input file, and allocate format context */
        if (avformat_open_input(&test_inst->fmt_ctx, test_inst->src_filename, NULL, &dict) < 0) {
            fprintf(stderr, "Could not open source file %s\n", test_inst->src_filename);
#ifdef WIN32
            Sleep(1);
#else
            usleep(1000*1000);
#endif
            continue;
        }

        /* retrieve stream information */
        if (avformat_find_stream_info(test_inst->fmt_ctx, NULL) < 0) {
            fprintf(stderr, "Could not find stream information\n");
            exit(1);
        }

        if (open_codec_context(&test_inst->video_stream_idx, &test_inst->video_dec_ctx, test_inst->fmt_ctx, AVMEDIA_TYPE_VIDEO, test_inst->sophon_idx) >= 0) {
            test_inst->video_stream = test_inst->fmt_ctx->streams[test_inst->video_stream_idx];

            /* allocate image where the decoded image will be put */
            test_inst->width = test_inst->video_dec_ctx->width;
            test_inst->height = test_inst->video_dec_ctx->height;
            if(codec_name_flag)
                test_inst->pix_fmt = AV_PIX_FMT_NV12;
            else
                test_inst->pix_fmt = test_inst->video_dec_ctx->pix_fmt;
        }

        if (open_codec_context(&test_inst->audio_stream_idx, &test_inst->audio_dec_ctx, test_inst->fmt_ctx, AVMEDIA_TYPE_AUDIO, test_inst->sophon_idx) >= 0) {
            test_inst->audio_stream = test_inst->fmt_ctx->streams[test_inst->audio_stream_idx];
        }

        /* dump input information to stderr */
        av_dump_format(test_inst->fmt_ctx, 0, test_inst->src_filename, 0);

        if (!test_inst->video_stream) {
            fprintf(stderr, "Could not find audio or video stream in the input, aborting\n");
            ret = 1;
            exit(1);
        }

        test_inst->frame = av_frame_alloc();
        if (!test_inst->frame) {
            fprintf(stderr, "Could not allocate frame\n");
            ret = AVERROR(ENOMEM);
            exit(1);
        }

        /* initialize packet, set data to NULL, let the demuxer fill it */
        av_init_packet(&test_inst->pkt);
        test_inst->pkt.data = NULL;
        test_inst->pkt.size = 0;
        test_inst->last_time_read_pkt = av_gettime();
        test_inst->fmt_ctx->interrupt_callback.callback = AVInterruptCallBackFun;
        test_inst->fmt_ctx->interrupt_callback.opaque   = test_inst;
        if (test_inst->video_stream)
            printf("Demuxing video from file '%s'\n", test_inst->src_filename);

        /* read frames from the file */
        while (1) {
            if(test_inst->end_of!=0) {
                break;
            }
            ret = get_input_packet(test_inst, &test_inst->pkt);

            if (ret == AVERROR(EAGAIN)) {
                if((av_gettime() - test_inst->last_time_read_pkt) > 1000*1000*60) {
                    break;
                }
                av_usleep(10000);
                continue;
            }
            else if(ret < 0)
                break;
            test_inst->last_time_read_pkt = av_gettime();
            AVPacket orig_pkt = test_inst->pkt;
            if((test_inst->first_pkt_flag == 0) && (test_inst->pkt.pts != AV_NOPTS_VALUE)) {
                test_inst->first_pkt_flag = 1;
                test_inst->first_pkt_pts = test_inst->pkt.pts;
                printf("first pkt pts: %ld\n", test_inst->first_pkt_pts);
                printf("base time: %ld, real time: %ld\n",
                       test_inst->fmt_ctx->start_time, test_inst->fmt_ctx->start_time_realtime);
            }
            do {
                ret = decode_packet(test_inst, &got_frame, 0);
                if (ret < 0)
                    break;
                test_inst->pkt.data += ret;
                test_inst->pkt.size -= ret;
            } while (test_inst->pkt.size > 0);
            av_packet_unref(&orig_pkt);
        }

        /* flush cached frames */
        test_inst->pkt.data = NULL;
        test_inst->pkt.size = 0;
        do {
            decode_packet(test_inst, &got_frame, 1);
        } while (got_frame);
        printf("Demuxing bitmain succeeded. inst index: %d\n", test_inst->inst_idx);

        if (test_inst->video_stream) {
            printf("Play the output video file with the command:\n"
                   "ffplay -f rawvideo -pix_fmt %s -video_size %dx%d\n",
                   av_get_pix_fmt_name(test_inst->pix_fmt), test_inst->width, test_inst->height);
        }
        printf("close the decoder....\n");
        avcodec_free_context(&test_inst->video_dec_ctx);
        avcodec_free_context(&test_inst->audio_dec_ctx);
        avformat_close_input(&test_inst->fmt_ctx);
        printf("close the frame....\n");
        av_frame_free(&test_inst->frame);
        printf("check the end_of....\n");
        if(test_inst->end_of!=0) {
            printf(" the end_of and break.......\n");
            break;
        }
    }
#ifdef WIN32
    return 0;
#else
    return NULL;
#endif
}
#if HAVE_PTHREADS
#ifdef WIN32
HANDLE thread_id[MAX_INST_NUM];
#else
pthread_t thread_id[MAX_INST_NUM];
#endif
#endif


/* read a key without blocking */
static int read_key(void)
{
    unsigned char ch;
#if HAVE_TERMIOS_H
    int n = 1;
#ifdef WIN32
    TIMEVAL tv;//设置超时等待时间
#else
    struct timeval tv;
#endif
    fd_set rfds;

    FD_ZERO(&rfds);
    FD_SET(0, &rfds);
    tv.tv_sec = 0;
    tv.tv_usec = 0;
    n = select(1, &rfds, NULL, NULL, &tv);
    if (n > 0) {
        n = read(0, &ch, 1);
        if (n == 1)
            return ch;

        return n;
    }
#elif HAVE_KBHIT
#    if HAVE_PEEKNAMEDPIPE
    static int is_pipe;
    static HANDLE input_handle;
    DWORD dw, nchars;
    if(!input_handle){
        input_handle = GetStdHandle(STD_INPUT_HANDLE);
        is_pipe = !GetConsoleMode(input_handle, &dw);
    }

    if (is_pipe) {
        /* When running under a GUI, you will end here. */
        if (!PeekNamedPipe(input_handle, NULL, 0, NULL, &nchars, NULL)) {
            // input pipe may have been closed by the program that ran ffmpeg
            return -1;
        }
        //Read it
        if(nchars != 0) {
            read(0, &ch, 1);
            return ch;
        }else{
            return -1;
        }
    }
#    endif
    if(kbhit())
        return(getch());
#endif
    return -1;
}

#ifdef __LOG4C__
#include "log4c.h"

static log4c_category_t *log_category = NULL;

static void custom_output(void* ptr, int level, const char* fmt,va_list vl)
{
    int a_priority = 0;
    switch(level) {
        case AV_LOG_QUIET:
            a_priority = LOG4C_PRIORITY_NOTSET;
            break;
        case AV_LOG_PANIC:
            a_priority = LOG4C_PRIORITY_NOTSET;
            break;
        case AV_LOG_FATAL:
            a_priority = LOG4C_PRIORITY_FATAL;
            break;
        case AV_LOG_ERROR:
            a_priority = LOG4C_PRIORITY_ERROR;
            break;
        case AV_LOG_WARNING:
            a_priority = LOG4C_PRIORITY_WARN;
            break;
        case AV_LOG_INFO:
            a_priority = LOG4C_PRIORITY_INFO;
            break;
        case AV_LOG_VERBOSE:
            a_priority = LOG4C_PRIORITY_NOTICE;
            break;
        case AV_LOG_DEBUG:
            a_priority = LOG4C_PRIORITY_DEBUG;
            break;
        case AV_LOG_TRACE:
            a_priority = LOG4C_PRIORITY_TRACE;
            break;
        default:
            a_priority = LOG4C_PRIORITY_NOTSET;
            break;
    }
    log4c_category_vlog(log_category, a_priority, fmt, vl);
}

#endif

static int usage(char **argv)
{
    fprintf(stderr, "usage: %s [-delay Xms] inst_num input_url [card] input_url [card] ...\n"
            "API example program to demo multi thread decoding.\n"
            "-delay Xms: delay X ms after decoder 1 frames for simulation testing.\n"
            "For Example:\n"
            "%s 3 in.mp4 in.mp4 in.mp4\n"
            "%s 3 in.mp4 1 in.mp4 1 in.mp4 1\n"
            "%s -delay 100 3 in.mp4 1 in.mp4 1 in.mp4 1\n"
            "\n", argv[0], argv[0], argv[0], argv[0]);
    return 0;
}

int main (int argc, char **argv)
{
    int ret = 0;
    int i=0;
    int argc_offset = 0;
    int sophon_idx = 0;
    int target_fps = 0;

    if (argc < 2) {
        usage(argv);
        exit(1);
    }

    /* TODO getopt */
    if (argc > 2 && !strcmp(argv[1], "-c")) {
        strcpy(codec_name, argv[2]);
        codec_name_flag = 1;
        argc_offset = 2;
        argc -= 2;
    }
    else if (argc > 2 && !strcmp(argv[1], "-sophon_idx")) {
        sophon_idx = atoi(argv[2]);
        printf("sophon_idx : %d\n", sophon_idx);
        argc_offset = 2;
        argc -= 2;
    }
    else if(argc > 2 && !strcmp(argv[1], "-p")) {
        pic_mode = 2;
        argc_offset = 1;
        argc -= 1;
    }
    else if(argc > 2 && !strcmp(argv[1], "-co")) {
        pic_mode = 101;
        argc_offset = 1;
        argc -= 1;
    }
    else if(argc > 2 && !strcmp(argv[1], "-ti")) {
        pic_mode = 100;
        argc_offset = 1;
        argc -= 1;
    }
    else if(argc > 2 && !strcmp(argv[1], "-inst_num")) {
        inst_num = atoi(argv[2]);
        printf("instance number : %d\n", inst_num);
        argc_offset = 2;
        argc -= 2;
    }
    else if(argc > 2 && !strcmp(argv[1], "-delay")) {
        delayms = atoi(argv[2]);
        printf("delay : %d ms\n", delayms);
        argc_offset = 2;
        argc -= 2;
    }
    else if(argc > 2 && !strcmp(argv[1], "-fps")) {
        target_fps = atoi(argv[2]);
        printf("target fps : %d ms\n", target_fps);
        argc_offset = 2;
        argc -= 2;
    }
    else if(argc > 2 && !strcmp(argv[1], "-zero_copy")) {
        zero_copy = atoi(argv[2]);
        printf("zero_copy : %d ms\n", zero_copy);
        argc_offset = 2;
        argc -= 2;
    }

    inst_num = atoi(argv[1+argc_offset]);
    argc_offset += 1;
    argc -= 1;

    if (((inst_num != argc-1) && (inst_num*2 != argc-1)) || argc > 2*MAX_INST_NUM + 1) {
        usage(argv);
        exit(1);
    }

#if 0
    if (argc == 5 && !strcmp(argv[1], "-refcount")) {
        refcount = 1;
        argv++;
    }
#endif

    if(inst_num > MAX_INST_NUM) {
        printf(".....exit, %d\n", inst_num);
        exit(1);
    }

#ifdef __LOG4C__
    log4c_init();
    log_category = log4c_category_get("ffmpeg");
    /*
    log4c_appender_t* myappender;
    myappender = log4c_appender_get("./mylog615.log");
    log4c_appender_set_type(myappender,log4c_appender_type_get("stream2"));
    log4c_category_set_appender(log_category, myappender);*/
    av_log_set_callback(custom_output);
    av_log_set_level(AV_LOG_INFO);
#endif

    if(inst_num == argc - 1) {
        for(i=1; i<argc; i++) {
            inst[i-1].src_filename = argv[i + argc_offset];
            inst[i-1].end_of = 0;
            inst[i-1].first_frame_flag = 0;
            inst[i-1].start_time_dec = 0;
            inst[i-1].get_time_dec = 0;
            inst[i-1].fps_dec = 0;
            inst[i-1].inst_idx = i-1;
            inst[i-1].sophon_idx = sophon_idx;
            inst[i-1].target_fps = target_fps;
        }
    } else {
        for(i=0; i<inst_num*2; i=i+2) {
            int j=i>>1;
            inst[j].src_filename = argv[i + argc_offset + 1];
            inst[j].end_of = 0;
            inst[j].first_frame_flag = 0;
            inst[j].start_time_dec = 0;
            inst[j].get_time_dec = 0;
            inst[j].fps_dec = 0;
            inst[j].inst_idx = j;
            inst[j].sophon_idx = atoi(argv[i + argc_offset + 2]);
            inst[j].target_fps = target_fps;
        }
    }
    av_log_set_level(AV_LOG_ERROR);

#if HAVE_PTHREADS
    for(i=0; i<inst_num; i++) {
#ifdef WIN32
        thread_id[i] = CreateThread(NULL, 0, start_one_inst, (void*)(&inst[i]), 0, NULL);
        int ret = 0;
#else
        pthread_create(&(thread_id[i]), NULL, start_one_inst, &(inst[i]));
#endif
    }
    int key;

    while(1) {
        double total_fps = 0, total_times = 0;
        unsigned long total_frames = 0;
        double inst_max_fps = 0.0, short_inst_max_fps = 0.0;
        double inst_min_fps = 10000.0, short_inst_min_fps = 10000.0;
        for(i=0; i<inst_num; i++) {
            //printf("[%d], [%10d], [%2.2f], {decoder delay:%ld ms}, {client pts: %ld ms}, {server pts: %ld ms}, {delay: %ld ms}\n", i, inst[i].video_frame_count, inst[i].fps_dec, inst[i].ave_delay, inst[i].input_pts, inst[i].output_pts, inst[i].total_delay);
            int64_t time_duration = inst[i].get_time_dec - inst[i].start_time_dec;
            double inst_times = ((double)time_duration/(1000*1000));
            time_duration = inst[i].get_time_dec - inst[i].short_start_time_dec;
            if(time_duration > 60*1000*1000) {
                inst[i].short_fps = (inst[i].video_frame_count - inst[i].latest_frame_count)*1000*1000.0/time_duration;
                inst[i].latest_frame_count = inst[i].video_frame_count;
                inst[i].short_start_time_dec = inst[i].get_time_dec;
            }

            printf("[inst: %2d], [frame count: %10u], [time: %10.1fs] [fps: %4.2f] [1Mfps: %4.2f]\n", i, inst[i].video_frame_count, inst_times, inst[i].fps_dec, inst[i].short_fps);
            inst[i].ave_delay = 0;
            total_fps += inst[i].fps_dec;
            total_times += inst_times;
            total_frames += inst[i].video_frame_count;
            if(inst[i].fps_dec > inst_max_fps)
                inst_max_fps = inst[i].fps_dec;
            if(inst[i].short_fps > short_inst_max_fps)
                short_inst_max_fps = inst[i].short_fps;
            if(inst[i].fps_dec < inst_min_fps)
                inst_min_fps = inst[i].fps_dec;
            if(inst[i].short_fps < short_inst_min_fps)
                short_inst_min_fps = inst[i].short_fps;
        }
        printf("total frame count: %10lu, ave time: %10.4fs, ave fps: %4.2f, max fps: %3.2f, min fps: %3.2f, max T fps: %3.2f, min T fps: %3.2f\n", total_frames, total_times/inst_num, total_fps, inst_max_fps, inst_min_fps, short_inst_max_fps, short_inst_min_fps);
        printf("\r");
        fflush(stdout);

        key = read_key();
        if (key == 'q') {
            for(i=0; i<inst_num; i++) {
                inst[i].end_of = 1;
            }
            for(i=0; i<inst_num; i++) {
#ifdef WIN32
                WaitForSingleObject(thread_id[i], INFINITE);
#else
                pthread_join(thread_id[i], NULL);
#endif
            }

            break;
        }

        av_usleep(5*1000*1000);
    }

#endif
#ifdef __LOG4C__
    log4c_fini();
#endif
    return ret < 0;
}

