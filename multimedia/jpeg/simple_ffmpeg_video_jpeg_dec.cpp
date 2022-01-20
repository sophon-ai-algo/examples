//
// Created by yuan on 8/9/21.
//

#include <stdio.h>
#include <iostream>

#define __STDC_CONSTANT_MACROS

/**
 * In order to verify the bm_ffmpeg output format whether is NV12.
 */

#ifdef _WIN32
//Windows
extern "C"
{
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"
};
#else
//Linux...
#ifdef __cplusplus
extern "C"
{
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libavdevice/avdevice.h>
#include "getopt.h"

#ifdef __cplusplus
};
#endif
#endif

#if LIBAVCODEC_VERSION_MAJOR > 56

int decode_video2(AVCodecContext* dec_ctx, AVFrame *frame, int *got_picture, AVPacket* pkt)
{
    int ret;
    *got_picture = 0;
    ret = avcodec_send_packet(dec_ctx, pkt);
    if (ret == AVERROR_EOF) {
        ret = 0;
    }
    else if (ret < 0) {
        fprintf(stderr, "Error sending a packet for decoding, %s\n", av_err2str(ret));
        return -1;
    }

    while (ret >= 0) {
        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN)) {
            printf("need more data!\n");
            ret = 0;
            break;
        }else if (ret == AVERROR_EOF) {
            printf("File end!\n");
            avcodec_flush_buffers(dec_ctx);
            ret = 0;
            break;
        }
        else if (ret < 0) {
            fprintf(stderr, "Error during decoding\n");
            break;
        }
        printf("saving frame %3d\n", dec_ctx->frame_number);
        *got_picture += 1;
        break;
    }

    if (*got_picture > 1) {
        printf("got picture %d\n", *got_picture);
    }

    return ret;
}

#else

int decode_video2(AVCodecContext* dec_ctx, AVFrame *frame, int *got_picture, AVPacket* pkt)
{
    return avcodec_decode_video2(dec_ctx, frame, got_picture, pkt);
}

#endif

int main(int argc, char* argv[])
{
    AVFormatContext	*pFormatCtx;
    int				i, videoindex;
    AVCodecContext	*pCodecCtx;
    AVCodec			*pCodec;
    AVFrame	*pFrame;
    unsigned char *out_buffer;
    AVPacket *packet;
    int y_size;
    int ret, got_picture;
    int frame_idx= 0;
    int frame_num = 100;

    char *filepath=NULL;

    std::cout << "sizeof(int)=" << sizeof(int) << std::endl;

#ifdef SAVE_YUV_FILE
    FILE *fp_yuv=fopen("output.yuv","wb+");
#endif

    int ch;
    int option_index = 0;
    int cache_count = 0;
    struct option long_options[] =
            {
                    {0, 0, 0, 0},
            };

    while ((ch = getopt_long(argc, argv, "i:n:", long_options, &option_index)) != -1)
    {

        switch (ch)
        {
            case 0:
                if (optarg) {
                    //todo
                }break;
            case 'i':
                filepath=optarg;
                break;
            case 'n':
            {
                int num = atoi(optarg);
                if (num > 0){
                    frame_num = num;
                }
            }break;
        }
    }

    if (NULL == filepath) {
        printf("Please input file with -i option.\n");
        return -1;
    }


    av_register_all();
    avdevice_register_all();
    avformat_network_init();
    av_log_set_level(AV_LOG_ERROR);
    pFormatCtx = avformat_alloc_context();
    AVDictionary *opts = nullptr;
    AVInputFormat* input_fmt = nullptr;

    bool is_video_capture = false;
    if (is_video_capture) {
        input_fmt = av_find_input_format("video4linux2");
    }

    if(avformat_open_input(&pFormatCtx,filepath,input_fmt,&opts)!=0){
        printf("Couldn't open input stream, file=%s\n", filepath);
        return -1;
    }
    if(avformat_find_stream_info(pFormatCtx,NULL)<0){
        printf("Couldn't find stream information.\n");
        return -1;
    }
    videoindex=-1;
    for(i=0; i<pFormatCtx->nb_streams; i++)
        if(pFormatCtx->streams[i]->codec->codec_type==AVMEDIA_TYPE_VIDEO){
            videoindex=i;
            break;
        }

    if(videoindex==-1){
        printf("Didn't find a video stream.\n");
        return -1;
    }

    pCodecCtx=pFormatCtx->streams[videoindex]->codec;
    pCodec=avcodec_find_decoder(pCodecCtx->codec_id);
    if(pCodec==NULL){
        printf("Codec not found.\n");
        return -1;
    }

    //SOC, default is nv12, so don't need to set options.
    av_dict_free(&opts);
    av_dict_set_int(&opts, "cbcr_interleave", 0, 0);
    av_dict_set(&opts, "output_format", "101", 18);
    if(avcodec_open2(pCodecCtx, pCodec, &opts)<0){
        printf("Could not open codec.\n");
        return -1;
    }

    pFrame=av_frame_alloc();

    packet=(AVPacket *)av_malloc(sizeof(AVPacket));
    //Output Info-----------------------------
    printf("\n--------------- File Information ----------------\n");
    av_dump_format(pFormatCtx, 0,filepath, 0);

    while(1){

        if (frame_idx > frame_num){
            break;
        }

        if (av_read_frame(pFormatCtx, packet) < 0){
            break;
        }

        if(packet->stream_index==videoindex){
            ret = decode_video2(pCodecCtx, pFrame, &got_picture, packet);
            if(ret < 0){
                printf("Decode Error.\n");
                av_packet_unref(packet);
                return -1;
            }

            if (frame_idx==0) cache_count++;
            std::cout << "got_picture = " << got_picture << ", cache " << cache_count << std::endl;

            if(got_picture){
                frame_idx++;

                if (frame_idx == 1) {
                    printf("decode output format: %s\n", av_get_pix_fmt_name((enum AVPixelFormat)pFrame->format));
                }

#if DUMP_YUV_FILE
                if(pFrame->format == AV_PIX_FMT_YUV420P || pFrame->format == AV_PIX_FMT_YUVJ420P)
                {
                    fwrite(pFrame->data[0],1,y_size,fp_yuv);    //Y
                    fwrite(pFrame->data[1],1,y_size/4,fp_yuv);  //U
                    fwrite(pFrame->data[2],1,y_size/4,fp_yuv);  //V
                }else if(pFrame->format == AV_PIX_FMT_NV12){
                    fwrite(pFrame->data[0],1,y_size,fp_yuv);    //Y
                    fwrite(pFrame->data[1],1,y_size >> 1,fp_yuv); //UV
                }
#endif
                printf("Succeed to decode %d frame!\n", frame_idx);
            }

        }

        av_packet_unref(packet);

    }

    printf("\n");

    //flush decoder

    //FIX: Flush Frames remained in Codec
    while (1) {
        ret = decode_video2(pCodecCtx, pFrame, &got_picture, packet);
        if (ret < 0)
            break;
        if (!got_picture)
            break;

        frame_idx += got_picture;
#if DUMP_YUV_FILE
        y_size=pCodecCtx->width*pCodecCtx->height;

        if(pFrame->format == AV_PIX_FMT_YUV420P || pFrame->format == AV_PIX_FMT_YUVJ420P)
        {
            fwrite(pFrame->data[0],1,y_size,fp_yuv);    //Y
            fwrite(pFrame->data[1],1,y_size/4,fp_yuv);  //U
            fwrite(pFrame->data[2],1,y_size/4,fp_yuv);  //V
        }else if(pFrame->format == AV_PIX_FMT_NV12){
            fwrite(pFrame->data[0],1,y_size,fp_yuv);    //Y
            fwrite(pFrame->data[1],1,y_size >> 1,fp_yuv); //UV
        }else{
            printf("Flush Decoder: unknown frame format!\n");
        }
#endif
        printf("Flush Decoder: Succeed to decode %d frame!\n", frame_idx);
    }

    printf("\n");

#if DUMP_YUV_FILE
    fclose(fp_yuv);
#endif
    av_freep(&packet);
    av_frame_free(&pFrame);
    avcodec_close(pCodecCtx);
    avformat_close_input(&pFormatCtx);

    return 0;
}
