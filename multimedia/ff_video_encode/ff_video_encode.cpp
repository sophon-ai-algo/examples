/*
 * Copyright (c) 2019 BitMain
 *
 * @file
 *  example for video encode from BGR data,
 *  PS: i implement a C++ encoder it will be more easy to understand ffmpeg
 */
#include <iostream>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"
#include "libavformat/avformat.h"
#include "libavfilter/buffersink.h"
#include "libavfilter/buffersrc.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
}

#define STEP_ALIGNMENT 32
#ifdef WIN32
#define strcasecmp stricmp
#endif
class VideoEnc_FFMPEG
{
public:
    VideoEnc_FFMPEG();
    ~VideoEnc_FFMPEG();

    int  openEnc(const char* filename, int soc_idx, int codecId, int framerate,
                 int width, int height,int inputformat,int bitrate);
    void closeEnc();
    int  writeFrame(const uint8_t* data, int step, int width, int height);
    int  flush_encoder();

private:
    AVFormatContext * ofmt_ctx;
    AVCodecContext  * enc_ctx;
    AVFrame         * picture;
    AVFrame         * input_picture;
    AVStream        * out_stream;
    uint8_t         * aligned_input;

    int               frame_width;
    int               frame_height;

    int               frame_idx;

    AVCodec* find_hw_video_encoder(int codecId)
    {
        AVCodec *encoder = NULL;
        switch (codecId)
        {
        case AV_CODEC_ID_H264:
            encoder = avcodec_find_encoder_by_name("h264_bm");
            break;
        case AV_CODEC_ID_H265:
            encoder = avcodec_find_encoder_by_name("h265_bm");
            break;
        default:
            break;
        }
        return encoder;
    }
};

VideoEnc_FFMPEG::VideoEnc_FFMPEG()
{
    ofmt_ctx = NULL;
    picture= NULL;
    input_picture = NULL;
    out_stream = NULL;
    aligned_input = NULL;
    frame_width = 0;
    frame_height = 0;
    frame_idx = 0;
}

VideoEnc_FFMPEG::~VideoEnc_FFMPEG()
{
    printf("#######VideoEnc_FFMPEG exit \n");
}

int VideoEnc_FFMPEG::openEnc(const char* filename, int soc_idx, int codecId, int framerate,
                             int width, int height, int inputformat, int bitrate)
{
    int ret = 0;
    AVCodec *encoder;
    AVDictionary *dict = NULL;
    frame_idx = 0;
    frame_width = width;
    frame_height = height;

    avformat_alloc_output_context2(&ofmt_ctx, NULL, NULL, filename);
    if (!ofmt_ctx) {
        av_log(NULL, AV_LOG_ERROR, "Could not create output context\n");
        return AVERROR_UNKNOWN;
    }

    encoder = find_hw_video_encoder(codecId);
    if (!encoder) {
        av_log(NULL, AV_LOG_FATAL, "hardware video encoder not found\n");
        return AVERROR_INVALIDDATA;
    }

    enc_ctx = avcodec_alloc_context3(encoder);
    if (!enc_ctx) {
        av_log(NULL, AV_LOG_FATAL, "Failed to allocate the encoder context\n");
        return AVERROR(ENOMEM);
    }
    enc_ctx->codec_id = (AVCodecID)codecId;
    enc_ctx->width  = width;
    enc_ctx->height = height;
    enc_ctx->pix_fmt = (AVPixelFormat)inputformat;
    enc_ctx->bit_rate_tolerance = bitrate;
    enc_ctx->bit_rate = (int64_t)bitrate;
    enc_ctx->gop_size = 32;
    /* video time_base can be set to whatever is handy and supported by encoder */
    // enc_ctx->time_base = (AVRational){1, framerate};   // only for network stream frame rate
    // enc_ctx->framerate = (AVRational){framerate,1};
    enc_ctx->time_base.num = 1;
    enc_ctx->time_base.den = framerate;
    enc_ctx->framerate.num = framerate;
    enc_ctx->framerate.den = 1;

    av_log(NULL, AV_LOG_DEBUG, "enc_ctx->bit_rate = %ld\n", enc_ctx->bit_rate);

    out_stream = avformat_new_stream(ofmt_ctx, encoder);
    out_stream->time_base = enc_ctx->time_base;
    out_stream->avg_frame_rate = enc_ctx->framerate;
    out_stream->r_frame_rate = out_stream->avg_frame_rate;
    av_dict_set_int(&dict, "sophon_idx", soc_idx, 0);
    av_dict_set_int(&dict, "gop_preset", 8, 0);
    /* Use system memory */
    av_dict_set_int(&dict, "is_dma_buffer", 0, 0);

    /* Third parameter can be used to pass settings to encoder */
    ret = avcodec_open2(enc_ctx, encoder, &dict);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot open video encoder ");
        return ret;
    }
    ret = avcodec_parameters_from_context(out_stream->codecpar, enc_ctx);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Failed to copy encoder parameters to output stream ");
        return ret;
    }
    if (!(ofmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&ofmt_ctx->pb, filename, AVIO_FLAG_WRITE);
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Could not open output file '%s'", filename);
            return ret;
        }
    }
    /* init muxer, write output file header */
    ret = avformat_write_header(ofmt_ctx, NULL);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Error occurred when opening output file\n");
        return ret;
    }
    //av_dump_format(ofmt_ctx, 0, filename, 1);   // only for debug

    picture = av_frame_alloc();
    picture->format = enc_ctx->pix_fmt;
    picture->width = width;
    picture->height = height;

    return 0;
}

/* data is alligned with 32 */
int VideoEnc_FFMPEG::writeFrame(const uint8_t* data, int step, int width, int height)
{
    int ret = 0 ;
    int got_output = 0;
    if (step % STEP_ALIGNMENT != 0) {
        av_log(NULL, AV_LOG_ERROR, "input step must align with STEP_ALIGNMENT\n");
        return -1;
    }

    av_image_fill_arrays(picture->data, picture->linesize, (uint8_t *) data, enc_ctx->pix_fmt, width, height, 1);
    picture->linesize[0] = step;

    picture->pts = frame_idx;
    frame_idx++;

    av_log(NULL, AV_LOG_DEBUG, "Encoding frame\n");

    /* encode filtered frame */
    AVPacket enc_pkt;
    enc_pkt.data = NULL;
    enc_pkt.size = 0;
    av_init_packet(&enc_pkt);
    ret = avcodec_encode_video2(enc_ctx, &enc_pkt, picture, &got_output);
    if (ret < 0)
        return ret;
    if (got_output == 0) {
        av_log(NULL, AV_LOG_WARNING, "No output from encoder\n");
        return -1;
    }

    /* prepare packet for muxing */
    av_log(NULL, AV_LOG_DEBUG, "enc_pkt.pts=%ld, enc_pkt.dts=%ld\n",
           enc_pkt.pts, enc_pkt.dts);
    av_packet_rescale_ts(&enc_pkt, enc_ctx->time_base,out_stream->time_base);
    av_log(NULL, AV_LOG_DEBUG, "rescaled enc_pkt.pts=%ld, enc_pkt.dts=%ld\n",
           enc_pkt.pts,enc_pkt.dts);

    av_log(NULL, AV_LOG_DEBUG, "Muxing frame\n");

    /* mux encoded frame */
    ret = av_interleaved_write_frame(ofmt_ctx, &enc_pkt);
    return ret;
}

int  VideoEnc_FFMPEG::flush_encoder()
{
    int ret;
    int got_frame = 0;

    if (!(enc_ctx->codec->capabilities & AV_CODEC_CAP_DELAY))
        return 0;

    while (1) {
        av_log(NULL, AV_LOG_INFO, "Flushing video encoder\n");
        AVPacket enc_pkt;
        enc_pkt.data = NULL;
        enc_pkt.size = 0;
        av_init_packet(&enc_pkt);

        ret = avcodec_encode_video2(enc_ctx, &enc_pkt, NULL, &got_frame);
        if (ret < 0)
            return ret;

        if (!got_frame)
            break;

        /* prepare packet for muxing */
        av_log(NULL, AV_LOG_DEBUG, "enc_pkt.pts=%ld, enc_pkt.dts=%ld\n",
               enc_pkt.pts,enc_pkt.dts);
        av_packet_rescale_ts(&enc_pkt, enc_ctx->time_base,out_stream->time_base);
        av_log(NULL, AV_LOG_DEBUG, "rescaled enc_pkt.pts=%ld, enc_pkt.dts=%ld\n",
               enc_pkt.pts,enc_pkt.dts);

        /* mux encoded frame */
        av_log(NULL, AV_LOG_DEBUG, "Muxing frame\n");
        ret = av_interleaved_write_frame(ofmt_ctx, &enc_pkt);
        if (ret < 0)
            break;
    }

    return ret;
}

void VideoEnc_FFMPEG::closeEnc()
{
    flush_encoder();
    av_write_trailer(ofmt_ctx);

    av_free(picture);

    if (input_picture)
        av_free(input_picture);

    avcodec_free_context(&enc_ctx);

    if (ofmt_ctx && !(ofmt_ctx->oformat->flags & AVFMT_NOFILE))
        avio_closep(&ofmt_ctx->pb);
    avformat_free_context(ofmt_ctx);
}

static void usage(char* app_name);

int main(int argc, char **argv)
{
    int soc_idx = 0;
    int enc_id = AV_CODEC_ID_H264;
    int inputformat = AV_PIX_FMT_YUV420P;
    int framerate = 30;
    int ret;

    if (argc < 6 || argc > 10) {
        usage(argv[0]);
        return -1;
    }

    if (strcasecmp(argv[3], "H265") == 0 ||
        strcasecmp(argv[3], "HEVC") == 0)
        enc_id = AV_CODEC_ID_H265;
    else if (strcasecmp(argv[3], "H264") == 0 ||
             strcasecmp(argv[3], "AVC") == 0)
        enc_id = AV_CODEC_ID_H264;
    else {
        usage(argv[0]);
        return -1;
    }

    int width  = atoi(argv[4]);
    int height = atoi(argv[5]);

    if (argc >= 7) {
        if (strcasecmp(argv[6], "I420") == 0 ||
            strcasecmp(argv[6], "YUV") == 0) // deprecated
            inputformat = AV_PIX_FMT_YUV420P;
        else if (strcasecmp(argv[6], "NV12") == 0)
            inputformat = AV_PIX_FMT_NV12;
        else {
            usage(argv[0]);
            return -1;
        }
    }

    int bitrate = framerate*width*height/8;
    if (enc_id == AV_CODEC_ID_H265)
        bitrate = bitrate/2;
    if (argc >= 8) {
        int temp = atoi(argv[7]);
        if (temp >10 && temp < 10000)
            bitrate = temp*1000;
    }

    if (argc >= 9) {
        int temp = atoi(argv[8]);
        if (temp >10 && temp <= 60)
            framerate = temp;
    }

    if (argc == 10) {
        soc_idx = atoi(argv[9]);
        if (soc_idx < 0)
            soc_idx = 0;
    }

    int stride = (width + STEP_ALIGNMENT - 1) & ~(STEP_ALIGNMENT - 1);
    int aligned_input_size = stride * height*3/2;

    // TODO
    uint8_t *aligned_input = (uint8_t*)av_mallocz(aligned_input_size);
    if (aligned_input==NULL) {
        av_log(NULL, AV_LOG_ERROR, "av_mallocz failed\n");
        return -1;
    }

    FILE *in_file = fopen(argv[1], "rb");   //Input raw YUV data
    if (in_file == NULL) {
        fprintf(stderr, "Failed to open input file\n");
        usage(argv[0]);
        return -1;
    }

    bool isFileEnd = false;

    VideoEnc_FFMPEG writer;

    ret = writer.openEnc(argv[2], soc_idx, enc_id, framerate , width, height, inputformat, bitrate);
    if (ret !=0 ) {
        av_log(NULL, AV_LOG_ERROR,"writer.openEnc failed\n");
        return -1;
    }

    while(1) {
        for (int y = 0; y < height*3/2; y++) {
            ret = fread(aligned_input + y*stride, 1, width, in_file);
            if (ret < width) {
                if (ferror(in_file))
                    av_log(NULL, AV_LOG_ERROR, "Failed to read raw data!\n");
                else if (feof(in_file))
                    av_log(NULL, AV_LOG_INFO, "The end of file!\n");
                isFileEnd = true;
                break;
            }
        }

        if (isFileEnd)
            break;

        writer.writeFrame(aligned_input, stride, width, height);
    }

    writer.closeEnc();

    av_free(aligned_input);

    fclose(in_file);
    av_log(NULL, AV_LOG_INFO, "encode finish! \n");
    return 0;
}

static void usage(char* app_name)
{
    char usage_str[] =
    "Usage:\n\t%s <input file> <output file>  <encoder> <width> <height> <input pixel format> <bitrate(kbps)> <frame rate> <sophon device index>\n"
    "\t encoder             : H264(default), H265.\n"
    "\t input pixel format  : I420(YUV, default), NV12. YUV is deprecated.\n"
    "\t sophon device index : used in PCIE mode.\n"
    "For example:\n"
    "\t%s <input file> <output file> H264 width height I420 3000 30 2\n"
    "\t%s <input file> <output file> H264 width height I420 3000 30\n"
    "\t%s <input file> <output file> H265 width height I420\n"
    "\t%s <input file> <output file> H265 width height NV12\n"
    "\t%s <input file> <output file> H265 width height\n";

    av_log(NULL, AV_LOG_ERROR, usage_str,
           app_name, app_name, app_name, app_name, app_name, app_name);
}
