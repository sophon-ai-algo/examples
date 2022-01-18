#ifndef __FF_VIDEO_DECODE_H
#define __FF_VIDEO_DECODE_H

#include <iostream>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>

}

class VideoDec_FFMPEG
{
public:
    VideoDec_FFMPEG();
    ~VideoDec_FFMPEG();

    int openDec(const char* filename,int codec_name_flag,
                const char *coder_name,int output_format_mode = 100,
                int extra_frame_buffer_num = 5 ,
                int sophon_idx = 0, int pcie_no_copyback = 0);
    void closeDec();

    AVCodecParameters* getCodecPar();
    AVFrame * grabFrame();

private:
    AVFormatContext   *ifmt_ctx;
    AVCodec           *decoder;
    AVCodecContext    *video_dec_ctx ;
    AVCodecParameters *video_dec_par;

    int width;
    int height;
    int pix_fmt;

    int video_stream_idx;
    AVFrame *frame;
    AVPacket pkt;
    int refcount;

    AVCodec* findBmDecoder(AVCodecID dec_id, const char *name = "h264_bm",
                           int codec_name_flag = 0,
                           enum AVMediaType type = AVMEDIA_TYPE_VIDEO);


    int openCodecContext(int *stream_idx, AVCodecContext **dec_ctx,
                         AVFormatContext *fmt_ctx, enum AVMediaType type,
                         int codec_name_flag, const char *coder_name,
                         int output_format_mode = 100,
                         int extra_frame_buffer_num = 5,
                         int sophon_idx = 0,int pcie_no_copyback = 0);

};


#endif /*__FF_VIDEO_DECODE_H*/

