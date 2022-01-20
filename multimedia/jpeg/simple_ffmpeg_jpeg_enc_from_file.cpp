
#define __STDC_CONSTANT_MACROS

#ifdef __cplusplus
extern "C" {
#endif

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavutil/file.h>
#include <libavutil/pixfmt.h>

#ifdef __cplusplus
}
#endif

#include <iostream>

using namespace std;

typedef struct {
    uint8_t* start;
    int      size;
    int      pos;
} bs_buffer_t;


static int writeJPEG(AVFrame* pFrame, int iIndex, string filename);


static int writeJPEG(AVFrame* pFrame, int iIndex, string filename)
{
    AVCodec         *pCodec  = nullptr;
    AVCodecContext  *enc_ctx = nullptr;
    AVDictionary    *dict    = nullptr;
    int ret = 0;

    string out_str = "new-" + filename + "-" + to_string(iIndex) + ".jpg";
    FILE *outfile = fopen(out_str.c_str(), "wb");

    /* Find HW JPEG encoder: jpeg_bm */
    pCodec = avcodec_find_encoder_by_name("jpeg_bm");
    if( !pCodec ) {
        cerr << "Codec jpeg_bm not found." << endl;
        return -1;
    }

    enc_ctx = avcodec_alloc_context3(pCodec);
    if (enc_ctx == NULL) {
        cerr << "Could not allocate video codec context!" << endl;
        return AVERROR(ENOMEM);
    }

    enc_ctx->pix_fmt = AVPixelFormat(pFrame->format);
    enc_ctx->width   = pFrame->width;
    enc_ctx->height  = pFrame->height;
    enc_ctx->time_base = (AVRational){1, 25};
    enc_ctx->framerate = (AVRational){25, 1};

#ifndef SOC_MODE
    //pcie mode
    av_dict_set_int(&dict, "pcie_board_id", 0, 0);
    /*do not copy data back to x86, default is copy back*/
    //av_dict_set_int(&dict, "pcie_no_copyback", 1, 0);
#endif

    /* Set parameters for jpeg_bm decoder */

    /* The output of bm jpeg decoder is chroma-separated,for example, YUVJ420P */
    av_dict_set_int(&dict, "chroma_interleave", 0, 0);

    /* 0: the data stored in virtual memory(pFrame->data[0-2]) */
    /* 1: the data stored in continuous physical memory(pFrame->data[3-5]) */
    int64_t value = 0;
    av_dict_set_int(&dict, "is_dma_buffer", value, 0);
    /* Open jpeg_bm encoder */
    ret = avcodec_open2(enc_ctx, pCodec, &dict);
    if (ret < 0) {
        cerr << "Could not open codec." << endl;
        return ret;
    }

    AVPacket *pkt = av_packet_alloc();
    if (!pkt) {
        cerr << "av_packet_alloc failed" << endl;
        return AVERROR(ENOMEM);
    }

    ret = avcodec_send_frame(enc_ctx, pFrame);
    if (ret < 0) {
        cerr << "Error sending a frame for encoding" << endl;
        return ret;
    }

    while (ret >= 0) {
        ret = avcodec_receive_packet(enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return ret;
        else if (ret < 0) {
            cerr << "Error during encoding" << endl;
            return ret;
        }

        cout << "packet size=" << pkt->size << endl;
        fwrite(pkt->data, 1, pkt->size, outfile);
        av_packet_unref(pkt);
    }

    av_packet_free(&pkt);

    fclose(outfile);

    av_dict_free(&dict);
    avcodec_free_context(&enc_ctx);

    cout << "Encode Successful." << endl;

    return 0;
}


int main(int argc, char* argv[])
{
    AVFrame         *pFrame = nullptr;
    FILE    *infile;
    int      numBytes;
    int      aviobuf_size = 32*1024; // 32K
    uint8_t *bs_buffer = nullptr;
    int      bs_size;
    bs_buffer_t bs_obj = {0, 0, 0};
    int ret = 0;
    int loop = 0;
    int pic_width = 0;
    int pic_height = 0;

    if (argc != 4) {
        cerr << "Usage:"<< argv[0] << " <xxx.yuv>" << " <width> " << " <height> " << endl;
        return 0;
    }

    string input_name = argv[1];

    infile = fopen(input_name.c_str(), "rb+");
    if (infile == nullptr) {
        cerr << "open file1 failed"  << endl;
        goto Func_Exit;
    }

    fseek(infile, 0, SEEK_END);
    numBytes = ftell(infile);
    cout << "infile size: " << numBytes << endl;
    fseek(infile, 0, SEEK_SET);

    bs_buffer = (uint8_t *)av_malloc(numBytes);
    if (bs_buffer == nullptr) {
        cerr << "av malloc for bs buffer failed" << endl;
        goto Func_Exit;
    }

    fread(bs_buffer, sizeof(uint8_t), numBytes, infile);
    fclose(infile);
    infile = nullptr;
    pic_width = stoul(string(argv[2]), nullptr, 0);
    pic_height = stoul(string(argv[3]), nullptr, 0);
    cout << "width: " << pic_width << " height: " << pic_height << endl;

    /*j420_buffer = (uint8_t *)av_malloc(numBytes);
    I420ToJ420(bs_buffer, pic_width,
               bs_buffer + (pic_width * pic_height), pic_width/2,
               bs_buffer + (pic_width * pic_height), pic_width/2,
               j420_buffer, pic_width,
               j420_buffer + (pic_width * pic_height), pic_width/2,
               j420_buffer + (pic_width * pic_height), pic_width/2,
               pic_width, pic_height);*/

    av_register_all();

    /* The bitstream buffer size (KB) */
    pFrame = av_frame_alloc();
    pFrame->data[0] = bs_buffer;
    pFrame->data[1] = bs_buffer + (pic_width * pic_height);
    pFrame->data[2] = bs_buffer + (pic_width * pic_height) + (pic_width * pic_height)/4;
    pFrame->linesize[0] = pic_width;
    pFrame->linesize[1] = pic_width/2;
    pFrame->linesize[2] = pic_width/2;
    pFrame->width = pic_width;
    pFrame->height = pic_height;
    pFrame->format = AV_PIX_FMT_YUVJ420P;
    if (pFrame == nullptr) {
        cerr << "av frame malloc failed" << endl;
        goto Func_Exit;
    }

    //convert AVFrame to JPEG
    cout << "pixel format: " << pFrame->format << endl;
    cout << "frame width : " << pFrame->width << endl;
    cout << "frame height: " << pFrame->height << endl;
    ret = writeJPEG(pFrame, loop, "yuv-jpeg");

    Func_Exit:

    if (pFrame) {
        av_frame_free(&pFrame);
    }

    if (infile) {
        fclose(infile);
    }

    if (bs_buffer) {
        av_free(bs_buffer);
    }
    return 0; // TODO
}
