//
// Created by yuan on 3/29/21.
//
#include "stream_decode.h"
#include <assert.h>
#include "bmutility_string.h"
#include "stream_pusher.h"


void test_push_jpeg(void)
{
    bm::FfmpegOutputer outputer;
}


int main(int argc, char *argv[])
{
    int ret = 0;
    std::string url = "/data/face_demo/sample/yanxi-1080p-2M.264";
    bm::StreamDecoder decoder(0);
    AVDictionary *opts = 0;
    av_dict_set_int(&opts, "sophon_idx", 0, 0);
    ret = decoder.open_stream(url, true, opts);
    assert(ret == 0);
    int frame_idx = 0;
    decoder.set_decoded_frame_callback([&frame_idx](const AVPacket* pkt, const AVFrame *frame) {
        printf("frame[%d] w=%d, h=%d\n", frame_idx++, frame->width, frame->height);

    });

    decoder.close_stream(true);
}