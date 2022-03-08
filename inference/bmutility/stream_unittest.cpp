//
// Created by yuan on 3/29/21.
//
#include "stream_decode.h"
#include <assert.h>
#include "bmutility_string.h"
#include "stream_pusher.h"
#include "bmutility_timer.h"


class DemuxerCounter : public bm::StreamDemuxerEvents {
protected:
    void on_avformat_opened(AVFormatContext *ifmt_ctx) override {}

    void on_avformat_closed() override {}

    int on_read_frame(AVPacket *pkt) override {
        //printf("frame [%d]\n", m_cnt++);
        m_cnt++;
        return 0;
    };

    void on_read_eof(AVPacket *pkt) override {}

public:
    int get_stat() { return m_cnt; }
private:
    int m_cnt{0};

};

void test_read_file()
{
    // 读文件有帧率控制
    // 需要将StreamDemuxer里段av_usleep部分注释掉
    bm::StreamDemuxer demuxer;
    DemuxerCounter    counter;
    std::string url = "/data/workspace/media/station-1080p-25fps-2000kbps.h264";
    demuxer.open_stream(url, &counter,true);
    bm::TimerQueuePtr tqp = bm::TimerQueue::create();
    int last_count = 0;

    uint64_t timer_id;
    tqp->create_timer(1000, [&](){
        int current_count = counter.get_stat();
        std::cout << "[" << bm::timeToString(time(0)) << "] read file fps = "
                  << std::setiosflags(std::ios::fixed) << std::setprecision(1) << current_count - last_count << std::endl;
        last_count = current_count;
    }, 1, &timer_id);

    tqp->run_loop();
    return;
}

void test_demux_and_decode() {
    static constexpr int kDecoderNum = 4;
    int ret = 0;

    std::string url = "/data/workspace/media/station-1080p-25fps-2000kbps.h264";
    int newCounter[kDecoderNum];
    int oldCounter[kDecoderNum];
    memset(newCounter, 0, sizeof(int) * kDecoderNum);
    memset(oldCounter, 0, sizeof(int) * kDecoderNum);

    for(int i = 0; i < kDecoderNum; ++i) {
        auto pDecoder = new bm::StreamDecoder(i);
        AVDictionary *opts = NULL;
        av_dict_set_int(&opts, "sophon_idx", 0, 0);
        av_dict_set(&opts, "output_format", "101", 18);
        av_dict_set(&opts, "extra_frame_buffer_num", "18", 0);
        pDecoder->set_avformat_opend_callback([&](AVFormatContext *ifmt) {
            std::cout << "stream " << i << " opened!";
        });
        pDecoder->open_stream(url, true, opts);
        av_dict_free(&opts);
        pDecoder->set_decoded_frame_callback([&](const AVPacket* pkt, const AVFrame *frame){
            newCounter[i]++;
        });
    }

    bm::TimerQueuePtr tqp = bm::TimerQueue::create();
    uint64_t timer_id;
    tqp->create_timer(1000, [&](){
        int fps = 0;
        for(int ii = 0; ii < kDecoderNum; ++ii) {
            fps += (newCounter[ii] - oldCounter[ii]);
        }
        std::cout << "[" << bm::timeToString(time(0)) << "] decode fps = "
                  << std::setiosflags(std::ios::fixed) << std::setprecision(1) << fps << std::endl;
        memcpy(oldCounter, newCounter, sizeof(int) * kDecoderNum);
    }, 1, &timer_id);

    tqp->run_loop();

}

int main(int argc, char *argv[])
{
#if 1
    test_demux_and_decode();
#else
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
#endif
    return 0;
}