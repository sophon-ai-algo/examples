#include <iostream>
#include <assert.h>
#include "ddr_reduction.h"
#include "stream_demuxer.h"
#include "bmutility_timer.h"
#include "bmutility_image.h"
#include "opencv2/opencv.hpp"
#include "bmgui-lite/bmgui-lite.h"

#define SHOWVIDEO 0
class TestDecoderSequence: public bm::StreamDemuxerEvents
{
    std::shared_ptr<DDRReduction> ddrr;
    bm_handle_t m_handle;
    bm::TimerQueuePtr  m_tq;
    int total_seek_frame_num{0};
    bm::StatToolPtr m_pps_tool;
    bm::StatToolPtr m_fps_tool;
    int m_loop;

public:
    TestDecoderSequence(int loop) {
        ddrr=(DDRReduction::create(0, AV_CODEC_ID_H264));
        assert(ddrr != NULL);
        bm_dev_request(&m_handle, 0);
        m_tq = bm::TimerQueue::create();
        m_pps_tool = bm::StatTool::create();
        m_fps_tool = bm::StatTool::create();
        m_loop = loop;
    }
    ~TestDecoderSequence(){
        bm_dev_free(m_handle);
    }

    void get_frame_test() {
        int ret = 0;
        int64_t ref_id = -1, cur_id;
        AVFrame *frame = av_frame_alloc();
        uint64_t  timer_id;
        std::thread timer_thread([this]{
            m_tq->run_loop();
        });

        m_tq->create_timer(1000, [this] {
            DDRReductionStat stat;
            ddrr->get_stat(&stat);

            m_fps_tool->update(stat.total_decode_frame_num);
            m_pps_tool->update(stat.total_recv_packet_num);

            printf("packet rate=%f, framerate=%f, que_size=%d\n", m_pps_tool->getSpeed(), m_fps_tool->getSpeed(),
                      stat.queue_packet_num);
        }, 1, &timer_id);

        while(true) {
            DDRReductionStat stat;
            ddrr->get_stat(&stat);
            if (stat.queue_packet_num > 10) {
                break;
            }
        }


        while(m_loop > 0) {
            int got_frame = 0;
            ret = ddrr->seek_frame(ref_id, frame, &got_frame, &cur_id);
            if (ret < 0) {
                std::cout << "seek_frame() err=" << ret << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            if (got_frame) {
                m_loop--;
#if SHOWVIDEO
                bm_image image;
                bm::BMImage::from_avframe(m_handle, frame, image, false);
                cv::Mat img;
                cv::bmcv::toMAT(&image, img);
                bm::imshow("test", img);
                bm::waitkey(20);
                bm_image_destroy(image);
#endif
                av_frame_unref(frame);
                total_seek_frame_num++;
                ddrr->free_packet(ref_id);
                ref_id = cur_id;
            }
        }

        av_frame_free(&frame);

        m_tq->stop();
        timer_thread.join();
    }

    virtual int on_read_frame(AVPacket *pkt) {
        ddrr->put_packet(pkt, nullptr);
        return  0;
    }

    virtual void on_read_eof(AVPacket *pkt) {
        ddrr->put_packet(pkt, 0);
    }

    static void RunTest(int loop)
    {
        bm::StreamDemuxer demuxer;
        TestDecoderSequence tester(loop);
        demuxer.open_stream("rtsp://admin:hk123456@11.73.12.20", &tester);
        tester.get_frame_test();
        demuxer.close_stream(false);
        printf("%s() exit\n", __FUNCTION__);
    }
};



class TestDecoderRandom : public bm::StreamDemuxerEvents
{
    std::shared_ptr<DDRReduction> m_ddrr;
    bm::StreamDemuxer m_demuxer;
    bm_handle_t m_bmhandle;
    bm::TimerQueuePtr  m_tq;
    bm::StatToolPtr m_pps_tool;
    bm::StatToolPtr m_inner_fps_tool;
    bm::StatToolPtr m_random_fps_tool;
    int64_t m_total_random_frame_num{0};
public:
    TestDecoderRandom() {
        m_ddrr = DDRReduction::create(0, AV_CODEC_ID_H264);
        assert(m_ddrr != 0);
        m_tq=bm::TimerQueue::create();
        m_pps_tool = bm::StatTool::create();
        m_inner_fps_tool = bm::StatTool::create();
        m_random_fps_tool = bm::StatTool::create();
        bm_dev_request(&m_bmhandle, 0);

    }

    ~TestDecoderRandom(){
        bm_dev_free(m_bmhandle);
    }

    int on_read_frame(AVPacket *pkt) override
    {
        if (m_ddrr) m_ddrr->put_packet(pkt, 0);
        return 0;
    }

    void on_read_eof(AVPacket *pkt) override {
        if (m_ddrr) m_ddrr->put_packet(pkt, 0);
    }

    void random_seek_frame(int frame_num) {
        uint64_t  timer_id;
        std::thread timer_thread([this]{
            m_tq->run_loop();
        });

        m_tq->create_timer(1000, [this] {
            DDRReductionStat stat;
            m_ddrr->get_stat(&stat);

            m_inner_fps_tool->update(stat.total_decode_frame_num);
            m_random_fps_tool->update(m_total_random_frame_num);
            m_pps_tool->update(stat.total_recv_packet_num);

            printf("packet rate=%f, framerate=%f:%f, que_size=%d\n", m_pps_tool->getSpeed(), m_inner_fps_tool->getSpeed(),
                   m_random_fps_tool->getSpeed(), stat.queue_packet_num);
        }, 1, &timer_id);

        // start streaming
        m_demuxer.open_stream("rtsp://admin:hk123456@11.73.12.20", this);
        while(true) {
            DDRReductionStat stat;
            m_ddrr->get_stat(&stat);
            if (stat.queue_packet_num > frame_num) {
                printf("1000 frame, packet size=%d\n", stat.total_buffer_packet_bytes);
                break;
            }
        }

        // stop streaming
        m_demuxer.close_stream(false);

        int ret = 0;
        int loop = frame_num;
        AVFrame *frame = av_frame_alloc();

        while(loop > 0) {
            int got_frame = 0;
            int64_t ref_id = random() % frame_num;
            int64_t cur_id = 0;
            ret = m_ddrr->seek_frame(ref_id, frame, &got_frame, &cur_id);
            if (ret < 0) {
                std::cout << "seek_frame() err=" << ret << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            if (got_frame) {
                loop --;
                m_total_random_frame_num ++;
            }
#if SHOWVIDEO
            bm_image image;
            bm::BMImage::from_avframe(m_bmhandle, frame, image, false);
            cv::Mat img;
            cv::bmcv::toMAT(&image, img);
            bm::imshow("test", img);
            bm::waitkey(20);
            bm_image_destroy(image);
#endif
            av_frame_unref(frame);
        }

        m_tq->stop();
        timer_thread.join();

    }

    static void RunTest(int loop)
    {
        TestDecoderRandom app;
        app.random_seek_frame(loop);
    }
};



int main(int argc, char *argv[]) {
    int loop = 1000;
    if (argc > 1) {
        loop = atoi(argv[1]);
    }

    TestDecoderSequence::RunTest(loop);
    //TestDecoderRandom::RunTest(loop);

}