#ifndef _VIDEO_STITCH_STITCH_H_
#define _VIDEO_STITCH_STITCH_H_

#include <thread>
#include <mutex>
#include "opencv2/opencv.hpp"
#include "bmutility.h"
#include "encoder.h"
#include "inference3.h"

class VideoStitchImpl : public bm::MediaDelegate<bm::FrameBaseInfo, bm::FrameInfo> {
    struct SFrameAndBbox{
        AVFrame* avframe;
        bm::NetOutputObjects objs;
        uint64_t seq;
        SFrameAndBbox()
          : avframe{nullptr},
            seq{0} {}
    };
public:
    VideoStitchImpl(int chann_start, int chann_count, std::shared_ptr<CVEncoder> &encoder);

    ~VideoStitchImpl();

    int stitch(std::vector<bm::FrameInfo>& frames, std::vector<bm::FrameBaseInfo>& output) override;

    int encode(std::vector<bm::FrameBaseInfo> &frames) override;

    inline bool setHandle(bm_handle_t handle) { m_handle = handle; }
private:
    int m_chan_start;
    int m_chan_count;
    uint64_t m_last_frame_ts;
    uint64_t m_last_sleep_time;
    std::map<int, SFrameAndBbox *> m_channels;
    uint64_t m_chan_mask;
    uint64_t m_chan_got_frame;
    std::shared_ptr<CVEncoder> m_encoder;
    bm_handle_t m_handle;
private:
    void fpsControl_(uint64_t msec_interval);

    void dataInput_(std::vector<bm::FrameInfo>& frames);
};
#endif // _VIDEO_STITCH_STITCH_H_