#ifndef _VIDEO_STITCH_STITCH_H_
#define _VIDEO_STITCH_STITCH_H_

#include <thread>
#include <mutex>
#include "opencv2/opencv.hpp"
#include "bmutility.h"

template<typename T1>
class IVideoStitch {
public:
    virtual bool startStitch()  = 0;
    virtual bool stopStitch()   = 0;
    virtual bool dataInput(T1*, int) = 0;
    virtual ~IVideoStitch()     = default;
};

struct CvFrameBaseInfo {
    int chan_id;
    uint64_t seq;
    cv::Mat* mat;
    bool skip;
    int height, width;
    CvFrameBaseInfo() {
        chan_id = 0;
        seq = 0;
        mat = 0;
        skip = false;
        width = height = 0;
    }
    void sync(const CvFrameBaseInfo* chan) {
        if (chan->chan_id != this->chan_id) return;
        this->seq    = chan->seq;
        if (this->mat != nullptr) delete this->mat;
        this->mat    = chan->mat;
        this->width  = chan->width;
        this->height = chan->height;
    }
};

struct CvFrameInfo {
    std::vector<CvFrameBaseInfo> frames;
    std::vector<bm_tensor_t> input_tensors;
    std::vector<bm_tensor_t> output_tensors;
    std::vector<bm::NetOutputDatum> out_datums;
};

class VideoStitchImpl : public IVideoStitch<CvFrameBaseInfo> {
public:
    VideoStitchImpl(int chann_start, int chann_count);
    ~VideoStitchImpl();
    bool startStitch()                override;
    bool stopStitch()                 override;
    bool dataInput(CvFrameBaseInfo* frame, int count) override;
    bool go(std::vector<CvFrameBaseInfo>& frames);

private:
    bool                            m_stitch_running;
    int                             m_chan_start;
    int                             m_chan_count;
    uint64_t                        m_last_frame_ts;
    uint64_t                        m_last_sleep_time;
    std::map<int, CvFrameBaseInfo*> m_channels;
    std::mutex                      m_stitch_lock;
    std::shared_ptr<std::thread>    m_stitch_thread;
private:
    void fpsControl_(uint64_t msec_interval);
};

#endif // _VIDEO_STITCH_STITCH_H_