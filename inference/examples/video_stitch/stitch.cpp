#include "bmutility_timer.h"
#include "stitch.h"

VideoStitchImpl::VideoStitchImpl(int chann_start, int chann_count)
 : m_stitch_running {false}, 
   m_chan_start{chann_start},
   m_chan_count{chann_count},
   m_last_frame_ts{0},
   m_last_sleep_time{0} {
     for (int i = m_chan_start; i < m_chan_start + m_chan_count; i++) {
         CvFrameBaseInfo* f = new CvFrameBaseInfo;
         f->chan_id = i;
         m_channels.insert(std::make_pair(i, f));
     }
}

VideoStitchImpl::~VideoStitchImpl() {
    stopStitch();
    auto iter = m_channels.begin();
    while (iter != m_channels.end()) {
        if (iter->second != nullptr) {
            if(iter->second->mat != nullptr) delete iter->second->mat;
            delete iter->second;
        }
        ++iter;
    }
}

bool VideoStitchImpl::startStitch() {
    // stopStitch();
    m_stitch_running = true;
    // m_stitch_thread  = std::make_shared<std::thread>(&VideoStitchImpl::stitch_, this);
    return m_stitch_running;
}

bool VideoStitchImpl::stopStitch() {
    // if (m_stitch_thread != nullptr) {
    //     m_stitch_running = false;
    //     m_stitch_thread->join();
    //     m_stitch_thread.reset();
    // }
    m_stitch_running = false;
    return m_stitch_running;
}

bool VideoStitchImpl::go(std::vector<CvFrameBaseInfo>& frames) {
    std::vector<cv::Mat>  in;
    std::vector<cv::Rect> srt;
    std::vector<cv::Rect> drt;
    cv::Rect rt;
    cv::Mat* stitch_image = new cv::Mat;

    int i          = m_chan_start;

    uint64_t start_ts = bm::gettime_msec();
    in.clear();
    srt.clear();
    drt.clear();
    stitch_image->cols = 1920;
    stitch_image->rows = 1080;
    bool got_first = false;

    for (i = m_chan_start; i < m_chan_start + m_chan_count; ++i) {
        auto iter = m_channels.find(i);
        got_first = false;
        if (iter != m_channels.end()) {
            if (iter->second->seq != 0) {
                in.emplace_back(*iter->second->mat);
                got_first = true;
            }
        }
        if (!got_first) {
            std::cerr << "Channel " << i << " not got first frame. waiting..." << std::endl;
            //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            return false;
        }
        rt.x = rt.y = 0;
        rt.width  = 1920;
        rt.height = 1080;
        srt.push_back(rt);

        rt.x = (i % 2) * (stitch_image->cols / 2);
        rt.width = stitch_image->cols/2;
        rt.y = (i / 2) * (stitch_image->rows / 2);
        rt.height = stitch_image->rows / 2;
        // rt.x = (i % 4) * (stitch_image->cols / 4);
        // rt.width = stitch_image->cols / 4;
        // rt.y = 0;
        // rt.height = 1080;
        drt.push_back(rt);
    }
    // Not all channels got first frame
    if (!got_first) return false;

    bm::BMPerf perf;
    perf.begin("stitch", 20);
    if (BM_SUCCESS != cv::bmcv::stitch(in, srt, drt, *stitch_image, true, BMCV_INTER_LINEAR)) {
        std::cerr << "Stitch error" << std::endl;
        return false;
    }
    perf.end();

    CvFrameBaseInfo frame;
    frame.mat = stitch_image;
    frames.push_back(frame);
}

bool VideoStitchImpl::dataInput(CvFrameBaseInfo* frame, int count) {
    assert(frame != nullptr && count != 0);
    for (int i = 0; i < count; ++i) {
        auto chan = m_channels.find(frame->chan_id);
        if (chan == m_channels.end()) {
            std::cerr << "dataInput failed, channel: " << frame->chan_id << " not found" << std::endl;
            continue;
        }
        chan->second->sync(frame++);
    }
}

void VideoStitchImpl::fpsControl_(uint64_t msec_interval) {
    if (m_last_frame_ts == 0) {
        m_last_frame_ts = bm::gettime_msec();
    } else {
        uint64_t currrent    = bm::gettime_msec();
        uint64_t totoal_cost = currrent - m_last_frame_ts;
        
        if (totoal_cost > m_last_sleep_time) {
            uint64_t stitch_cost = totoal_cost - m_last_sleep_time;
            m_last_frame_ts      = currrent;
            if (msec_interval > stitch_cost) {
                m_last_sleep_time = msec_interval - stitch_cost;
                std::cout << "stitch sleep " << m_last_sleep_time << "ms" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(m_last_sleep_time));
            } else {
                m_last_sleep_time = 0;
            }
        } else {
            m_last_sleep_time = 0;
        }
        
    }
}