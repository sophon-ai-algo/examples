#include "bmutility_timer.h"
#include "stitch.h"
#include "bmutility_profile.h"

VideoStitchImpl::VideoStitchImpl(int chann_start, int chann_count, std::shared_ptr<CVEncoder>& encoder)
  : m_chan_start{chann_start},
    m_chan_count{chann_count},
    m_last_frame_ts{0},
    m_last_sleep_time{0},
    m_chan_mask{0},
    m_chan_got_frame{0},
    m_encoder{encoder} {

    for (int i = 0; i < m_chan_count; ++i) {
        m_chan_mask |= 1 << i;
    }
    for (int i = m_chan_start; i < m_chan_start + m_chan_count; i++) {
        SFrameAndBbox* f = new SFrameAndBbox;
        m_channels.insert(std::make_pair(i, f));
    }
}

VideoStitchImpl::~VideoStitchImpl() {
    auto iter = m_channels.begin();
    while (iter != m_channels.end()) {
        if (iter->second != nullptr) {
            if (iter->second->avframe != nullptr) {
                av_frame_unref(iter->second->avframe);
                av_frame_free(&iter->second->avframe);
            }
            delete iter->second;
        }
        ++iter;
    }
}

int VideoStitchImpl::encode(std::vector<bm::FrameBaseInfo>& frames) {
    for (auto& f : frames) {
        m_encoder->encode(f.cvimg);
    }
    return 0;
}

int VideoStitchImpl::draw(std::vector<bm::FrameInfo>& frames, std::vector<bm::FrameInfo>& output) {
    // Sync data
    dataInput_(frames);
    // draw when all channel updated
    if (m_chan_got_frame != m_chan_mask) {
        return -1;
    }

    m_chan_got_frame = 0;
    bm::FrameInfo fi;
    static int draw_flag = true;
    for (int i = m_chan_start; i < m_chan_start + m_chan_count; ++i) {
        auto iter = m_channels.find(i);
        if (iter->second->objs.size() > 0) {
            bmcv_rect_t rects[iter->second->objs.size()];
            for (int j = 0; j < iter->second->objs.size(); ++j) {
                rects[j].start_x = iter->second->objs[j].x1;
                rects[j].start_y = iter->second->objs[j].y1;
                rects[j].crop_w = iter->second->objs[j].x2 - iter->second->objs[j].x1;
                rects[j].crop_h = iter->second->objs[j].y2 - iter->second->objs[j].y1;
                // track id
                if (!draw_flag)
                    continue;
                std::string text = std::to_string(iter->second->objs[j].track_id);
                bmcv_point_t org = {rects[j].start_x, rects[j].start_y};
                bmcv_color_t color = {255, 0, 0};
                int thickness = 4;
                float font_scale = 4;
                bmcv_image_put_text(m_handle, *iter->second->bmimage, text.c_str(), org, color, font_scale, thickness);
            }
            bmcv_image_draw_rectangle(m_handle, *iter->second->bmimage, iter->second->objs.size(), rects, 3, 255, 0, 0);
        }

        bm::FrameBaseInfo fbi;
        fbi.original = *iter->second->bmimage;
        delete iter->second->bmimage;
        iter->second->bmimage = nullptr;
        fi.frames.push_back(fbi);
        fi.out_datums.push_back( iter->second->objs);
    }
    draw_flag = !draw_flag;
    output.push_back(fi);
    return 0;
}

int VideoStitchImpl::stitch(std::vector<bm::FrameInfo>& frames, std::vector<bm::FrameBaseInfo>& output) {
    assert(frames.size() == 1);
    auto& frame_info = frames[0];
    std::vector<bm_image>  in;
    std::vector<bmcv_rect_t> srt;
    std::vector<bmcv_rect_t> drt;
    bmcv_rect_t rt;
    bm_image stitch_image;

    bm_image_create(m_handle, 1080, 1920, FORMAT_YUV420P, DATA_TYPE_EXT_1N_BYTE, &stitch_image);
    bm_image_alloc_dev_mem(stitch_image, BMCV_HEAP1_ID);

    in.clear();
    srt.clear();
    drt.clear();
    stitch_image.width  = 1920;
    stitch_image.height = 1080;

    for (int i = 0; i < 4; ++i) {
        in.push_back(frame_info.frames[i].original);
        rt.start_x = rt.start_y = 0;
        rt.crop_w  = frame_info.frames[i].width;
        rt.crop_h  = frame_info.frames[i].height;
        srt.push_back(rt);

        rt.start_x = (i % 2) * (stitch_image.width / 2);
        rt.crop_w  = stitch_image.width / 2;
        rt.start_y = (i / 2) * (stitch_image.height / 2);
        rt.crop_h  = stitch_image.height / 2;
        drt.push_back(rt);
    }

    if (BM_SUCCESS != bmcv_image_vpp_stitch(m_handle,
                                            m_chan_count,
                                            &in[0],
                                            stitch_image,
                                            &drt[0])) {
        std::cerr << "Stitch error" << std::endl;
        return false;
    }

    cv::Mat output_mat;
    cv::bmcv::toMAT(&stitch_image, output_mat, true);
#if 0
    static int fff = 0;
    std::string filename = "jpgs/jpg_" + std::to_string(fff) + ".jpg";
    cv::imwrite(filename.c_str(), output_mat);
    for (int j = 0; j < 4; ++j) {
        cv::Mat output_mat;
        cv::bmcv::toMAT(& in[j], output_mat, true);
        std::string filename = "jpgs/jpg_" + std::to_string(fff) +
                + "_" + std::to_string(j) + ".jpg";
        cv::imwrite(filename.c_str(), output_mat);
    }
    fff++;
#endif

    bm::FrameBaseInfo frame_to_encode;
    frame_to_encode.cvimg = output_mat;
    output.push_back(frame_to_encode);

    for (int i = 0; i < 4; ++i) {
        bm_image_destroy(frame_info.frames[i].original);
    }
    bm_image_destroy(stitch_image);

}

void VideoStitchImpl::dataInput_(std::vector<bm::FrameInfo>& frames) {
    for (auto& f : frames) {
        auto& frames_info = f.frames;

        for (int i = 0; i < frames_info.size(); ++i) {

            auto chan = m_channels.find(frames_info[i].chan_id);
            if (chan == m_channels.end()) {
                std::cerr << "dataInput failed, channel: " << frames_info[i].chan_id << " not found" << std::endl;
                bm_image_destroy( frames_info[i].original);
                continue;
            }
            if (chan->second->seq > 0 && frames_info[i].seq <= chan->second->seq) {
                bm_image_destroy( frames_info[i].original);
                continue;
            }
            m_chan_got_frame |= 1 << (frames_info[i].chan_id - m_chan_start);

            if (chan->second->bmimage != nullptr) {
                bm::BMImage::safe_dalete_bm_image_ptr(&chan->second->bmimage);
            }

            chan->second->objs    = f.out_datums[i].track_rects;

            chan->second->bmimage  = new bm_image;
            *chan->second->bmimage = frames_info[i].original;
            chan->second->seq      =  frames_info[i].seq;
        }
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