//
// Created by xwang on 2022-2-10.
//

#include "stream_cvdecode.h"

namespace bm {
    CvStreamDecoder::CvStreamDecoder(int id)
            : m_observer(nullptr)  {
        std::cout << "CvStreamDecoderEvents() ctor..." << std::endl;
        m_cvcap = cv::VideoCapture();
        m_is_waiting_iframe = true;
        m_id = id;
    }

    CvStreamDecoder::~CvStreamDecoder() {
        std::cout << "~CvStreamDecoderEvents() dtor..." << std::endl;
        on_video_capture_closed();
    }

    int CvStreamDecoder::get_frame() {

    }

    void CvStreamDecoder::on_video_capture_closed() {
        clear_packets();
        if (m_cvcap.isOpened()) {
            m_cvcap.release();
        }
    }

    void CvStreamDecoder::clear_packets() {
        while (m_list_packets.size() > 0) {
            auto mat = m_list_packets.front();
            m_list_packets.pop_front();
        }
        return;
    }


    int CvStreamDecoder::open_stream(std::string url, bool repeat)
    {
        m_url = url;
        m_keep_running = true;
        m_thread_reading = new std::thread([&] {
            if (!m_cvcap.open(m_url, cv::CAP_ANY, 0)) {
                std::cerr << "open " << url << " failed!" << std::endl;
                m_keep_running = false;
                return;
            }
            if (m_OnCvCaptureOpenedFunc != nullptr) {
                m_OnCvCaptureOpenedFunc(m_cvcap);
            }
            while (m_keep_running) {
                CvMatPtr p_frame = new cv::Mat;
                bool ret = m_cvcap.read(*p_frame);
                if (!ret || p_frame->empty()) {
                    std::cerr << m_url << " cv read frame failed" << std::endl;
                    continue;
                }
                m_OnDecodedFrameFunc(p_frame);
            }
        });

        return 0;
    }

    int CvStreamDecoder::close_stream(bool is_waiting) {
        m_keep_running = false;
        if (m_thread_reading != nullptr) {
            m_thread_reading->join();
            delete m_thread_reading;
            m_thread_reading = nullptr;
        }
        if (m_cvcap.isOpened()) {
            m_cvcap.release();
            std::cout << "close" << m_url << std::endl;
        }
        return 0;
    }
}
