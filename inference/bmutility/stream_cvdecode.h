#ifndef _STREAM_CVDECODE_H_
#define _STREAM_CVDECODE_H_

#include <thread>
#include "opencv2/opencv.hpp"

namespace bm {

    struct CvStreamDecoderEvents {
        virtual ~CvStreamDecoderEvents() {}

        virtual void on_decoded_mat(const cv::Mat& mat) = 0;
        virtual void on_stream_eof() {};
    };


    using CvMatPtr = cv::Mat*;

    class CvStreamDecoder {
        CvStreamDecoderEvents *m_observer;

        using OnDecodedFrameCallback    = std::function<void(CvMatPtr)>;
        using OnStreamEofCallback       = std::function<void()>;
        using OnCvCaptureEventCallback  = std::function<void(cv::VideoCapture&)>;

        OnDecodedFrameCallback   m_OnDecodedFrameFunc   {nullptr};
        OnCvCaptureEventCallback m_OnCvCaptureOpenedFunc{nullptr};
        OnCvCaptureEventCallback m_onCvCaptureClosedFunc{nullptr};

    protected:
        cv::VideoCapture m_cvcap;
        std::list<cv::Mat*> m_list_packets;

        bool m_is_waiting_iframe{true};
        bool m_keep_running{false};
        std::string m_url{""};
        int m_id{0};
        std::thread* m_thread_reading{nullptr};

        void clear_packets();

        int get_frame();
        void on_video_capture_closed();


    public:
        CvStreamDecoder(int id);
        virtual ~CvStreamDecoder();


        void set_decoded_frame_callback(OnDecodedFrameCallback func) {
            m_OnDecodedFrameFunc = func;
        }
        void set_cvcapture_opened_callback(OnCvCaptureEventCallback func) {
            m_OnCvCaptureOpenedFunc = func;
        }
        void set_cvcapture_closed_callback(OnCvCaptureEventCallback func) {
            m_onCvCaptureClosedFunc = func;
        }
        int open_stream(std::string url, bool repeat = true);
        int close_stream(bool is_waiting);
    };
}

#endif // _STREAM_CVDECODE_H_
