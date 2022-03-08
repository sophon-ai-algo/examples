#ifndef _VIDEO_STITCH_ENCODER_H_
#define _VIDEO_STITCH_ENCODER_H_

#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/vpp.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include "rtsp/Live555RtspServer.h"
#include "configuration.h"

class CVEncoder {
public:
    CVEncoder(int fps, int w, int h, int card, BTRTSPServer* p_rtsp, AppStatis* appStatis);
    virtual ~CVEncoder();
    bool encode(cv::Mat& mat);

private:
    int m_fps;
    int m_width;
    int m_height;
    int m_card;
    std::shared_ptr<cv::VideoWriter> m_writer;
    std::unique_ptr<char[]>       m_buffer;
    BTRTSPServer* m_rtsp;
    AppStatis* m_appstatis;
};


#endif // _VIDEO_STITCH_ENCODER_H_