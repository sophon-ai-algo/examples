#include <iostream>
#include <queue>
#include "encoder.h"
#include "rtsp/Live555RtspServer.h"

std::queue<PktData>   g_queEncodeNal;
pthread_mutex_t  g_lockQueEncodeNal;



CVEncoder::CVEncoder(int fps, int w, int h, int card, BTRTSPServer* p_rtsp, AppStatis* appStatis)
    : m_fps{fps},
    m_width{w},
    m_height{h},
    m_card{card},
    m_rtsp{p_rtsp},
    m_appstatis{appStatis} {

    std::cout << "CVEncoder construct" << std::endl;
    m_writer = std::make_shared<cv::VideoWriter>();
    m_writer->open(
        "", cv::VideoWriter::fourcc('H', '2', '6', '4'),
        m_fps,
        cv::Size(m_width,m_height),
        true,
        m_card
    );
    m_buffer = std::make_unique<char[]>(m_width * m_height * 4);
    memset(m_buffer.get(), 0, m_width * m_height * 4);
}

CVEncoder::~CVEncoder() {
    if (m_writer->isOpened()) {
        m_writer->release();
    }
    m_writer.reset();
}

bool CVEncoder::encode(cv::Mat& mat) {
    if (!m_writer->isOpened()) {
        std::cerr << "encoder not open" << std::endl;
        return false;
    }

    PktData pPktData = std::make_shared<_tagPktData>();
    pPktData->len = m_width * m_height * 4;
    pPktData->buf = new char[pPktData->len];

    m_writer->write(mat, (char*)pPktData->buf ,&pPktData->len);
    mat.release();
    m_appstatis->m_total_statis++;

    if (pPktData->len <= 0) {
        pPktData.reset();
        return false;
    }

    if (pPktData->buf[4] == 0x67) {
       m_rtsp->flushData();
    }
    m_rtsp->inputData(pPktData);

   return true;
}