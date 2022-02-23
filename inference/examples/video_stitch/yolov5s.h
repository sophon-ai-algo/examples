//
// Created by yuan on 2/22/21.
//

#include "opencv2/opencv.hpp"
#include <sys/time.h>
#include "inference3.h"
#include "stream_cvdecode.h"
#include "stitch.h"
#include "encoder.h"



class YoloV5 : public bm::DetectorDelegate<CvFrameBaseInfo, CvFrameInfo> {
    int MAX_BATCH = 1;
    bm::BMNNContextPtr m_bmctx;
    bm::BMNNNetworkPtr m_bmnet;
    int m_net_h, m_net_w;

    //configuration
    float m_confThreshold= 0.2;
    float m_nmsThreshold = 0.5;
    float m_objThreshold = 0.2;
    std::vector<std::string> m_class_names;
    int m_class_num = 80; // default is coco names
    //const float m_anchors[3][6] = {{10.0, 13.0, 16.0, 30.0, 33.0, 23.0}, {30.0, 61.0, 62.0, 45.0, 59.0, 119.0},{116.0, 90.0, 156.0, 198.0, 373.0, 326.0}};
    std::vector<std::vector<std::vector<int>>> m_anchors{{{10, 13}, {16, 30}, {33, 23}},
                                                         {{30, 61}, {62, 45}, {59, 119}},
                                                         {{116, 90}, {156, 198}, {373, 326}}};
    const int m_anchor_num = 3;
    std::shared_ptr<VideoStitchImpl> m_stitch;
    std::shared_ptr<CVEncoder>       m_encoder;

public:
    YoloV5(bm::BMNNContextPtr bmctx, int max_batch=1);
    ~YoloV5();

    virtual int preprocess(std::vector<CvFrameBaseInfo>& frames, std::vector<CvFrameInfo>& frame_info) override ;
    virtual int forward(std::vector<CvFrameInfo> &frame_info) override;
    virtual int postprocess(std::vector<CvFrameInfo> &frame_infos, std::vector<CvFrameBaseInfo>& frames) override;
    virtual int stitch(std::vector<CvFrameBaseInfo> &frames, std::vector<CvFrameBaseInfo> &output) override;
    virtual int encode(std::vector<CvFrameBaseInfo>& frames) override;
    void setStitchImpl(std::shared_ptr<VideoStitchImpl> p_stitch) {
        m_stitch = p_stitch;
    }
    void setEncoder(std::shared_ptr<CVEncoder> encoder) {
        m_encoder = encoder;
    }

private:
    float sigmoid(float x);
    int argmax(float* data, int dsize);
    static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *alignWidth);
    void NMS(bm::NetOutputObjects &dets, float nmsConfidence);
    void extract_yolobox_cpu(CvFrameInfo& frameInfo);
};