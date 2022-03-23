//
// Created by yuan on 3/4/21.
//

#include "opencv2/opencv.hpp"
#include "worker.h"
#include "configuration.h"
#include "bmutility_timer.h"
#include <iomanip>
#include "face_extract.h"
#include "resnet50.h"

enum ModelType {
    MODEL_FACE_DETECT=0,
    MODEL_RESNET50=1
};

int main(int argc, char *argv[])
{
    const char *base_keys="{help | 0 | Print help information.}"
                          "{model_type | 1 | Model Type(0: face_detect 1: resnet50)}"
                          "{bmodel | /data/models/cvs10.bmodel | input bmodel path}"
                          "{max_batch | 4 | Max batch size}"
                          "{enable_l2_ddr_reduction | 1 | L2 ddr reduction}"
                          "{feat_delay | 1000 | feature delay in msec}"
                          "{feat_num | 8 | feature num per channel}"
                          "{skip | 1 | skip N frames to detect}"
                          "{num | 1 | Channels to run}"
                          "{config | ./cameras.json | path to cameras.json}";

    std::string keys;
    keys = base_keys;
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    std::string bmodel_file = parser.get<std::string>("bmodel");
    std::string config_file = parser.get<std::string>("config");

    int skip = parser.get<int>("skip");
    int model_type = parser.get<int>("model_type");
    int total_num = parser.get<int>("num");
    int feature_delay = parser.get<int>("feat_delay");
    int feature_num = parser.get<int>("feat_num");

    int enable_l2_ddrr = parser.get<int>("enable_l2_ddr_reduction");

    Config cfg(config_file.c_str());
    if (!cfg.valid_check(total_num)) {
        std::cout << "ERROR:cameras.json config error, please check!" << std::endl;
        return -1;
    }

    AppStatis appStatis(total_num);

    int card_num = cfg.cardNums();
    int channel_num_per_card = total_num/card_num;
    int last_channel_num = total_num % card_num == 0 ? 0:total_num % card_num;

    std::shared_ptr<bm::VideoUIApp> gui;
#if USE_QTGUI
    gui = bm::VideoUIApp::create(argc, argv);
    gui->bootUI(total_num);
#endif

    bm::TimerQueuePtr tqp = bm::TimerQueue::create();
    int start_chan_index = 0;
    std::vector<OneCardInferAppPtr> apps;
    for(int card_idx = 0; card_idx < card_num; ++card_idx) {
        int dev_id = cfg.cardDevId(card_idx);
        // load balance
        int channel_num = 0;
        if (card_idx < last_channel_num) {
            channel_num = channel_num_per_card + 1;
        }else{
            channel_num = channel_num_per_card;
        }

        bm::BMNNHandlePtr handle = std::make_shared<bm::BMNNHandle>(dev_id);
        bm::BMNNContextPtr contextPtr = std::make_shared<bm::BMNNContext>(handle, bmodel_file);

        int max_batch = parser.get<int>("max_batch");
        std::shared_ptr<bm::DetectorDelegate<bm::cvs10FrameBaseInfo, bm::cvs10FrameInfo>> detector;
        if (MODEL_FACE_DETECT == model_type) {
            detector = std::make_shared<FaceDetector>(contextPtr, max_batch);
        }else if (MODEL_RESNET50 == model_type) {
            detector = std::make_shared<Resnet>(contextPtr, max_batch);
        }

        std::cout << "start_chan_index=" << start_chan_index << ", channel_num=" << channel_num << std::endl;
        OneCardInferAppPtr appPtr = std::make_shared<OneCardInferApp>(appStatis, gui,
                tqp, contextPtr, start_chan_index, channel_num, skip, feature_delay, feature_num,
                enable_l2_ddrr);
        start_chan_index += channel_num;

        // set detector delegator
        appPtr->setDetectorDelegate(detector);
        std::shared_ptr<bm::DetectorDelegate<bm::FeatureFrame, bm::FeatureFrameInfo>> feature_delegate;
        feature_delegate = std::make_shared<FaceExtract>(contextPtr, max_batch);
        appPtr->setFeatureDelegate(feature_delegate);
        appPtr->start(cfg.cardUrls(card_idx), cfg);
        apps.push_back(appPtr);
    }

    uint64_t timer_id;
    tqp->create_timer(1000, [&appStatis](){
        int ch = 0;
        appStatis.m_chan_det_fpsPtr->update(appStatis.m_chan_statis[ch]);
        appStatis.m_total_det_fpsPtr->update(appStatis.m_total_statis);

        appStatis.m_chan_feat_fpsPtr->update(appStatis.m_chan_feat_stat[ch]);
        appStatis.m_total_feat_fpsPtr->update(appStatis.m_total_feat_stat);

        double chanfps = appStatis.m_chan_det_fpsPtr->getSpeed();
        double totalfps = appStatis.m_total_det_fpsPtr->getSpeed();

        double feat_chanfps = appStatis.m_chan_feat_fpsPtr->getSpeed();
        double feat_totalfps = appStatis.m_total_feat_fpsPtr->getSpeed();

        std::cout << "[" << bm::timeToString(time(0)) << "] det (total fps ="
        << std::setiosflags(std::ios::fixed) << std::setprecision(1) << totalfps
        <<  ",ch=" << ch << ": speed=" << chanfps
        << ") feature (total fps=" << std::setiosflags(std::ios::fixed) << std::setprecision(1)
        << feat_totalfps <<  ",ch=" << ch << ": speed=" << feat_chanfps << ")" << std::endl;
    }, 1, &timer_id);

    tqp->run_loop();

    return 0;
}
