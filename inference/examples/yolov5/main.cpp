//
// Created by yuan on 3/4/21.
//

#include "opencv2/opencv.hpp"
#include "worker.h"
#include "configuration.h"
#include "bmutility_timer.h"
#include <iomanip>

const char* APP_ARG_STRING= //"{bmodel | /data/models/yolov5s_4batch_int8.bmodel | input bmodel path}"
                       "{bmodel | /data/models/yolov5s_1batch_fp32.bmodel | input bmodel path}"
                       "{max_batch | 1 | Max batch size}"
                        "{config | ./cameras.json | path to cameras.json}";


int main(int argc, char *argv[])
{
    const char *base_keys="{help | 0 | Print help information.}"
                     "{output | None | Output stream URL}"
                     "{skip | 1 | skip N frames to detect}"
                     "{num | 1 | Channels to run}";

    std::string keys;
    keys = base_keys;
    keys += APP_ARG_STRING;
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    std::string bmodel_file = parser.get<std::string>("bmodel");
    std::string output_url  = parser.get<std::string>("output");
    std::string config_file = parser.get<std::string>("config");

    int total_num = parser.get<int>("num");
    Config cfg(config_file.c_str());
    if (!cfg.valid_check()) {
        std::cout << "ERROR:" << config_file <<  " error, please check!" << std::endl;
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
    int skip = parser.get<int>("skip");
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
        bmlib_log_set_level(BMLIB_LOG_VERBOSE);

        int max_batch = parser.get<int>("max_batch");
        std::shared_ptr<YoloV5> detector = std::make_shared<YoloV5>(contextPtr, max_batch);

        OneCardInferAppPtr appPtr = std::make_shared<OneCardInferApp>(appStatis, gui,
                tqp, contextPtr, output_url, start_chan_index, channel_num, skip, max_batch);
        start_chan_index += channel_num;
        // set detector delegator
        appPtr->setDetectorDelegate(detector);
        appPtr->start(cfg.cardUrls(card_idx), cfg);
        apps.push_back(appPtr);
    }

    uint64_t timer_id;
    tqp->create_timer(1000, [&appStatis](){
        int ch = 0;
        appStatis.m_stat_imgps->update(appStatis.m_chan_statis[ch]);
        appStatis.m_total_fpsPtr->update(appStatis.m_total_statis);
        double imgps = appStatis.m_stat_imgps->getSpeed();
        double totalfps = appStatis.m_total_fpsPtr->getSpeed();
        std::cout << "[" << bm::timeToString(time(0)) << "] total fps ="
        << std::setiosflags(std::ios::fixed) << std::setprecision(1) << totalfps
        <<  ",ch=" << ch << ": speed=" << imgps  << std::endl;
    }, 1, &timer_id);

    tqp->run_loop();

    return 0;
}
