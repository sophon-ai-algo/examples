#include "opencv2/opencv.hpp"
#include "worker.h"
#include "configuration.h"
#include "bmutility_timer.h"
#include <iomanip>


const char *APP_ARG_STRING= "{bmodel | /data/workspace/models/ruilai/RETINAFACE_RUILAI_BATCH4_INT8_5_6.bmodel | input bmodel path}"
                       "{config | ./cameras.json | path to cameras.json}"
                       "{cls_model1 | /data/workspace/models/ruilai/mobilenetv2_batch4.bmodel | class model1}"
                       "{cls_model2 | /data/workspace/models/ruilai/WSDAN_5_5_BATCH4_bmnetp_INT8_BMODEL.bmodel | class model2}"
                       "{cls_model3 | /data/workspace/models/ruilai/WSDAN_5_5_BATCH4_bmnetp_INT8_BMODEL.bmodel | class model3}"
                       "{cls_model4 | /data/workspace/models/ruilai/WSDAN_5_5_BATCH4_bmnetp_INT8_BMODEL.bmodel | class model4}"
                       "{cards | 1 | cards amount}";

int main(int argc, char *argv[])
{

    const char *base_keys="{help | 0 | Print help information.}"
                     "{output | None | Output stream URL}"
                     "{skip | 1 | skip N frames to detect}"
                     "{num | 4 | Channels to run}";

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
    std::string cls_model1  = parser.get<std::string>("cls_model1");
    std::string cls_model2  = parser.get<std::string>("cls_model2");
//    std::string cls_model3  = parser.get<std::string>("cls_model3");
//    std::string cls_model4  = parser.get<std::string>("cls_model4");

    int card_num  = parser.get<int>("cards");
    int total_num = parser.get<int>("num");
    int skip      = parser.get<int>("skip");
    int resize_q  = 4;

    Config cfg(config_file.c_str());
    if (!cfg.valid_check()) {
        std::cout << "ERROR:cameras.json config error, please check!" << std::endl;
        return -1;
    }

    AppStatis appStatis(total_num);

    int channel_num_per_card = total_num/card_num;
    int last_channel_num = total_num % card_num == 0 ? 0:total_num % card_num;

    std::shared_ptr<bm::VideoUIApp> gui;


    bm::TimerQueuePtr tqp = bm::TimerQueue::create();
    int start_chan_index = 0;
    std::vector<OneCardInferAppPtr> apps;
    bmlib_log_set_level(BMLIB_LOG_VERBOSE);

    for(int card_idx = 0; card_idx < card_num; ++card_idx) {
        int dev_id = card_idx;
        // load balance
        int channel_num = 0;
        if (card_idx < last_channel_num) {
            channel_num = channel_num_per_card + 1;
        }else{
            channel_num = channel_num_per_card;
        }

        bm::BMNNHandlePtr handle          = std::make_shared<bm::BMNNHandle>(dev_id);
        bm::BMNNContextPtr detContextPtr  = std::make_shared<bm::BMNNContext>(handle, bmodel_file);
        bm::BMNNContextPtr clsContextPtr1 = std::make_shared<bm::BMNNContext>(handle, cls_model1);
        bm::BMNNContextPtr clsContextPtr2 = std::make_shared<bm::BMNNContext>(handle, cls_model2);
//        bm::BMNNContextPtr clsContextPtr3 = std::make_shared<bm::BMNNContext>(handle, cls_model3);
//        bm::BMNNContextPtr clsContextPtr4 = std::make_shared<bm::BMNNContext>(handle, cls_model4);


        auto detector  = std::make_shared<Retinaface>(detContextPtr);
        auto classify1 = std::make_shared<MobileNetV2>(clsContextPtr1);
        auto classify2 = std::make_shared<WSDAN>(clsContextPtr2);
//        auto classify3 = std::make_shared<Resnet>(clsContextPtr3);
//        auto classify4 = std::make_shared<Resnet>(clsContextPtr4);


        OneCardInferAppPtr appPtr = std::make_shared<OneCardInferApp>(appStatis, gui,
                tqp, handle, start_chan_index, channel_num, resize_q, skip);
        start_chan_index += channel_num;
        // set detector delegator
        appPtr->setDetectorDelegate(detector);
        appPtr->setClassifyDelegate_224(classify1);
        appPtr->setClassifyDelegate_320(classify2);
//        appPtr->setClassifyDelegate_320(classify3);
//        appPtr->setClassifyDelegate_320(classify4);

        appPtr->start(cfg.cardUrls(0), cfg);
        apps.push_back(appPtr);
    }

    uint64_t timer_id;
    tqp->create_timer(1000, [&appStatis](){
        int ch = 0;
        appStatis.m_stat_imgps->update(appStatis.m_chan_statis[ch]);
        appStatis.m_total_fpsPtr->update(appStatis.m_total_statis);
        double imgps = appStatis.m_stat_imgps->getSpeed();
        double totalfps = appStatis.m_total_fpsPtr->getSpeed();
        std::cout << "[" << bm::timeToString(time(0)) << "] total fps = "
        << std::setiosflags(std::ios::fixed) << std::setprecision(1) << totalfps
        <<  ",ch=" << ch << ": speed=" << imgps  << std::endl;
    }, 1, &timer_id);

    tqp->run_loop();

    return 0;
}
