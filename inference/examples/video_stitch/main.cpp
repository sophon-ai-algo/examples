//
// Created by xwang on 2/10/22.
//
#include <iomanip>
#include "opencv2/opencv.hpp"
#include "worker.h"
#include "configuration.h"
#include "bmutility_timer.h"
#include "rtsp/Live555RtspServer.h"
#include "encoder.h"
#include "stitch.h"


const char* APP_ARG_STRING= //"{bmodel | /data/models/yolov5s_4batch_int8.bmodel | input bmodel path}"
                       "{bmodel | /data/models/yolov5s_1batch_fp32.bmodel | input bmodel path}"
                       "{max_batch | 4 | Max batch size}"
                       "{config | ./cameras.json | path to cameras.json}";


int main(int argc, char *argv[])
{
    const char *base_keys="{help | 0 | Print help information.}"
                     "{skip | 2 | skip N frames to detect}"
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
    std::string config_file = parser.get<std::string>("config");

    int total_num = parser.get<int>("num");
    if (total_num != 4) {
        std::cerr << "Only support 2x2 layout, make the num be equal to 4!!";
        return -1;
    }
    Config cfg(config_file.c_str());
    if (!cfg.valid_check(total_num)) {
        std::cout << "ERROR:cameras.json config error, please check!" << std::endl;
        return -1;
    }

    int card_num = cfg.cardNums();
    int channel_num_per_card = total_num/card_num;
    int last_channel_num = total_num % card_num == 0 ? 0:total_num % card_num;

    std::shared_ptr<bm::VideoUIApp> gui;

    bm::TimerQueuePtr tqp = bm::TimerQueue::create();
    int start_chan_index = 0;
    std::vector<OneCardInferAppPtr> apps;
    int skip = parser.get<int>("skip");

    // live555 RTSP Server
    CreateRtspServer();
    BTRTSPServer* pRtspServer = GetRTSPInstance();

    // Only statistics encoder fps
    AppStatis appStatis(1);

    std::shared_ptr<CVEncoder>       encoder = std::make_shared<CVEncoder>(25 / skip, 1920, 1080, 0, pRtspServer, &appStatis);
    std::shared_ptr<VideoStitchImpl> stitch  = std::make_shared<VideoStitchImpl>(0, total_num, encoder);
    bm::BMMediaPipeline<bm::FrameBaseInfo, bm::FrameInfo> m_media_pipeline;
    bm::MediaParam param;
    param.stitch_thread_num = 1;
    param.stitch_queue_size = 20;
    param.encode_thread_num = 1;
    param.encode_queue_size = 10;

    m_media_pipeline.init(
        param, stitch,
        bm::FrameInfo::FrameInfoDestroyFn,
        bm::FrameBaseInfo::FrameBaseInfoDestroyFn
    );

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

        if (card_idx == card_num - 1) {
            stitch->setHandle(contextPtr->handle());
        }

        int max_batch = parser.get<int>("max_batch");
        std::shared_ptr<YoloV5> detector = std::make_shared<YoloV5>(contextPtr, max_batch);
        detector->set_next_inference_pipe(&m_media_pipeline);


        OneCardInferAppPtr appPtr = std::make_shared<OneCardInferApp>(
                tqp, contextPtr, start_chan_index, channel_num, skip, max_batch);
        start_chan_index += channel_num;
        // set detector delegator
        appPtr->setDetectorDelegate(detector);
        appPtr->start(cfg.cardUrls(card_idx), cfg);
        apps.push_back(appPtr);
    }

    uint64_t timer_id;
    tqp->create_timer(1000, [&appStatis](){
        appStatis.m_total_fpsPtr->update(appStatis.m_total_statis);
        double totalfps = appStatis.m_total_fpsPtr->getSpeed();
        std::cout << "[" << bm::timeToString(time(0)) << "] encode fps ="
        << std::setiosflags(std::ios::fixed) << std::setprecision(1) << totalfps << std::endl;
    }, 1, &timer_id);

    tqp->run_loop();

    return 0;
}
