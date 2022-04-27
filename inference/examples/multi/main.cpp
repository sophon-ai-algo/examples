
#include "worker.h"
#include "configuration.h"

const char* APP_ARG_STRING = "{config | ./cameras_v1.json | path to cameras.json}";


int main(int argc, char *argv[])
{
    bmlib_log_set_level(BMLIB_LOG_VERBOSE);

    const char *base_keys="{help | 0 | Print help information.}";
    std::string keys;
    keys = base_keys;
    keys += APP_ARG_STRING;
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }
    std::string config_file = parser.get<std::string>("config");


    Config cfg(config_file.c_str());
    if (!cfg.valid_check()) {
        std::cout << "ERROR:" << config_file <<  " error, please check!" << std::endl;
        return -1;
    }
    int total_num = (int)cfg.getTotalUrlNum();
    AppStatis appStatis(total_num);
    std::vector<OneCardInferAppPtr> apps;
    
    auto modelConfig = cfg.getModelConfig();
    int card_num = cfg.cardNums();
    int stream_ch = 0;
    for(int card_idx = 0; card_idx < card_num; ++card_idx) {
        int dev_id = cfg.cardDevId(card_idx);
        std::set<std::string> distinct_models = cfg.getDistinctModels(dev_id);
        std::map<std::string, OneCardInferAppPtr> model2App;
        
        // init pipeline by model
        for (auto iter = distinct_models.begin(); iter != distinct_models.end(); iter++) {
            std::string model_name = *iter;
            auto& model_cfg = modelConfig[model_name];
            bm::BMNNHandlePtr handle         = std::make_shared<bm::BMNNHandle>(dev_id);
            bm::BMNNContextPtr contextPtr    = std::make_shared<bm::BMNNContext>(handle, model_cfg.path);
            std::shared_ptr<YoloV5> detector = std::make_shared<YoloV5>(contextPtr);
            // model thresholds
            detector->set_cls(model_cfg.class_threshold);
            detector->set_obj(model_cfg.obj_threshold);
            detector->set_nms(model_cfg.nms_threshold);
            OneCardInferAppPtr appPtr = std::make_shared<OneCardInferApp>(appStatis, contextPtr, detector->getBatch(), model_cfg.skip_frame);
            apps.push_back(appPtr);
            appPtr->setDetectorDelegate(detector);
            model2App.insert(std::make_pair(model_name, appPtr));
        }

        std::vector<std::string> urls   = cfg.cardUrls(card_idx);
        std::vector<std::string> models = cfg.cardModels(card_idx);
        assert(urls.size() == models.size());
        
        // insert stream into pipeline
        for (size_t i = 0; i < urls.size(); ++i) {
            model2App[models[i]]->addStream(urls[i], stream_ch++);
        }

        for (auto iter = model2App.begin(); iter != model2App.end(); ++iter) {
            // start pipeline
            iter->second->start(cfg);
        }
    }

    uint64_t timer_id;
    bm::TimerQueuePtr tqp = bm::TimerQueue::create();
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
