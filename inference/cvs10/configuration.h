//
// Created by yuan on 3/12/21.
//

#ifndef INFERENCE_FRAMEWORK_CONFIGURATION_H
#define INFERENCE_FRAMEWORK_CONFIGURATION_H

#include <fstream>
#include <unordered_map>
#include "json/json.h"

struct CardConfig {
    int devid;
    std::vector<std::string> urls;
};

struct SConcurrencyConfig {
    int  thread_num {4};
    int  queue_size {4};
    bool blocking   {false};

    SConcurrencyConfig() = default;

    SConcurrencyConfig(Json::Value& value) {
        load(value);
    }

    void load(Json::Value& value) {
        thread_num = value["thread_num"].asInt();
        queue_size = value["queue_size"].asInt();
        blocking   = value["blocking"].asBool();
    }
};

class Config {
    std::vector<CardConfig> m_cards;
    std::unordered_map<std::string, SConcurrencyConfig> m_concurrency;

    void load_cameras(std::vector<CardConfig> &vctCardConfig, const char* config_file = "cameras.json") {
#if 1
        Json::Reader reader;
        Json::Value json_root;

        std::ifstream in(config_file);
        if (!in.is_open()) {
            printf("Can't open file: %s\n", config_file);
            return;
        }

        if (!reader.parse(in, json_root, false)) {
            return;
        }


        if (json_root["cards"].isNull() || !json_root["cards"].isArray()){
            in.close();
            return;
        }

        int card_num = json_root["cards"].size();
        for(int card_index = 0; card_index < card_num; ++card_index) {
            Json::Value jsonCard = json_root["cards"][card_index];
            CardConfig card_config;
            card_config.devid = jsonCard["devid"].asInt();
            int camera_num = jsonCard["cameras"].size();
            Json::Value jsonCameras = jsonCard["cameras"];
            for(int i = 0;i < camera_num; ++i) {
                auto json_url_info = jsonCameras[i];
                int chan_num = json_url_info["chan_num"].asInt();
                for(int j = 0; j < chan_num; ++j) {
                    auto url = json_url_info["address"].asString();
                    card_config.urls.push_back(url);
                }
            }

            vctCardConfig.push_back(card_config);
        }
        // load thread_num, queue_size for concurrency
        if (json_root.isMember("pipeline")) {
            Json::Value pipeline_config = json_root["pipeline"];
            maybe_load_concurrency_cfg(pipeline_config, "preprocess");
            maybe_load_concurrency_cfg(pipeline_config, "inference");
            maybe_load_concurrency_cfg(pipeline_config, "postprocess");
        }
        in.close();
#else
        for(int i=0; i < 2; i ++) {
            CardConfig cfg;
            cfg.devid = i;
            std::string url = "/home/yuan/station.avi";
            for(int j = 0;j < 1;j ++) {
                cfg.urls.push_back(url);
            }

            m_cards.push_back(cfg);
        }
#endif
    }

public:
    Config(const char* config_file = "cameras.json") {
        load_cameras(m_cards, config_file);
    }

    int cardNums() {
        return m_cards.size();
    }

    int cardDevId(int index){
        return m_cards[index].devid;
    }

    const std::vector<std::string>& cardUrls(int index) {
        return m_cards[index].urls;
    }

    bool valid_check(int total) {
        if (m_cards.size() == 0) return false;

        for(int i = 0;i < m_cards.size(); ++i) {
            if (m_cards.size() == 0) return false;
        }

        return true;
    }

    bool maybe_load_concurrency_cfg(Json::Value& json_node, const char* phrase) {
        if (json_node.isMember(phrase)) {
            SConcurrencyConfig cfg(json_node[phrase]);
            m_concurrency.insert(std::make_pair(phrase, cfg));
        }
    }

    bool get_phrase_config(const char* phrase, SConcurrencyConfig& cfg) {
        if (m_concurrency.find(phrase) != m_concurrency.end()) {
            cfg = m_concurrency[phrase];
            return true;
        }
        return false;
    }
};


struct AppStatis {
    int m_channel_num;
    std::mutex m_statis_lock;
    bm::StatToolPtr m_chan_det_fpsPtr;
    bm::StatToolPtr m_total_det_fpsPtr;
    bm::StatToolPtr m_chan_feat_fpsPtr;
    bm::StatToolPtr m_total_feat_fpsPtr;

    uint64_t *m_chan_statis;
    uint64_t m_total_statis = 0;
    uint64_t  *m_chan_feat_stat;
    uint64_t  m_total_feat_stat=0;


    AppStatis(int num):m_channel_num(num) {
        m_chan_det_fpsPtr = bm::StatTool::create(5);
        m_total_det_fpsPtr = bm::StatTool::create(5);
        m_chan_feat_fpsPtr = bm::StatTool::create(5);
        m_total_feat_fpsPtr = bm::StatTool::create(5);

        m_chan_statis = new uint64_t[m_channel_num];
        memset(m_chan_statis, 0, sizeof(uint64_t)*m_channel_num);
        m_chan_feat_stat = new uint64_t[m_channel_num];
        memset(m_chan_feat_stat, 0, sizeof(uint64_t)*m_channel_num);
        assert(m_chan_feat_stat != nullptr);
    }

    ~AppStatis() {
        delete [] m_chan_statis;
        delete [] m_chan_feat_stat;
    }


};




#endif //INFERENCE_FRAMEWORK_CONFIGURATION_H
