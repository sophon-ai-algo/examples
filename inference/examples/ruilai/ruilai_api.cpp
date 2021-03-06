#include "ruilai_api.h"
#include "configuration.h"

RuiLaiAPIWrapper::RuiLaiAPIWrapper(int card_num, 
                                   std::string retinaface_bmodel, float face_threshold,  // 这些threshold接口层先暴露给客户
                                   std::string cls_bmodel1, float cls1_threshold,        // 底层还未实现对应可配参数
                                   std::string cls_bmodel2, float cls2_threshold,        // 先保留下来
                                   std::string cls_bmodel3, float cls3_threshold,
                                   std::string cls_bmodel4, float cls4_threshold,
                                   ImgResultCallBackFunc func,
                                   std::string config_file)
  : m_cards{card_num}, m_imageIndex{0}, m_callbackFunc{func}, m_appStatis{1} {

    Config cfg(config_file.c_str());
    if (!cfg.valid_check()) {
        std::cout << "ERROR:cameras.json config error, please check!" << std::endl;
        throw std::runtime_error("read config fail");
    }  

    m_jpegQueue = std::make_shared<BlockingQueue<JPGUint>>("jpeg", 0, 32);
    m_jpegWorkerPool.init(m_jpegQueue.get(), 64, 1, 1);
    m_jpegWorkerPool.startWork([this](std::vector<JPGUint> &items) {
        for (int i = 0; i < items.size(); ++i) {
            JPGUint &jpg = items[i];
            int card_index = jpg.image_id % m_cards;

            std::vector<uchar> pic(jpg.jpeg_data, jpg.jpeg_data + jpg.len);
            cv::Mat cvimage;
            // 使用硬件转换为 bm_image对象
            // 这里默认使用设备0
            // 应根据实际card_num来平均负载到各设备上硬解
            cv::imdecode(pic, cv::IMREAD_AVFRAME, &cvimage, card_index);
            bm::FrameBaseInfo fbi;
            fbi.chan_id = 0;
            fbi.seq = jpg.image_id;
            bm_image image;
            cv::bmcv::toBMI(cvimage, &image, false);
#if 0
            char save_path[256];
            snprintf(save_path, 256, "output.bmp");
            bm_image_write_to_bmp(image, save_path);
#endif
            fbi.original = image;
            // 平均分发给各芯片
            m_vApps[card_index]->pushFrame(&fbi);
        }
        
    });

    for(int card_idx = 0; card_idx < card_num; ++card_idx) {
        int dev_id = card_idx;

        bm::BMNNHandlePtr handle          = std::make_shared<bm::BMNNHandle>(dev_id);
        bm::BMNNContextPtr detContextPtr  = std::make_shared<bm::BMNNContext>(handle, retinaface_bmodel);
        bm::BMNNContextPtr clsContextPtr1 = std::make_shared<bm::BMNNContext>(handle, cls_bmodel1);
        bm::BMNNContextPtr clsContextPtr2 = std::make_shared<bm::BMNNContext>(handle, cls_bmodel2);
        // bm::BMNNContextPtr clsContextPtr3 = std::make_shared<bm::BMNNContext>(handle, cls_bmodel3);
        // bm::BMNNContextPtr clsContextPtr4 = std::make_shared<bm::BMNNContext>(handle, cls_bmodel4);

        auto detector  = std::make_shared<Retinaface>(detContextPtr);
        auto classify1 = std::make_shared<MobileNetV2>(clsContextPtr1);
        auto classify2 = std::make_shared<WSDAN>(clsContextPtr2);
        // auto classify3 = std::make_shared<Resnet>(clsContextPtr3);
        // auto classify4 = std::make_shared<Resnet>(clsContextPtr4);

        OneCardInferAppPtr appPtr = std::make_shared<OneCardInferApp>(m_appStatis, nullptr, handle, 0, 0, 4, 1, classify1->getBatchSize(), detector->getBatchSize());

        // set detector delegator
        appPtr->setDetectorDelegate(detector);
        // set classify delegator
        appPtr->setClassifyDelegate_224(classify1);
        appPtr->setClassifyDelegate_320(classify2);
        // appPtr->setClassifyDelegate_320(classify3);
        // appPtr->setClassifyDelegate_320(classify4);
        appPtr->setImgResultCallback(m_callbackFunc);
        std::vector<std::string> dummy_urls;
        appPtr->start(dummy_urls, cfg);
        m_vApps.push_back(appPtr);
    }
}

RuiLaiAPIWrapper::~RuiLaiAPIWrapper() {}


uint64_t RuiLaiAPIWrapper::Infer(const unsigned char *jpeg_data, int len) {
    uint64_t image_id = m_imageIndex++;
    JPGUint unit;
    unit.jpeg_data = jpeg_data;
    unit.len = len;
    unit.image_id = image_id;
    m_jpegQueue->push(unit);
    return image_id;
}

