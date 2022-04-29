#include <assert.h>
#include <map>
#include <vector>
#include "bmcv_api_ext.h"
#include "ruilai_api.h"
#include <functional>

#define TEST_PICS_NUM 1000
#define FFALIGN(x, a) (((x)+(a)-1)&~((a)-1))

static int counter = 0;
auto start = std::chrono::high_resolution_clock::now();

void call_back(uint64_t image_id, bool ret, float score) {
    if (++counter == TEST_PICS_NUM) {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "!!!!!!!" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

    }
    std::cout << "image_id: " << image_id << ", "
              << "ret:      " << ret      << ", "
              << "score:    " << score    << std::endl;

}

int main(int argc, char* argv[]) {
    // 初始化参数
    int card_num = 1;
    std::string jpg_path = "/data/workspace/media/station.jpg";

    std::string retinaface_bmodel = "/data/workspace/models/retinaface_mobilenet0.25_384x640_fp32_b4.bmodel";
    float face_threshold = 0.5f;
    std::string cls_bmodel1 = "/data/workspace/models/ruilai/RUILAI_MOBILENETV2_BATCH4_BMODEL/compilation.bmodel";
    float cls1_threshold = 0.5f;
    std::string cls_bmodel2 = "/data/workspace/models/ruilai/WSDAN_BATCH1/compilation.bmodel";
    float cls2_threshold = 0.5f;
    std::string cls_bmodel3 = "/data/workspace/models/ruilai/WSDAN_BATCH1/compilation.bmodel";
    float cls3_threshold = 0.5f;
    std::string cls_bmodel4 = "/data/workspace/models/ruilai/WSDAN_BATCH1/compilation.bmodel";
    float cls4_threshold = 0.5f;
    ImgResultCallBackFunc func = std::bind(call_back, std::placeholders::_1,
                                        std::placeholders::_2,
                                        std::placeholders::_3);
    std::string config_file = "./cameras.json";
    RuiLaiAPIWrapper instance(card_num,
                              retinaface_bmodel, face_threshold,
                              cls_bmodel1, cls1_threshold,
                              cls_bmodel2, cls2_threshold,
                              cls_bmodel3, cls3_threshold,
                              cls_bmodel4, cls4_threshold,
                              func,
                              config_file);
    // 这里申请一个设备
    // 多芯机器上按需申请多个handle               
    bm_handle_t handle;
    int dev_id = 0;
    int ret = bm_dev_request(&handle, dev_id);
    bm::DataPtr jpeg_data = bm::read_binary(jpg_path);
    void *data_ptr = jpeg_data->ptr<uint8_t>();
    size_t data_size = jpeg_data->size();
#if 0
    char save_path[256];
    snprintf(save_path, 256, "output.bmp");
    call(bm_image_write_to_bmp, image, save_path);
#endif
    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; ++i) {
        uint64_t image_id = instance.Infer((const unsigned char*)data_ptr,
                                           data_size);
    }
    std::this_thread::sleep_for(std::chrono::seconds(1000));
    std::cout << "exit.." << std::endl;
    return 0;
}