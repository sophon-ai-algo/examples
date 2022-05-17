#include <assert.h>
#include <map>
#include <vector>
#include "bmcv_api_ext.h"
#include "ruilai_api.h"
#include <functional>

#define TEST_PICS_NUM 1000
#define EXTRA_PIC_NUM 4
#define FFALIGN(x, a) (((x)+(a)-1)&~((a)-1))

static int counter = 0;
auto start = std::chrono::high_resolution_clock::now();

void call_back(uint64_t image_id, bool ret, float score) {
    if (++counter >= TEST_PICS_NUM) {
        auto end = std::chrono::high_resolution_clock::now();
        auto count = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Finish! " << TEST_PICS_NUM << " images, cost time:" << count << "ms, fps:"
            << 1000.f * TEST_PICS_NUM / count  << std::endl;
        exit(0);
    }
//    std::cout << "image_id: " << image_id << ", "
//              << "ret:      " << ret      << ", "
//              << "score:    " << score    << std::endl;

}

int main(int argc, char* argv[]) {
    // 初始化参数
    if (argc < 6) {
        std::cerr << "e.g: ./ruilai_comp <card_nums> <jpg_path> <retinaface_path> <mobilenetv2_path> <wsdan_path>" << std::endl;
        exit(1);
    }
    int card_num = std::atoi(argv[1]);
    std::string jpg_path = argv[2];
    std::string retinaface_bmodel = argv[3];
    std::string cls_bmodel1 = argv[4];
    std::string cls_bmodel2 = argv[5];
    std::string cls_bmodel3 = argv[5];
    std::string cls_bmodel4 = argv[5];
    float face_threshold = 0.5f;
    float cls1_threshold = 0.5f;
    float cls2_threshold = 0.5f;
    float cls3_threshold = 0.5f;
    float cls4_threshold = 0.5f;
    std::string config_file = "./cameras.json";

    ImgResultCallBackFunc func = std::bind(call_back, std::placeholders::_1,
                                        std::placeholders::_2,
                                        std::placeholders::_3);
    RuiLaiAPIWrapper instance(card_num,
                              retinaface_bmodel, face_threshold,
                              cls_bmodel1, cls1_threshold,
                              cls_bmodel2, cls2_threshold,
                              cls_bmodel3, cls3_threshold,
                              cls_bmodel4, cls4_threshold,
                              func,
                              config_file);

    bm::DataPtr jpeg_data = bm::read_binary(jpg_path);
    void *data_ptr = jpeg_data->ptr<uint8_t>();
    size_t data_size = jpeg_data->size();
#if 0
    char save_path[256];
    snprintf(save_path, 256, "output.bmp");
    call(bm_image_write_to_bmp, image, save_path);
#endif
    start = std::chrono::high_resolution_clock::now();

#if 1
    for (int i = 0; i < TEST_PICS_NUM + EXTRA_PIC_NUM; ++i) {
        uint64_t image_id = instance.Infer((const unsigned char*)data_ptr,
                                           data_size);
    }
#else
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        uint64_t image_id = instance.Infer((const unsigned char*)data_ptr,
                                           data_size);
    }
#endif
    std::this_thread::sleep_for(std::chrono::seconds(1000));
    std::cout << "exit.." << std::endl;
    return 0;
}