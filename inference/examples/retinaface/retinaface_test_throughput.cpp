#include <glog/logging.h>
#include <fstream>
#include <atomic>
#include "macros.h"
#include "bmutility.h"
#include "retinaface.h"

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    google::LogToStderr();
    bmlib_log_set_level(BMLIB_LOG_VERBOSE);

    if (argc != 4)
    {
        LOG(INFO) << argv[0] << " <.bmodel> <image> <round>";
        return 1;
    }

    int dev_id = 0;
    std::string bmodel_file(argv[1]), image_path(argv[2]);
    int round = std::stoi(argv[3]);
    bm::BMNNHandlePtr handle = std::make_shared<bm::BMNNHandle>(dev_id);
    bm::BMNNContextPtr context = std::make_shared<bm::BMNNContext>(handle, bmodel_file);
    std::shared_ptr<bm::BMNNNetwork> net = context->network(0);
    LOG(INFO) << *net;
    bool keep_original = true;
    float nms_threshold = 0.4;
    float conf_threshold = 0.02;
    bm::Watch w;
    const char *net_name = "";
#if 0
    size_t target_size = 1600;
    auto retinaface = std::make_shared<RetinafaceEval>(
        context, target_size, keep_original,
        nms_threshold, conf_threshold);
#else
    auto retinaface = std::make_shared<Retinaface>(
        context, keep_original,
        nms_threshold, conf_threshold, net_name, &w);
#endif

    std::atomic<int> process_index(0);
    retinaface->set_detected_callback(
        [&](bm::FrameInfo &frameInfo) {
            int num = frameInfo.frames.size();
            for (int i = 0; i < num; ++i)
            {
                LOG(INFO) << "process_index " << process_index;
                ++process_index;
            }
        });
    bm::BMInferencePipe<bm::FrameBaseInfo, bm::FrameInfo> pipeline;
    bm::DetectorParam param;
    int cpu_num = std::thread::hardware_concurrency();
    int tpu_num = 1;
    param.preprocess_thread_num = 1;
    param.preprocess_queue_size = 5;
    param.inference_thread_num = tpu_num;
    param.inference_queue_size = 8;
    param.postprocess_thread_num = 1;
    param.postprocess_queue_size = 5;
    pipeline.init(param, retinaface);

    size_t input_index = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < round; ++i)
    {
        w.mark("decode");
        bm::FrameBaseInfo fbi;
        fbi.filename = image_path;
        retinaface->read_image(fbi);
        w.mark("decode");
        pipeline.push_frame(&fbi);
        LOG(INFO) << "input_index " << input_index;
        ++input_index;
    }

    pipeline.flush_frame();
    while (input_index > process_index)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    LOG(INFO) << w;
    LOG(INFO) << "Total " << input_index << " images. Average "
              << (ms / input_index) << "ms";

    return 0;
}
