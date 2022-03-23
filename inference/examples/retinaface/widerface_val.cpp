#include <glog/logging.h>
#include <fstream>
#include <atomic>
#include <experimental/filesystem>
#include "macros.h"
#include "bmutility.h"
#include "retinaface.h"

namespace fs = std::experimental::filesystem;

bm::DataPtr read_binary(const std::string &fn)
{
    std::ifstream in(fn, std::ifstream::binary | std::ifstream::ate);
    int size = in.tellg();
    if (size < 0)
    {
        LOG(ERROR) << "failed to read " << fn;
        throw std::runtime_error("io failed");
    }
    auto data = std::make_shared<bm::Data>(new uint8_t[size], size);
    in.seekg(0);
    in.read(data->ptr<char>(), size);
    return data;
}

void create_directory(const fs::path &dir)
{
    if (fs::exists(dir)) return;
    if (!fs::create_directory(dir))
    {
        LOG(ERROR) << "failed to create dir \""
                   << dir
                   << "\"";
        throw std::runtime_error("os error");
    }
}

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    google::LogToStderr();
    //bmlib_log_set_level(BMLIB_LOG_VERBOSE);

    if (argc != 4)
    {
        LOG(INFO) << argv[0] << " <.bmodel> <val> <output>";
        return 1;
    }

    int dev_id = 0;
    std::string bmodel_file(argv[1]), val_path(argv[2]), output_dir(argv[3]);
    bm::BMNNHandlePtr handle = std::make_shared<bm::BMNNHandle>(dev_id);
    bm::BMNNContextPtr context = std::make_shared<bm::BMNNContext>(handle, bmodel_file);
    std::shared_ptr<bm::BMNNNetwork> net = context->network(0);
    LOG(INFO) << *net;
    bool keep_original = false;
    float nms_threshold = 0.4;
    float conf_threshold = 0.02;
#if 0
    size_t target_size = 1600;
    auto retinaface = std::make_shared<RetinafaceEval>(
        context, target_size, keep_original,
        nms_threshold, conf_threshold);
#else
    auto retinaface = std::make_shared<Retinaface>(
        context, keep_original,
        nms_threshold, conf_threshold);
#endif

    std::atomic<int> process_index(0);
    retinaface->set_detected_callback(
        [&](bm::FrameInfo &frameInfo) {
            fs::create_directory(output_dir);
            int num = frameInfo.frames.size();
            for (int i = 0; i < num; ++i)
            {
                fs::path path(frameInfo.frames[i].filename);
                auto dir = output_dir / path.parent_path().filename();
                fs::create_directory(dir);
                auto out_fn = (dir / path.filename().replace_extension("txt")).string();
                std::ofstream out(out_fn);
                out << path.filename().stem().string() << std::endl;
                std::vector<bm::NetOutputObject> objs = frameInfo.out_datums[i].obj_rects;
                auto &frame = frameInfo.frames[i];
                out << objs.size() << std::endl;
                for (const auto &obj : objs)
                {
                    int x = obj.x1;
                    int y = obj.y1;
                    int w = obj.x2 - obj.x1 + 1;
                    int h = obj.y2 - obj.y1 + 1;
                    if (frame.width != frame.original_width)
                    {
                        LOG(INFO) << path << " recover from "
                                  << frame.width << "x" << frame.height
                                  << " to "
                                  << frame.original_width
                                  << "x"
                                  << frame.original_height;
                        float factor = frame.original_width / frame.width;
                        x = round(x * factor);
                        y = round(y * factor);
                        w = round(w * factor);
                        h = round(h * factor);
                    }
                    out << x << " "
                        << y << " "
                        << w << " "
                        << h << " "
                        << obj.score << std::endl;
                }
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
    for (auto &dir : fs::directory_iterator(val_path))
    {
        // dir represents a class
        for (auto &jpg : fs::directory_iterator(dir))
        {
            bm::FrameBaseInfo fbi;
            fbi.filename = jpg.path().string();
            pipeline.push_frame(&fbi);
            LOG(INFO) << "input_index " << input_index;
            ++input_index;
        }
    }

    pipeline.flush_frame();
    while (input_index > process_index)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    LOG(INFO) << "Total " << input_index << " images. Average "
              << (ms / input_index) << "ms";

    return 0;
}
