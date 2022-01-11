#include <glog/logging.h>
#include "macros.h"
#include "bmutility.h"
#include "retinaface.h"

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    google::LogToStderr();
    bmlib_log_set_level(BMLIB_LOG_VERBOSE);

    if (argc != 3)
    {
        LOG(INFO) << argv[0] << " <.bmodel> <val>";
        return 1;
    }

    int dev_id = 0;
    std::string bmodel_file(argv[1]), val_path(argv[2]);
    bm::BMNNHandlePtr handle = std::make_shared<bm::BMNNHandle>(dev_id);
    bm::BMNNContextPtr context = std::make_shared<bm::BMNNContext>(handle, bmodel_file);
    std::shared_ptr<bm::BMNNNetwork> net = context->network(0);
    LOG(INFO) << *net;
    bool keep_original = true;
    float nms_threshold = 0.4;
    float conf_threshold = 0.6;
#if 1
    Retinaface retinaface(context, keep_original, nms_threshold, conf_threshold);
#else
    const size_t target_size = 1600;
    RetinafaceEval retinaface(context, target_size, keep_original);
#endif
    for (int i = 0; i < 1000; ++i)
    {
        std::vector<bm::FrameBaseInfo> frames(1);
        auto &frame = frames[0];
        frame.filename = val_path;
        std::vector<bm::FrameInfo> frame_infos;
        retinaface.preprocess(frames, frame_infos);
        retinaface.forward(frame_infos);
        retinaface.postprocess(frame_infos);
        std::vector<bm::NetOutputObject> objs = frame_infos[0].out_datums[0].obj_rects;
        std::vector<bmcv_rect_t> rects, pt_rects;
        LOG(INFO) << "image size "
                  << frame.original.width << "x"
                  << frame.original.height;
        for (const auto &obj : objs)
        {
            int x = obj.x1;
            int y = obj.y1;
            int w = obj.x2 - obj.x1 + 1;
            int h = obj.y2 - obj.y1 + 1;
            LOG(INFO) << w << "x" << h << "@(" << x << "," << y << ")";
            rects.push_back({x, y, w, h});
            for (int i = 0; i < 5; ++i)
            {
                int x = obj.landmark.x[i];
                int y = obj.landmark.y[i];
                const int stroke = 2;
                pt_rects.push_back({
                    x - (stroke + 1) / 2,
                    y - (stroke + 1) / 2,
                    stroke,
                    stroke});
            }
        }
        const size_t line_width = 2;
        const uint8_t r_value = 0;
        const uint8_t g_value = 255;
        const uint8_t b_value = 0;
        call(
            bmcv_image_draw_rectangle, context->handle(),
            frame.original, rects.size(), rects.data(), line_width,
            r_value, g_value, b_value);
        call(
            bmcv_image_fill_rectangle, context->handle(),
            frame.original, pt_rects.size(), pt_rects.data(),
            r_value, g_value, b_value);
        LOG(INFO) << "got " << objs.size() << " faces";
        call(bm_image_write_to_bmp, frame.original, "output.bmp");
    }
    return 0;
}
