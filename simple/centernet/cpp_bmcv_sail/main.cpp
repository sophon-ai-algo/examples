#include <iostream>
#include <sstream>
#include <string>
#include <numeric>
#include <chrono>
#include <stdlib.h>
#include <string.h>
#include "boost/filesystem.hpp"
#include "spdlog/spdlog.h"
#include "cvwrapper.h"
#include "engine.h"
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
#include "processor.h"

// 全局变量
// 记录模型信息
std::vector<std::string> gh_names;       // graph名
std::vector<std::string> input_names;    // 网络输入名
std::vector<int>         input_shape;    // 输入shape
std::vector<std::string> output_names;   // 网络输出名
std::vector<int>         output_shape;   // 输出shape
bm_data_type_t           input_dtype;    // 输入数据类型
bm_data_type_t           output_dtype;   // 输出数据类型

// 初始化Glog
static void InitGlog(const char* cmd) {
    FLAGS_logbufsecs = 0;
    FLAGS_logtostderr = 1;
    google::InitGoogleLogging(cmd);
}

// 输出bmodel输出层信息
void PrintModelInputInfo(sail::Engine& engine) {
    // graph名
    gh_names = engine.get_graph_names();
    std::string gh_info;
    for_each(gh_names.begin(), gh_names.end(), [&](std::string& s) {
        gh_info += "0: " + s + "; ";
    });
    LOG(INFO) << "grapgh name -> " << gh_info;

    // 网络输入名
    input_names = engine.get_input_names(gh_names[0]);
    assert(input_names.size() > 0);
    std::string input_tensor_names;
    for_each(input_names.begin(), input_names.end(), [&](std::string& s) {
        input_tensor_names += "0: " + s + "; ";
    });
    LOG(INFO) << "net input name -> " << input_tensor_names;

    // 网络输出名字
    output_names = engine.get_output_names(gh_names[0]);
    assert(output_names.size() > 0);
    std::string output_tensor_names;
    for_each(output_names.begin(), output_names.end(), [&](std::string& s) {
        output_tensor_names += "0: " + s + "; ";
    });
    LOG(INFO) << "net output name -> " << output_tensor_names;

    // 网络输入尺寸
    input_shape = engine.get_input_shape(gh_names[0], input_names[0]);
    std::string input_tensor_shape;
    for_each(input_shape.begin(), input_shape.end(), [&](int s) {
        input_tensor_shape += std::to_string(s) + " ";
    });
    LOG(INFO) << "input tensor shape -> " << input_tensor_shape;
    
    if (input_shape[0] > 1) {
        LOG(FATAL) << "此cppDemo暂时支持1batch模型, 多batch的请查看python例程" << input_tensor_shape;
        exit(0);
    }

    // 网络输出尺寸
    output_shape = engine.get_output_shape(gh_names[0], output_names[0]);
    std::string output_tensor_shape;
    for_each(output_shape.begin(), output_shape.end(), [&](int s) {
        output_tensor_shape += std::to_string(s) + " ";
    });
    LOG(INFO) << "output tensor shape -> " << output_tensor_shape;

    // 网络输入数据类型
    input_dtype = engine.get_input_dtype(gh_names[0], input_names[0]);
    LOG(INFO) << "input dtype -> "<< input_dtype << ", is fp32=" << ((input_dtype == BM_FLOAT32) ? "true" : "false");

    // 网络输出数据类型
    output_dtype = engine.get_output_dtype(gh_names[0], output_names[0]);
    LOG(INFO) << "input dtype -> "<< output_dtype << ", is fp32=" << ((input_dtype == BM_FLOAT32) ? "true" : "false");
}



int main(int argc, char** argv) {
    const char *keys="{ bmodel | /workspace/examples/centernet_test/CenterNet_object/data/models/ctdet_coco_dlav0_1x_fp32.bmodel | bmodel file path}"
                     "{ tpu_id | 0    | TPU device id}"
                     "{ conf   | 0.35 | confidence threshold for filter boxes}"
                     "{ image  | /workspace/examples/centernet_test/CenterNet_object/data/ctdet_test.jpg | input stream file path}"
                     "{ help   | 0    | Print help information.}";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }
    // Glog初始化
    InitGlog(argv[0]);
    // 模型路径
    std::string bmodel_file = parser.get<std::string>("bmodel");
    // 图片名
    std::string image_file  = parser.get<std::string>("image");
    // 设备号
    int tpu_id              = parser.get<int>("tpu_id");
    // 置信度
    float confidence        = parser.get<float>("conf");
    // 测试次数
    int test_loop           = 1;

    // bmodel不存在
    if (!boost::filesystem::exists(bmodel_file)) {
        LOG(ERROR) << "Invalid bmodel file: " << bmodel_file;
        exit(1);
    }
    // 生成engine，加载模型
    sail::Engine engine(tpu_id);
    if (!engine.load(bmodel_file)) {
        // 加载模型失败
        LOG(ERROR) << "Engine load bmodel "<< bmodel_file << "failed";
        exit(0);
    }

    // 输出模型信息
    PrintModelInputInfo(engine);

    sail::Handle handle = engine.get_handle();
    sail::Tensor input_tensor(handle,  input_shape,  input_dtype,  true, true);
    sail::Tensor output_tensor(handle, output_shape, output_dtype, true,  true);

    std::map<std::string, sail::Tensor*> input_tensors  = {{ input_names[0],  &input_tensor}}; 
    std::map<std::string, sail::Tensor*> output_tensors = {{ output_names[0], &output_tensor}}; 

    engine.set_io_mode(gh_names[0], sail::SYSIO);
    sail::Bmcv bmcv(handle);

    // 根据网络输入类型确定网络图片输入类型
    bm_image_data_format_ext img_dtype = bmcv.get_bm_image_data_format(input_dtype);

    CenterNetPreprocessor preprocessor(bmcv, input_shape[3], input_shape[2], 
                                       engine.get_input_scale(gh_names[0], input_names[0]));
    sail::Decoder decoder((const string)image_file, true, tpu_id);
    CenterNetPostprocessor postprocessor(output_shape, confidence);

    // 网络输入的batch
    int input_batch_size = input_shape[0];

    for (int i = 0; i < test_loop; i++) {
        // sail::BMImageArray<4> imgs_0;
        // sail::BMImageArray<4> imgs_1(handle, input_shape[2], input_shape[3],
        //                              FORMAT_BGR_PLANAR, img_dtype);
        sail::BMImage imgs_0;
        sail::BMImage imgs_1(handle, input_shape[2], input_shape[3],
                             FORMAT_BGR_PLANAR, img_dtype);
        // read 4 images from image files or a video file
        bool flag = false;
        std::vector<std::pair<int,int>> ost_size_list;
        for (int j = 0; j < input_batch_size; ++j) {
            imgs_0 = decoder.read(handle);
            // if (ret != 0) {
            //     LOG(INFO) << "Read the End!";
            //     flag = true;
            //     break;
            // }
            // ost_size_list.push_back(std::pair<int,int>(imgs_0[j].width, imgs_0[j].height));
        }
        // if (flag) {
        //     break;
        // }
        sail::BMImage rgb_img = bmcv.convert_format(imgs_0);

        LOG(INFO) << "Preprocess begin";
        preprocessor.Process(rgb_img, imgs_1);
        bmcv.bm_image_to_tensor(imgs_1, input_tensor);
        // float* sys_data = reinterpret_cast<float*>(input_tensor.sys_data());

        // preprocessor.BGRNormalization(sys_data);
        // input_tensor.sync_s2d();
        LOG(INFO) << "Preprocess end";

        LOG(INFO) << "Inference begin";
        engine.process(gh_names[0], input_tensors, output_tensors);
        LOG(INFO) << "Inference end";

        LOG(INFO) << "Postprocess begin";
        float* output_data = reinterpret_cast<float*>(output_tensor.sys_data());
        postprocessor.Process(output_data);
        std::shared_ptr<std::vector<BMBBox>> pVectorBBox =
            postprocessor.CenternetCorrectBBox(rgb_img.height(), rgb_img.width());
        LOG(INFO) << "Postprocess end";

        for (auto iter = pVectorBBox->begin(); iter != pVectorBBox->end(); iter++) {
            LOG(INFO) << "Got one object, confidence:" << iter->conf;
            bmcv.rectangle(rgb_img, iter->x, iter->y, 
                            iter->w, iter->h, std::make_tuple(255, 0, 0), 3);
        }
        // save result
        auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::stringstream ss;
        ss << std::put_time(std::localtime(&t), "%Y-%m-%d-%H-%M-%S");
        bmcv.imwrite("ctdet_result_" + ss.str() + ".jpg", rgb_img);
        LOG(INFO) << "save result";

    }


    return 0;
}