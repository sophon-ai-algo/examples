#include <sstream>
#include <string>
#include <numeric>
#include "spdlog/spdlog.h"
#ifndef USE_FFMPEG
#define USE_FFMPEG
#endif
#ifndef USE_BMCV
#define USE_BMCV
#endif
#include "cvwrapper.h"
#include "engine.h"
#include <iostream>
#include <boost/filesystem.hpp>
#include <stdlib.h>
#include <string.h>
#include "processor.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace sail;
namespace fs = boost::filesystem;

#define MAX_BATCH_SIZE 4


std::vector<string> getlabels(std::string label_file){
    std::vector<string> m_class_names;
    m_class_names.clear();
    std::ifstream ifs(label_file);
    if (ifs.is_open()) {
        std::string line;
        while(std::getline(ifs, line)) {
            m_class_names.push_back(line);
        }
    }
    return m_class_names;
}

int main(int argc, char** argv)
{
    if (argc != 9){
        printf("USAGE: \n");
        printf("      %s video <video url> <bmodel path> <name file> <test count> <detect threshold> <nms threshold> <device id>\n",argv[0]);
        exit(1);
    }
    bool is_video = false;
    if (strcmp(argv[1], "video") == 0)
        is_video = true;

    string image_name = argv[2];
    if (!fs::exists(image_name)) {
        printf("Cannot find input file: %s\n",image_name.c_str());
        exit(1);
    }

    fs::path image_file(image_name);
    string name_save = image_file.filename().string();

    printf("*****************************************************\n");
    printf("%s\n",image_name.c_str());
    printf("%s\n",name_save.c_str());
    printf("*****************************************************\n");

    string bmodel_file = argv[3];
    if (!fs::exists(bmodel_file)){
        printf("Can not find valid model file: %s\n",bmodel_file.c_str());
        exit(1);
    }

    string name_file = argv[4];
    if(!fs::exists(name_file)){
        printf("Can not find name file: %s\n",name_file.c_str());
        exit(1);
    }

    std::vector<string> m_class_names = getlabels(name_file);

    unsigned long test_loop = stoul(string(argv[5]), nullptr, 0);
    if(test_loop <= 0){
        printf("test_loop must large 0 !\n");
        exit(1);
    }

    float threshold = atof(argv[6]);                 //0.25
    float nms_threshold = atof(argv[7]);             //0.45
    int device_id = atoi(argv[8]);

    sail::Engine engine(device_id);
    bool return_value = engine.load(bmodel_file);
    if (!return_value)    {
        printf("load model [%s] fialed!\n",bmodel_file.c_str());
        exit(1);
    }
    std::vector<std::string> gh_names = engine.get_graph_names();
    for (size_t i = 0; i < gh_names.size(); i++)
    {
       printf("gh_names[%d]: %s\n",i,gh_names[i].c_str());
    }
    std::string graph_name(gh_names[0]);
    
    std::string input_tensor_name = engine.get_input_names(gh_names[0])[0];
    printf("input_tensor_name: %s\n",input_tensor_name.c_str());
    std::string output_tensor_name = engine.get_output_names(gh_names[0])[0];
    printf("output_tensor_name: %s\n",output_tensor_name.c_str());

    std::vector<int> input_shape = engine.get_input_shape(gh_names[0],input_tensor_name);
    printf("input_shape:[ ");
    for (size_t i = 0; i < input_shape.size(); i++)    {
        printf("%d ",input_shape[i]);
    }
    printf("]\n");
    int input_batch_size = input_shape[0];

    std::vector<int> output_shape = engine.get_output_shape(gh_names[0],output_tensor_name);
    printf("output_shape:[ ");
    for (size_t i = 0; i < output_shape.size(); i++)    {
        printf("%d ",output_shape[i]);
    }
    printf("]\n");

    bm_data_type_t input_dtype = engine.get_input_dtype(gh_names[0],input_tensor_name);
    bool is_fp32 = (input_dtype == BM_FLOAT32);
    bm_data_type_t ouput_dtype = engine.get_output_dtype(gh_names[0],output_tensor_name);
    sail::Handle handle = engine.get_handle();


    sail::Tensor input_tensor(handle,input_shape,input_dtype,false,false);
    sail::Tensor output_tensor(handle,output_shape,ouput_dtype,true,true);

    std::map<std::string,sail::Tensor*> input_tensors = {{input_tensor_name,&input_tensor}}; 
    std::map<std::string,sail::Tensor*> output_tensors = {{output_tensor_name,&output_tensor}}; 

    engine.set_io_mode(graph_name, sail::SYSO);
    // init preprocessor and postprocessor
    sail::Bmcv bmcv(handle);
    auto img_dtype = bmcv.get_bm_image_data_format(input_dtype);
    float scale = engine.get_input_scale(graph_name, input_tensor_name);



    BmcvPreProcessor preprocessor(bmcv, input_shape[input_shape.size()-1], input_shape[input_shape.size()-2], scale);
    sail::Decoder decoder((const string)image_name, true, device_id);

    vector<int> strides;
    strides.push_back(8);
    strides.push_back(16);
    strides.push_back(32);
    YoloX_PostForward postprocessor(input_shape[3], input_shape[2], strides);
    bool status = true;
    // pipeline of inference
    for (int i = 0; i < test_loop; ++i) {
        sail::BMImageArray<4> imgs_0;
        sail::BMImageArray<4> imgs_1(handle, input_shape[2], input_shape[3],
                                 FORMAT_BGR_PLANAR, img_dtype);
        // read 4 images from image files or a video file
        bool flag = false;
        std::vector<std::pair<int,int>> ost_size_list;
        for (int j = 0; j < input_batch_size; ++j) {
            int ret = decoder.read_(handle, imgs_0[j]);
            if (ret != 0) {
                printf("Read the End!\n");
                flag = true;
                break;
            }
            ost_size_list.push_back(std::pair<int,int>(imgs_0[j].width,imgs_0[j].height));
        }
        if (flag) {
            break;
        }
        double m_time = cv::getTickCount();
        preprocessor.process(imgs_0,imgs_1);
        bmcv.bm_image_to_tensor(imgs_1, input_tensor);
        engine.process(graph_name, input_tensors, output_tensors);
        float* output_data = reinterpret_cast<float*>(output_tensor.sys_data());
        vector<vector<ObjRect>> detections;
        postprocessor.process(output_data,output_shape,ost_size_list,threshold,nms_threshold,detections);
        printf("time use: %.2f ms\n",(cv::getTickCount()-m_time)/cv::getTickFrequency()*1000);
        for (int m=0;m<detections.size();++m){
            for(int n=0;n<detections[m].size();++n){
            bmcv.rectangle(imgs_0[m], int(detections[m][n].left), int(detections[m][n].top), 
                int(detections[m][n].width), int(detections[m][n].height), std::make_tuple(255, 0, 0), 3);
            }

            char save_name[256]={0};
            sprintf(save_name,"loop-%d-batch-%d-dev-%d.jpg",i,m,device_id);
            bmcv.imwrite(std::string(save_name),imgs_0[m]);
        }

    }
    return 0;
}