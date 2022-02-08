#include "yolox.hpp"
#include <boost/filesystem.hpp>
#include <iostream>
#include <string>
#include <math.h>

using namespace std;
namespace fs = boost::filesystem;

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
        printf("      %s image <image file> <bmodel path> <name file> <test count> <detect threshold> <nms threshold> <device id>\n",argv[0]);
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
    int device_count = 0;
    bm_dev_getcount(&device_count);
    printf("device_count: %d\n",device_count);
    if(device_id >= device_count){
        printf("ERROR: Input device id=%d exceeds the maximum number %d\n",device_id,device_count);
        exit(1);
    }

    vector<int> strides;
    strides.push_back(8);
    strides.push_back(16);
    strides.push_back(32);

 
    bm_handle_t bm_handle;
    bm_status_t ret = bm_dev_request(&bm_handle,device_id);
    if(ret != BM_SUCCESS){
        printf("bm_dev_request Fialed! ret = %d\n",ret);
        exit(1);
    }

      // profiling
    TimeStamp yolox_ts;
    TimeStamp *ts = &yolox_ts;

    YOLOX net(bm_handle, bmodel_file, strides);

    // for profiling
    net.enableProfile(ts);
    int batch_size =  net.getInputBatchSize();
    if(!is_video){
        for (size_t loop_idx = 0; loop_idx < test_loop; loop_idx++)      {
            vector<string> image_name_list ;
            vector<cv::Mat> images;
            vector<cv::Mat> images_rgb;
            vector<vector<ObjRect>> detections;
            vector<bm_image> input_img_bmcv;
            for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx++)        {
                image_name_list.push_back(image_name);
                cv::Mat image_input = cv::imread(image_name);
                if(image_name.empty()){
                    printf("Can not open picture: %s\n",image_name.c_str());
                    exit(1);
                }
                images.push_back(image_input);
            }
            double m_time_start = cv::getTickCount();
            bm_image_from_mat(bm_handle, images, input_img_bmcv);
            if(net.preForward (input_img_bmcv) == 0){
                // do inference
                net.forward(threshold,nms_threshold);
                net.postForward (input_img_bmcv , detections);
                printf("bm_inference Time use %.2f ms\n",(cv::getTickCount()-m_time_start)/cv::getTickFrequency()*1000);

                for (size_t idx_resu = 0; idx_resu < detections.size(); idx_resu++)    {
                    cv::Mat image_save = images[idx_resu].clone();
                    for (size_t obj_idx = 0; obj_idx < detections[idx_resu].size(); obj_idx++)        {
                        cv::rectangle(image_save, cv::Rect(detections[idx_resu][obj_idx].left, detections[idx_resu][obj_idx].top, 
                                detections[idx_resu][obj_idx].right - detections[idx_resu][obj_idx].left,
                                detections[idx_resu][obj_idx].bottom - detections[idx_resu][obj_idx].top), cv::Scalar(0, 0, 255), 2);

                        char txt_put[128]={0};
                        sprintf(txt_put,"%s: %.3f",m_class_names[detections[idx_resu][obj_idx].class_id].c_str(),detections[idx_resu][obj_idx].score);

                        cv::putText(image_save, txt_put, cv::Point(detections[idx_resu][obj_idx].left, detections[idx_resu][obj_idx].top),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
                    }

                    // check result directory
                    if (!fs::exists("./results")) {
                        fs::create_directory("results");
                    }
                    
                    char save_name[256]={0};
                    if (net.getPrecision()) 
                        sprintf(save_name,"results/loop-%d-batch-%d-int8-dev-%d-%s",loop_idx,idx_resu,device_id,name_save.c_str());
                    else
                        sprintf(save_name,"results/loop-%d-batch-%d-fp32-dev-%d-%s",loop_idx,idx_resu,device_id,name_save.c_str());
                    printf("Save image: %s\n",save_name);
                    cv::imwrite(save_name,image_save);
                }
            }
            // destory bm_image
            for (size_t i = 0; i < input_img_bmcv.size();i++) {
                bm_image_destroy (input_img_bmcv[i]);
            }
        }
        
    }else{
        cv::VideoCapture cap(image_name);
        if (!cap.isOpened()) {
            printf("open stream %s failed!\n",image_name);
            exit(1);
        }
        for (size_t loop_idx = 0; loop_idx < test_loop; loop_idx++)      {
            vector<bm_image> input_img_bmcv;
            vector<cv::Mat> input_img_cv;
            for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx++)        {
               cv::Mat image_capture;
               cap.read(image_capture);
               if(image_capture.empty()){
                    cap.release();
                   exit(1);
               }
               input_img_cv.push_back(image_capture);
            }

            double m_time_start = cv::getTickCount();
            bm_image_from_mat(bm_handle, input_img_cv, input_img_bmcv);

            vector<vector<ObjRect>> detections;
            if(net.preForward (input_img_bmcv) == 0){
                // do inference
                net.forward(threshold,nms_threshold);
                net.postForward (input_img_bmcv , detections);
                printf("bm_inference Time use %.2f ms\n",(cv::getTickCount()-m_time_start)/cv::getTickFrequency()*1000);

                for (size_t idx_resu = 0; idx_resu < detections.size(); idx_resu++)    {
                    cv::Mat image_save = input_img_cv[idx_resu].clone();
                    for (size_t obj_idx = 0; obj_idx < detections[idx_resu].size(); obj_idx++)        {
                        cv::rectangle(image_save, cv::Rect(detections[idx_resu][obj_idx].left, detections[idx_resu][obj_idx].top, 
                                detections[idx_resu][obj_idx].right - detections[idx_resu][obj_idx].left,
                                detections[idx_resu][obj_idx].bottom - detections[idx_resu][obj_idx].top), cv::Scalar(0, 0, 255), 2);

                        char txt_put[128]={0};
                        sprintf(txt_put,"%s: %.3f",m_class_names[detections[idx_resu][obj_idx].class_id].c_str(),detections[idx_resu][obj_idx].score);

                        cv::putText(image_save, txt_put, cv::Point(detections[idx_resu][obj_idx].left, detections[idx_resu][obj_idx].top),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
                    }

                    // check result directory
                    if (!fs::exists("./results")) {
                        fs::create_directory("results");
                    }
                    
                    char save_name[256]={0};
                    if (net.getPrecision()) 
                        sprintf(save_name,"results/loop-%d-batch-%d-int8-dev-%d-video.jpg",loop_idx,idx_resu,device_id);
                    else
                        sprintf(save_name,"results/loop-%d-batch-%d-fp32-dev-%d-video.jpg",loop_idx,idx_resu,device_id);
                    printf("Save image: %s\n",save_name);
                    cv::imwrite(save_name,image_save);
                }
            }

            for (size_t i = 0; i < input_img_bmcv.size();i++) {
                bm_image_destroy (input_img_bmcv[i]);
            }
        }
        cap.release();
    }
    return 0;
}

int main_tez()
{
    int device_id = 0;
    float threshold = 0.25;
    // float threshold = 0.9;
    // float threshold = 0.0;
    float nms_threshold = 0.45;
    vector<int> strides;
    strides.push_back(8);
    strides.push_back(16);
    strides.push_back(32);
    // const std::string model_name = "/workspace/YOLOX/models/yolox_s_fp32_batch1/compilation.bmodel";
    const std::string model_name = "/workspace/YOLOX/models/yolox_m/int8model/compilation_1.bmodel";
    const std::string image_name_0 = "/workspace/YOLOX/data/val2014/COCO_val2014_000000576564.jpg";
    const std::string image_name_1 = "/workspace/YOLOX/data/val2014/COCO_val2014_000000000073.jpg";
    const std::string image_name_2 = "/workspace/YOLOX/data/val2014/COCO_val2014_000000000074.jpg";
    const std::string image_name_3 = "/workspace/YOLOX/data/val2014/COCO_val2014_000000578255.jpg";
    // const  std::string model_name = "/workspace/examples/SSD_object/model/out/fp32_ssd300.bmodel";
    if( !fs::exists(model_name)){
        printf("Can not find file: %s\n",model_name.c_str());
        return -1;
    }
    int device_count = 0;
    bm_dev_getcount(&device_count);
    printf("device_count: %d\n",device_count);
    if(device_id >= device_count){
        printf("ERROR: Input device id=%d exceeds the maximum number %d\n",device_id,device_count);
        return -2;
    }

    bm_handle_t bm_handle;
    bm_status_t ret = bm_dev_request(&bm_handle,device_id);
    if(ret != BM_SUCCESS){
        printf("bm_dev_request Fialed! ret = %d\n",ret);
        return -3;
    }

      // profiling
    TimeStamp ssd_ts;
    TimeStamp *ts = &ssd_ts;

    YOLOX net(bm_handle, model_name, strides);

    // for profiling
    net.enableProfile(ts);

    cv::Mat image0 = cv::imread(image_name_0);
    cv::Mat image1 = cv::imread(image_name_1);
    cv::Mat image2 = cv::imread(image_name_2);
    cv::Mat image3 = cv::imread(image_name_3);
    vector<vector<ObjRect>> detections;
    vector<cv::Mat> images;
    images.push_back (image3);
    // images.push_back (image2);
    // images.push_back (image1);
    // images.push_back (image0);

    vector<bm_image> input_img_bmcv;
    for (int loop_idx = 0; loop_idx < 100; loop_idx++)
    {
        /* code */
        input_img_bmcv.clear();
        // ts->save("attach input");
        // double m_time_start = cv::getTickCount();
        bm_image_from_mat(bm_handle, images, input_img_bmcv);
        // ts->save("attach input");


        // ts->save("detection");
        if(net.preForward (input_img_bmcv) == 0){
    // do inference
            net.forward(threshold,nms_threshold);

            net.postForward (input_img_bmcv , detections);

            // printf("%d:  Time use %.2f ms\n",loop_idx,(cv::getTickCount()-m_time_start)/cv::getTickFrequency()*1000);


            ts->save("detection");
            for (size_t i = 0; i < detections.size(); i++)    {
                cv::Mat image_save = images[i].clone();
                for (size_t j = 0; j < detections[i].size(); j++)        {
                        cv::rectangle(image_save, cv::Rect(detections[i][j].left, detections[i][j].top, 
                            detections[i][j].right - detections[i][j].left,
                            detections[i][j].bottom - detections[i][j].top), cv::Scalar(0, 0, 255), 2);

                        char txt_put[128]={0};
                        sprintf(txt_put,"%d: %.3f",detections[i][j].class_id,detections[i][j].score);

                        cv::putText(image_save, txt_put, cv::Point(detections[i][j].left, detections[i][j].top),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
                }
                
                char save_name[256]={0};
                sprintf(save_name,"save_name_idx_%d.jpg",i);
                cv::imwrite(save_name,image_save);
            }
        }
    }

    // destory bm_image
    for (size_t i = 0; i < input_img_bmcv.size();i++) {
        bm_image_destroy (input_img_bmcv[i]);
    }
    return 0;
}

