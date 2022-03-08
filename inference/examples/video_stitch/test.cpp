//
// Created by xwang on 2/10/22.
//

#include <thread>
#include <chrono>
#include "opencv2/opencv.hpp"
#include "bmutility_timer.h"


void test_bmimage() {
    bm_handle_t handle;
    bm_dev_request(&handle, 0);
    bm_image stitch_image;

    bm_image_create(handle, 1080, 1920, FORMAT_YUV420P, DATA_TYPE_EXT_1N_BYTE, &stitch_image);
    bm_image_alloc_dev_mem(stitch_image, BMCV_HEAP0_ID);
    cv::Mat output_mat;
    cv::bmcv::toMAT(&stitch_image, output_mat, false);
    bm_image_destroy(stitch_image);
}
int main(int argc, char *argv[]) {
#if 1
    int i = 0;
    while (i++ < 1000) {
        bm::BMPerf perf;
        perf.begin("test", 0);
        test_bmimage();
        perf.end();
    }
#else
    cv::VideoCapture cap1;
    //cap1 = new cv::VideoCapture;



    cap1.open("rtsp://admin:hk123456@11.73.12.22", cv::CAP_ANY, 0);


    assert(cap1.isOpened());
    cap1.set(cv::CAP_PROP_OUTPUT_YUV, PROP_TRUE);

    int fps    = cap1.get(cv::CAP_PROP_FPS);
    int height = (int)cap1.get(cv::CAP_PROP_FRAME_HEIGHT);
    int width  = (int)cap1.get(cv::CAP_PROP_FRAME_WIDTH);
    cv::Mat stitch_image;
    stitch_image.cols = 7680;
    stitch_image.rows = 1080;
    std::vector<cv::Mat>  in;
    std::vector<cv::Rect> srt;
    std::vector<cv::Rect> drt;
    cv::Rect rt;
    int frame_count = 0;

    while (true) {
        in.clear();
        srt.clear();
        drt.clear();
        
        cv::Mat *frame1 = new cv::Mat;

        cap1.read(*frame1);



//        in.emplace_back(*frame1);
//        in.emplace_back(*frame2);
//        in.emplace_back(*frame3);
//        in.emplace_back(*frame4);
//
//
//        //in.push_back(cv::imread("/inference_framework/release/video_stitch_demo/result.jpg"));
//
//        for (int i = 0; i < 4; ++i) {
//            rt.x = rt.y = 0;
//            rt.width  = 1920;
//            rt.height = 1080;
//            srt.push_back(rt);
//
//            // rt.x = (i % 2) * (stitch_image.cols / 2);
//            // rt.width = stitch_image.cols/2;
//            // rt.y = (i / 2) * (stitch_image.rows / 2);
//            // rt.height = stitch_image.rows / 2;
//            rt.x = (i % 4) * (stitch_image.cols / 4);
//            rt.width = stitch_image.cols / 4;
//            rt.y = 0;
//            rt.height = 1080;
//            drt.push_back(rt);
//        }
//
//        if (BM_SUCCESS != cv::bmcv::stitch(in, srt, drt, stitch_image, true, BMCV_INTER_LINEAR)) {
//            std::cerr << "Stitch error" << std::endl;
//            continue;
//        }
//
//        cv::imwrite("stitch-result.jpg", stitch_image);
        delete frame1;


        std::cout << "frame " << frame_count++ << std::endl;
        std::this_thread::sleep_for(std::chrono::microseconds(30));
    }
#endif
    return 0;
}
