//
// Created by xwang on 2/10/22.
//

#include <thread>
#include <chrono>
#include "opencv2/opencv.hpp"

int main(int argc, char *argv[]) {
    cv::VideoCapture cap1, cap2, cap3, cap4;
    cap1.open("rtsp://admin:hk123456@11.73.12.20", cv::CAP_ANY, 0);
    cap2.open("rtsp://admin:hk123456@11.73.12.22", cv::CAP_ANY, 0);
    cap3.open("rtsp://admin:hk123456@11.73.12.23", cv::CAP_ANY, 0);
    cap4.open("rtsp://admin:hk123456@11.73.12.20", cv::CAP_ANY, 0);

    assert(cap1.isOpened() && cap2.isOpened() && cap3.isOpened() &&cap4.isOpened());

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
        
        cv::Mat *frame1 = new cv::Mat,
                *frame2 = new cv::Mat,
                *frame3 = new cv::Mat,
                *frame4 = new cv::Mat;
        cap1.read(*frame1);
        cap2.read(*frame2);
        cap3.read(*frame3);
        cap4.read(*frame4);


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
        delete frame2;
        delete frame3;
        delete frame4;

        std::cout << "frame " << frame_count++ << std::endl;
        std::this_thread::sleep_for(std::chrono::microseconds(30));
    }

    return 0;
}
