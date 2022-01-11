//
// Created by yuan on 12/6/21.
//

#ifndef INFERENCE_FRAMEWORK_FACE_COMMON_H
#define INFERENCE_FRAMEWORK_FACE_COMMON_H

namespace bm {
    struct FrameBaseInfo2 {
        int chan_id;
        uint64_t seq;
        // origin data
        AVPacket *avpkt;
        AVFrame *avframe;
        bm::DataPtr jpeg_data;
        bm_image origin_image;
        int width, height;
        bool skip;
        int model_type;

        // for next network
        std::vector<bm_image> crop_bmimgs;

        FrameBaseInfo2() : chan_id(0), seq(0), avpkt(nullptr), avframe(nullptr), jpeg_data(nullptr),
                           skip(false), model_type(0) {

        }

        FrameBaseInfo2(const FrameBaseInfo2 &other) {
            chan_id = other.chan_id;
            seq = other.seq;
            avpkt = other.avpkt;
            avframe = other.avframe;
            jpeg_data = other.jpeg_data;
            width = other.width;
            height = other.height;
            skip = other.skip;
            model_type = other.model_type;
            origin_image = other.origin_image;
            crop_bmimgs = other.crop_bmimgs;
        }
    };

    struct NetForward {
        int batch_size;
        std::vector<bm_tensor_t> input_tensors;
        std::vector<bm_tensor_t> output_tensors;

        NetForward() {
            batch_size = 0;
        }
    };

    struct FrameInfo2 {
        //AVFrame based, 4batch frames.size=4
        std::vector<FrameBaseInfo2> frames;
        std::vector<NetForward> forwards;
        std::vector<bm::NetOutputDatum> out_datums;
    };
}

#endif //INFERENCE_FRAMEWORK_FACE_COMMON_H
