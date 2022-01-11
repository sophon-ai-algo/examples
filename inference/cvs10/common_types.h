//
// Created by yuan on 11/24/21.
//

#ifndef INFERENCE_FRAMEWORK_COMMON_TYPES_H
#define INFERENCE_FRAMEWORK_COMMON_TYPES_H



namespace bm {
    struct FeatureFrame {
        cv::Mat img;
        int chan_id;
        uint64_t seq;

        FeatureFrame():chan_id(0),seq(0) {


        }

        FeatureFrame(const struct FeatureFrame& rf)
        {
            img = rf.img;
            chan_id = rf.chan_id;
            seq = rf.seq;
        }

        FeatureFrame(struct FeatureFrame&& rf)
        {
            img = rf.img;
            chan_id = rf.chan_id;
            seq = rf.seq;
        }

        bm::FeatureFrame& operator =(const bm::FeatureFrame& rf)
        {
            img = rf.img;
            chan_id = rf.chan_id;
            seq = rf.seq;

            return *this;
        }

        bm::FeatureFrame& operator =(bm::FeatureFrame&& rf)
        {
            img = rf.img;
            chan_id = rf.chan_id;
            seq = rf.seq;

            return *this;
        }
    };

    struct FeatureFrameInfo {
        std::vector<FeatureFrame> frames;
        std::vector<bm_tensor_t> input_tensors;
        std::vector<bm_tensor_t> output_tensors;
        std::vector<bm::NetOutputDatum> out_datums;
    };

}







#endif //INFERENCE_FRAMEWORK_COMMON_TYPES_H
