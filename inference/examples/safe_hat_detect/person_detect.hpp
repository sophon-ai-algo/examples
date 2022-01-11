#ifndef __PERSON_DETECT_HPP
#define  __PERSON_DETECT_HPP

#include "bmutility.h"
#include "inference2.h"
#include "common.h"

class Person_Detect : public bm::DetectorDelegate<bm::FrameInfo2> {
    int MAX_BATCH = 1;
    bm::BMNNContextPtr m_bmctx;
    bm::BMNNNetworkPtr m_bmnet;
    int m_net_h, m_net_w;
    bool               is4N_;
    //std::function<void(bm::FrameInfo &infos)> m_detect_finish_func;

    //configuration
    bool m_use_custom_scale {false};
    float m_input_scale;
    float m_output_scale;
    float base_threshold_{0.5};
    /*enum MODLE_TYPE{
        MODEL_PERSON = 0,
        MODEL_SAFETY_HAT = 1
    };

    MODLE_TYPE m_model_type{MODEL_PERSON};*/
public:
    Person_Detect(bm::BMNNContextPtr bmctx, int maxBatch=4);
    void setParams(bool useCustomScale, float customInputScale, float customOutputScale);

    ~Person_Detect();

    virtual int preprocess(std::vector<bm::FrameInfo2>& frame_info) override;

    virtual int forward(std::vector <bm::FrameInfo2> &frame_info) override;

    virtual int postprocess(std::vector <bm::FrameInfo2> &frame_info) override;

#if 0x0
    int person_preprocess();
    int safety_hat_preprocess();

    int person_preprocess();
    int safety_hat_preprocess();

    int person_postprocess();
    int safety_hat_postprocess();
#endif
private:
    bm::BMNNTensorPtr get_output_tensor(const std::string &name, bm::FrameInfo2& frame_info, float scale=1.0);
    void decode_from_output_tensor(bm::FrameInfo2& frame_info);

};
#endif