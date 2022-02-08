#ifndef _CPP_BMCV_SAIL_H_
#define _CPP_BMCV_SAIL_H_

#include "cvwrapper.h"


typedef struct BMBBox {
    int x{0};
    int y{0};
    int w{0};
    int h{0};
    int c{-1};
    float conf{0.f};


    BMBBox(int _x, int _y, int _w, int _h, int _c, float _conf)
        : x{_x},
          y{_y},
          w{_w},
          h{_h},
          c{_c},
          conf{_conf} {}
} BMBBox;

#ifdef USE_BMCV
class CenterNetPreprocessor {

public:
    /**
     * @brief Constructor.
     *
     * @param bmcv  Reference to a Bmcv instance
     * @param scale Scale factor from float32 to int8
     */
    CenterNetPreprocessor(sail::Bmcv& bmcv, int result_width, int result_height, float scale = 1.0);

    /**
     * @brief deconstructor.
     */
    virtual ~CenterNetPreprocessor() = default;

    /**
     * @brief Execution function of preprocessing.
     *
     * @param input Input data
     * @param input Output data
     */
    void Process(sail::BMImage& input, sail::BMImage& output);

    /**
     * @brief Execution function of preprocessing for multiple images.
     *
     * @param input Input data
     * @param input Output data
     */
    template<std::size_t N>
    void process(sail::BMImageArray<N>& input, sail::BMImageArray<N>& output) {
        sail::BMImageArray<N> tmp = bmcv_.vpp_resize(input, resize_w_, resize_w_);
        bmcv_.convert_to(tmp, output,
                         std::make_tuple(std::make_pair(ab_[0], ab_[1]),
                                         std::make_pair(ab_[2], ab_[3]),
                                         std::make_pair(ab_[4], ab_[5])));
    }

    void BGRNormalization(float* data);

private:
    sail::Bmcv& bmcv_;
    int resize_w_;
    int resize_h_;
    float ab_[6];

};
#endif


class CenterNetPostprocessor {
    constexpr static int kHeatMapChannels = 80;     // channel number of heatmap
    constexpr static int kHeightWChannels = 2;      // channel number of width and height
    constexpr static int kOffsetChanels   = 2;      // channel number of offset to center

public:
    CenterNetPostprocessor(std::vector<int>& output_shape, float threshold);

    virtual ~CenterNetPostprocessor();

    // Entry of postprocess
    void Process(float* output_data);

    // Simple implemtation of maxpool, stride is 3, padding 1
    void SimplyMaxpool2D(float* data);

    // Find the object according the confidence in 80 classes
    float FindMaxConfidenceObject(float score[], int count, int& idx);

    // Get predicted bboxes
    std::shared_ptr<std::vector<BMBBox>> CenternetCorrectBBox(int height, int width);
private:
    std::vector<int>                          output_shape_;                     // 网络输出shape
    int                                       output_size_[4];                   // 按byte计算每个维度的stride
    float*                                    pred_hms_[kHeatMapChannels];       // 热力图
    float*                                    pred_whs_[kHeightWChannels];       // 宽高
    float*                                    pred_off_[kOffsetChanels];         // 偏移
    float                                     threshold_;                        // 目标阈值
    std::unique_ptr<int[]>                    confidence_mask_ptr_;              // centernet中每个grid置信度最高，且大于阈值的Mask
    std::unique_ptr<float[]>                  confidence_ptr_;                   // centernet中每个grid置信度最高的confidence
    std::unique_ptr<float[]>                  detected_objs_;                    // 经过后处理，检测到bbox坐标+置信度+种类
    int                                       detected_count_;                   // 经过后处理，检测到的目标框数量
};

#endif //_CPP_BMCV_SAIL_H_