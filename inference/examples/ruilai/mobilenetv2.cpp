#include "mobilenetv2.h"
#include <numeric>

MobileNetV2::MobileNetV2(bm::BMNNContextPtr bmctx):m_bmctx(bmctx) {
    m_bmnet = std::make_shared<bm::BMNNNetwork>(m_bmctx->bmrt(), m_bmctx->network_name(0));
    assert(m_bmnet != nullptr);
//    m_beta  = -103.94;
//    m_alpha = 1.0;

    auto shape = m_bmnet->inputTensor(0)->get_shape();
    // for NCHW
    m_net_h = shape->dims[2];
    m_net_w = shape->dims[3];

}

MobileNetV2::~MobileNetV2()
{

}

int MobileNetV2::preprocess(std::vector<bm::ResizeFrameInfo> &frames)
{
    int ret = 0;
    bm_handle_t handle = m_bmctx->handle();

    // Only one frame and
    // full batch images are in v_resized_imgs
    assert(frames.size() == 1);
    auto &the_frame = frames[0];
    int num         = the_frame.v_resized_imgs.size();

    //2. Convert to
    bm_image convertto_imgs[num];

    float R_bias = -103.94;
    float G_bias = -116.78;
    float B_bias = -123.68;

    bm_image_data_format_ext img_type;
    auto tensor = m_bmnet->inputTensor(0);
    if (tensor->get_dtype() == BM_INT8) {
        img_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    } else {
        img_type = DATA_TYPE_EXT_FLOAT32;
    }

    ret = bm::BMImage::create_batch(handle, m_net_h, m_net_w, FORMAT_BGR_PLANAR, img_type, convertto_imgs, num, 1,
                                    false, true);
    assert(BM_SUCCESS == ret);

    bm_tensor_t input_tensor = *tensor->bm_tensor();
    bm::bm_tensor_reshape_NCHW(handle, &input_tensor, num, 3, m_net_h, m_net_w);

    ret = bm_image_attach_contiguous_mem(num, convertto_imgs, input_tensor.device_mem);
    assert(BM_SUCCESS == ret);
    float input_scale = tensor->get_scale();
    bmcv_convert_to_attr convert_to_attr;
    convert_to_attr.alpha_0 = input_scale;
    convert_to_attr.alpha_1 = input_scale;
    convert_to_attr.alpha_2 = input_scale;
    convert_to_attr.beta_0 = B_bias * input_scale;
    convert_to_attr.beta_1 = G_bias * input_scale;
    convert_to_attr.beta_2 = R_bias * input_scale;

    ret = bmcv_image_convert_to(m_bmctx->handle(), num, convert_to_attr, the_frame.v_resized_imgs.data(), convertto_imgs);
    assert(ret == 0);

    bm_image_dettach_contiguous_mem(num, convertto_imgs);

    the_frame.input_tensors.push_back(input_tensor);

    bm::BMImage::destroy_batch(the_frame.v_resized_imgs.data(), 
                               the_frame.v_resized_imgs.size());
    bm::BMImage::destroy_batch(convertto_imgs, num);

}

int MobileNetV2::forward(std::vector<bm::ResizeFrameInfo> &frame_infos)
{
    int ret = 0;
    for(int b = 0; b < frame_infos.size(); ++b) {
        for (int i = 0; i < m_bmnet->outputTensorNum(); ++i) {
            bm_tensor_t tensor;
            frame_infos[b].output_tensors.push_back(tensor);
        }
        ret = m_bmnet->forward(frame_infos[b].input_tensors.data(), frame_infos[b].input_tensors.size(),
                               frame_infos[b].output_tensors.data(), frame_infos[b].output_tensors.size());
        assert(BM_SUCCESS == ret);
    }
}

int MobileNetV2::postprocess(std::vector<bm::ResizeFrameInfo> &frameinfos)
{
    for(int i=0;i < frameinfos.size(); ++i) {

        // Free AVFrames
        auto &frame_info = frameinfos[i];

        // extract face features
        //extract_feature_cpu(frame_info);

        if (m_pfnDetectFinish != nullptr) {
            m_pfnDetectFinish(frame_info);
        }

        // Free Tensors
        for(auto& tensor : frame_info.input_tensors) {
            bm_free_device(m_bmctx->handle(), tensor.device_mem);
        }

        for(auto& tensor: frame_info.output_tensors) {
            bm_free_device(m_bmctx->handle(), tensor.device_mem);
        }
    }
}

void MobileNetV2::extract_feature_cpu(bm::ResizeFrameInfo &frame_info) {

    int frameNum = frame_info.v_resized_imgs.size();
    for(int frameIdx = 0; frameIdx < frameNum;++frameIdx) {
        bm::BMNNTensor output_tensor(m_bmctx->handle(), "", 1.0, &frame_info.output_tensors[0]);
        float *data = output_tensor.get_cpu_data();
        auto output_shape = output_tensor.get_shape();
        int batch_size = output_shape->dims[0];
        int class_num = output_shape->dims[1];
        int total = batch_size * class_num;
        std::cout << std::fixed << std::setprecision(4);
        for (int i = 0; i < total; ++i) {
            std::cout << data[i] << ", ";
        }
        std::cout << std::endl;
        std::cout.unsetf(std::ios_base::fixed);


        // int TOPK = 5;
        // std::vector<std::vector<int>> result;
        // for (int i = 0; i < batch_size; ++i) {
        //     // initialize original index locations
        //     std::vector<int> idx(class_num);
        //     std::iota(idx.begin(), idx.end(), 0);
        //     // sort indexes based on comparing values in data
        //     std::stable_sort(idx.begin(), idx.end(),
        //                      [&data](int i1, int i2) {return data[i1] > data[i2];});
        //     idx.resize(TOPK);
        //     result.push_back(idx);
        //     data += class_num;
        // }
    }
}

bm::BMNNTensorPtr MobileNetV2::get_output_tensor(const std::string &name, bm::ResizeFrameInfo& frame_info, float scale=1.0) {
    int output_tensor_num = frame_info.output_tensors.size();
    int idx = m_bmnet->outputName2Index(name);
    if (idx < 0 && idx > output_tensor_num - 1) {
        std::cout << "ERROR:idx=" << idx << std::endl;
        assert(0);
    }
    bm::BMNNTensorPtr tensor = std::make_shared<bm::BMNNTensor>(m_bmctx->handle(), name, scale,
                                                                &frame_info.output_tensors[idx]);
    return tensor;
}