//
// Created by yuan on 11/24/21.
//

#include "face_extract.h"

FaceExtract::FaceExtract(bm::BMNNContextPtr bmctx, int max_batch):m_bmctx(bmctx),MAX_BATCH(max_batch) {
    m_bmnet = std::make_shared<bm::BMNNNetwork>(m_bmctx->bmrt(), "r50i-model");
    assert(m_bmnet != nullptr);
    m_alpha_int8 = 1.003921316;
    m_beta_int8  = -127.5 * 1.003921316;
    m_alpha_fp32 = 0.0078125;
    m_beta_fp32  = -127.5 * 0.0078125;

    auto shape = m_bmnet->inputTensor(0)->get_shape();
    // for NCHW
    m_net_h = shape->dims[2];
    m_net_w = shape->dims[3];

}

FaceExtract::~FaceExtract()
{

}

int FaceExtract::preprocess(std::vector<bm::FeatureFrame> &frames, std::vector<bm::FeatureFrameInfo> &of)
{
    int ret = 0;
    bm_handle_t handle = m_bmctx->handle();

    // Check input
    int total = frames.size();
    int left = (total%MAX_BATCH == 0 ? MAX_BATCH: total%MAX_BATCH);
    int batch_num = total%MAX_BATCH==0 ? total/MAX_BATCH: (total/MAX_BATCH + 1);
    for(int batch_idx = 0; batch_idx < batch_num; ++ batch_idx) {
        int num = MAX_BATCH;
        int start_idx = batch_idx*MAX_BATCH;
        if (batch_idx == batch_num-1) {
            // last one
            num = left;
        }

        bm::FeatureFrameInfo finfo;

        //1. Resize
        bm_image resized_imgs[MAX_BATCH];
        ret = bm::BMImage::create_batch(handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, resized_imgs, num, 64);
        assert(BM_SUCCESS == ret);

        for(int i = 0;i < num; ++i) {
            bm_image image1;
            cv::Mat cvm1=frames[start_idx + i].img;
            cv::bmcv::toBMI(cvm1, &image1);
            ret = bmcv_image_vpp_convert(handle, 1, image1, &resized_imgs[i]);
            assert(BM_SUCCESS == ret);

            finfo.frames.push_back(frames[start_idx + i]);
            bm_image_destroy(image1);
        }

        //2. Convert to
        bm_image convertto_imgs[MAX_BATCH];
        float alpha, beta;

        bm_image_data_format_ext img_type;
        auto tensor = m_bmnet->inputTensor(0);
        if (tensor->get_dtype() == BM_INT8) {
            alpha            = m_alpha_int8;
            beta             = m_beta_int8;
            img_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
        }else{
            alpha            = m_alpha_fp32;
            beta             = m_alpha_fp32;
            img_type = DATA_TYPE_EXT_FLOAT32;
        }

        ret = bm::BMImage::create_batch(handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_type, convertto_imgs, num, 1, false, true);
        assert(BM_SUCCESS == ret);

        bm_tensor_t input_tensor = *tensor->bm_tensor();
        bm::bm_tensor_reshape_NCHW(handle, &input_tensor, num, 3, m_net_h, m_net_w);

        ret = bm_image_attach_contiguous_mem(num, convertto_imgs, input_tensor.device_mem);
        assert(BM_SUCCESS == ret);

        bmcv_convert_to_attr convert_to_attr;
        convert_to_attr.alpha_0 = alpha;
        convert_to_attr.alpha_1 = alpha;
        convert_to_attr.alpha_2 = alpha;
        convert_to_attr.beta_0  = beta;
        convert_to_attr.beta_1  = beta;
        convert_to_attr.beta_2  = beta;

        ret = bmcv_image_convert_to(m_bmctx->handle(), num, convert_to_attr, resized_imgs, convertto_imgs);
        assert(ret == 0);

        bm_image_dettach_contiguous_mem(num, convertto_imgs);

        finfo.input_tensors.push_back(input_tensor);

        bm::BMImage::destroy_batch(resized_imgs, num);
        bm::BMImage::destroy_batch(convertto_imgs, num);

        of.push_back(finfo);
    }
}

int FaceExtract::forward(std::vector<bm::FeatureFrameInfo> &frame_infos)
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

int FaceExtract::postprocess(std::vector<bm::FeatureFrameInfo> &frameinfos)
{
    for(int i=0;i < frameinfos.size(); ++i) {

        // Free AVFrames
        auto &frame_info = frameinfos[i];

        // extract face features
        extract_facefeature_cpu(frame_info);

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

void FaceExtract::extract_facefeature_cpu(bm::FeatureFrameInfo &frame_info) {

    int frameNum = frame_info.frames.size();
    frame_info.out_datums.resize(frameNum);
    for(int frameIdx = 0; frameIdx < frameNum;++frameIdx) {
        bm::BMNNTensorPtr output_tensor = get_output_tensor("fc1_scale", frame_info, 1.0);
        const void *out_data = (const void *) output_tensor->get_cpu_data();
        auto output_shape = output_tensor->get_shape();
        int out_c = output_shape->dims[1];
        int out_n = output_shape->dims[0];

        for (int n = 0; n < out_n; n++) {
            const float *data = (const float *) out_data + out_c * n;
            bm::ObjectFeature features;
            features.clear();
            for (int i = 0; i < out_c; i++) {
                features.push_back(data[i]);
            }
            frame_info.out_datums[frameIdx].face_features.push_back(features);
        }
    }
}

bm::BMNNTensorPtr FaceExtract::get_output_tensor(const std::string &name, bm::FeatureFrameInfo& frame_info, float scale=1.0) {
    int output_tensor_num = frame_info.output_tensors.size();
    int idx = m_bmnet->outputName2Index(name);
    if (idx < 0 && idx > output_tensor_num-1) {
        std::cout << "ERROR:idx=" << idx << std::endl;
        assert(0);
    }
    bm::BMNNTensorPtr tensor = std::make_shared<bm::BMNNTensor>(m_bmctx->handle(), name, scale,
                                                                &frame_info.output_tensors[idx]);
    return tensor;
}