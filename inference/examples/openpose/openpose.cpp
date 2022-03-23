//
// Created by yuan on 3/9/21.
//

#include "openpose.h"
#include "pose_postprocess.h"

OpenPose::OpenPose(bm::BMNNContextPtr bmctx, int maxBatch, std::string strModelType):m_bmctx(bmctx),MAX_BATCH(maxBatch)
{
    m_model_type = strModelType.compare("coco_18") == 0 ? bm::PoseKeyPoints::EModelType::COCO_18 : bm::PoseKeyPoints::EModelType::BODY_25;

    // the bmodel has only one yolo network.
    auto net_name = m_bmctx->network_name(0);
    m_bmnet = std::make_shared<bm::BMNNNetwork>(m_bmctx->bmrt(), net_name);
    assert(m_bmnet != nullptr);
    assert(m_bmnet->inputTensorNum() == 1);
    auto tensor = m_bmnet->inputTensor(0);

    m_net_h = tensor->get_shape()->dims[2];
    m_net_w = tensor->get_shape()->dims[3];
}

OpenPose::~OpenPose()
{

}

void OpenPose::setParams(bool useCustomScale, float customInputScale, float customOutputScale)
{
    m_use_custom_scale = useCustomScale;
    m_input_scale = customInputScale;
    m_output_scale = customOutputScale;
}

int OpenPose::preprocess(std::vector<bm::FrameBaseInfo>& frames, std::vector<bm::FrameInfo>& frame_infos)
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

        bm::FrameInfo finfo;
        //1. Resize
        bm_image resized_imgs[MAX_BATCH];
        ret = bm::BMImage::create_batch(handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, resized_imgs, num, 64);
        assert(BM_SUCCESS == ret);

        for(int i = 0;i < num; ++i) {
            bm_image image1;
            bm::BMImage::from_avframe(handle, frames[start_idx + i].avframe, image1, true);
            ret = bmcv_image_vpp_convert(handle, 1, image1, &resized_imgs[i]);
            assert(BM_SUCCESS == ret);

            // convert data to jpeg
            uint8_t *jpeg_data=NULL;
            size_t out_size = 0;
#if USE_QTGUI
            bmcv_image_jpeg_enc(handle, 1, &image1, (void**)&jpeg_data, &out_size);
#endif
            frames[start_idx + i].jpeg_data = std::make_shared<bm::Data>(jpeg_data, out_size);
            frames[start_idx + i].width = frames[start_idx+i].avframe->width;
            frames[start_idx + i].height = frames[start_idx+i].avframe->height;
            av_frame_unref(frames[start_idx + i].avframe);
            av_frame_free(&frames[start_idx + i].avframe);

            finfo.frames.push_back(frames[start_idx+i]);
            bm_image_destroy(image1);
        }

        //2. Convert to
        bm_image convertto_imgs[MAX_BATCH];
        float alpha, beta;

        bm_image_data_format_ext img_type = DATA_TYPE_EXT_FLOAT32;
        auto tensor = m_bmnet->inputTensor(0);

        float scale;
        if (m_use_custom_scale) {
            scale = m_input_scale;
        }else{
            scale = tensor->get_scale();
        }

        if (tensor->get_dtype() == BM_INT8) {
            img_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
            alpha            = 1.0 / 256.f * scale;
            beta             = -0.5 * scale;
            img_type = (DATA_TYPE_EXT_1N_BYTE_SIGNED);
        }else{
            alpha            = 1.0/256.f;
            beta             = -0.5;
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

        frame_infos.push_back(finfo);
    }


}

int OpenPose::forward(std::vector<bm::FrameInfo>& frame_infos)
{
    int ret = 0;
    for(int b = 0; b < frame_infos.size(); ++b) {
        for (int i = 0; i < m_bmnet->outputTensorNum(); ++i) {
            bm_tensor_t tensor;
            frame_infos[b].output_tensors.push_back(tensor);
        }

#if DUMP_FILE
        bm::BMImage::dump_dev_memory(bmctx_->handle(), frame_infos[b].input_tensors[0].device_mem, "convertto",
                frame_infos[b].frames.size(), m_net_h, m_net_w, false, false);
#endif
        ret = m_bmnet->forward(frame_infos[b].input_tensors.data(), frame_infos[b].input_tensors.size(),
                               frame_infos[b].output_tensors.data(), frame_infos[b].output_tensors.size());
        assert(BM_SUCCESS == ret);
    }

    return 0;
}

int OpenPose::postprocess(std::vector<bm::FrameInfo> &frame_infos) {
    for (int i = 0; i < frame_infos.size(); ++i) {

        // Free AVFrames
        auto frame_info = frame_infos[i];

        // decode result
        decode_from_output_tensor(frame_info);

        if (m_detect_finish_func != nullptr) {
            m_detect_finish_func(frame_info);
        }

        for (int j = 0; j < frame_info.frames.size(); ++j) {

            auto reff = frame_info.frames[j];
            assert(reff.avpkt != nullptr);
            av_packet_unref(reff.avpkt);
            av_packet_free(&reff.avpkt);

            assert(reff.avframe == nullptr);
            av_frame_unref(reff.avframe);
            av_frame_free(&reff.avframe);
        }

        // Free Tensors
        for (auto &tensor : frame_info.input_tensors) {
            bm_free_device(m_bmctx->handle(), tensor.device_mem);
        }

        for (auto &tensor: frame_info.output_tensors) {
            bm_free_device(m_bmctx->handle(), tensor.device_mem);
        }

    }
}

bm::BMNNTensorPtr OpenPose::get_output_tensor(const std::string &name, bm::FrameInfo &frame_info, float scale) {
    int idx = 0;
    if (!name.empty()) {
        int output_tensor_num = frame_info.output_tensors.size();
        idx = m_bmnet->outputName2Index(name);
        if (idx < 0 || idx > output_tensor_num - 1) {
            std::cout << "ERROR:idx=" << idx << std::endl;
            assert(0);
        }
    }else{

    }
    bm::BMNNTensorPtr tensor = std::make_shared<bm::BMNNTensor>(m_bmctx->handle(), name, scale,
                                                                &frame_info.output_tensors[idx]);
    return tensor;
}

void OpenPose::decode_from_output_tensor(bm::FrameInfo &frame_info) {
    //bm::BMNNTensorPtr tensorPtr = get_output_tensor("net_output", frame_info, 1.0);
    // int8 and fp32 network's name is different, so we always we index 0.
    auto tensor = m_bmnet->outputTensor(0);
    float scale;
    if (m_use_custom_scale) {
        scale = m_output_scale;
    }else {
        scale = tensor->get_scale();
    }

    //float scale = 0.0104515;
    bm::BMNNTensorPtr tensorPtr = get_output_tensor(tensor->get_name(), frame_info, scale);
    cv::Size netInputSize;
    netInputSize.width = frame_info.input_tensors[0].shape.dims[3];
    netInputSize.height = frame_info.input_tensors[0].shape.dims[2];

    std::vector<bm::PoseKeyPoints> vct_keypoints;
    cv::Size originSize (frame_info.frames[0].width, frame_info.frames[0].height);
    OpenPosePostProcess::getKeyPoints(tensorPtr, netInputSize, originSize, vct_keypoints, m_model_type);
    for(int i = 0;i < vct_keypoints.size(); ++i) {
        frame_info.out_datums.push_back(bm::NetOutputDatum(vct_keypoints[i]));
    }

    if (m_pfnDetectFinish != nullptr) {
        m_pfnDetectFinish(frame_info);
    }
}


