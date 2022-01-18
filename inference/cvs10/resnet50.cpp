//
// Created by yuan on 11/24/21.
//

#include "resnet50.h"
#include <numeric>

Resnet::Resnet(bm::BMNNContextPtr bmctx, int max_batch):m_bmctx(bmctx),MAX_BATCH(max_batch) {
    m_bmnet = std::make_shared<bm::BMNNNetwork>(m_bmctx->bmrt(), "resnet-50");
    assert(m_bmnet != nullptr);
    m_beta = -103.94;
    m_alpha = 1.0;

    auto shape = m_bmnet->inputTensor(0)->get_shape();
    // for NCHW
    m_net_h = shape->dims[2];
    m_net_w = shape->dims[3];

}

Resnet::~Resnet()
{

}

int Resnet::preprocess(std::vector<bm::cvs10FrameBaseInfo> &frames, std::vector<bm::cvs10FrameInfo> &of)
{
    int ret = 0;
    bm_handle_t handle = m_bmctx->handle();

    // Check input
    int total = frames.size();
    if (total != 4) {
        printf("total = %d\n", total);
    }
    int left = (total%MAX_BATCH == 0 ? MAX_BATCH: total%MAX_BATCH);
    int batch_num = total%MAX_BATCH==0 ? total/MAX_BATCH: (total/MAX_BATCH + 1);
    for(int batch_idx = 0; batch_idx < batch_num; ++ batch_idx) {
        int num = MAX_BATCH;
        int start_idx = batch_idx * MAX_BATCH;
        if (batch_idx == batch_num - 1) {
            // last one
            num = left;
        }

        bm::cvs10FrameInfo finfo;
        //1. Resize
        bm_image resized_imgs[MAX_BATCH];
        ret = bm::BMImage::create_batch(handle, m_net_h, m_net_w, FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE,
                                        resized_imgs, num, 64);
        assert(BM_SUCCESS == ret);

        for (int i = 0; i < num; ++i) {
            bm_image image1;
            //FrameBaseInfo frameBaseInfo;

            bm::BMImage::from_avframe(handle, frames[start_idx + i].avframe, image1, true);
            ret = bmcv_image_vpp_convert(handle, 1, image1, &resized_imgs[i]);
            assert(BM_SUCCESS == ret);

            uint8_t *jpeg_data = NULL;
            size_t out_size = 0;
#if USE_QTGUI
            bmcv_image_jpeg_enc(handle, 1, &image1, (void **) &jpeg_data, &out_size);
#endif
            frames[start_idx + i].jpeg_data = std::make_shared<bm::Data>(jpeg_data, out_size);
            frames[start_idx + i].height = image1.height;
            frames[start_idx + i].width = image1.width;

            av_frame_unref(frames[start_idx + i].avframe);
            av_frame_free(&frames[start_idx + i].avframe);

            finfo.frames.push_back(frames[start_idx + i]);
            bm_image_destroy(image1);

#ifdef DEBUG
            if (frames[start_idx].chan_id == 0)
                 std::cout << "[" << frames[start_idx].chan_id << "]total index =" << start_idx + i << std::endl;
#endif
        }

        //2. Convert to
        bm_image convertto_imgs[MAX_BATCH];

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

        ret = bmcv_image_convert_to(m_bmctx->handle(), num, convert_to_attr, resized_imgs, convertto_imgs);
        assert(ret == 0);

        bm_image_dettach_contiguous_mem(num, convertto_imgs);

        finfo.input_tensors.push_back(input_tensor);

        bm::BMImage::destroy_batch(resized_imgs, num);
        bm::BMImage::destroy_batch(convertto_imgs, num);

        of.push_back(finfo);
    }
}

int Resnet::forward(std::vector<bm::cvs10FrameInfo> &frame_infos)
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

int Resnet::postprocess(std::vector<bm::cvs10FrameInfo> &frameinfos)
{
    for(int i=0;i < frameinfos.size(); ++i) {

        // Free AVFrames
        auto &frame_info = frameinfos[i];

        // extract face features
        extract_feature_cpu(frame_info);

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

void Resnet::extract_feature_cpu(bm::cvs10FrameInfo &frame_info) {

    int frameNum = frame_info.frames.size();
    frame_info.out_datums.resize(frameNum);
    for(int frameIdx = 0; frameIdx < frameNum;++frameIdx) {
        bm::BMNNTensorPtr output_tensor = get_output_tensor("prob", frame_info, 1.0);
        float *data = output_tensor->get_cpu_data();
        auto output_shape = output_tensor->get_shape();
        int batch_size = output_shape->dims[0];
        int class_num = output_shape->dims[1];

        int TOPK=5;
        std::vector<std::vector<int>> result;
        for (int i = 0; i < batch_size; ++i) {
            // initialize original index locations
            std::vector<int> idx(class_num);
            std::iota(idx.begin(), idx.end(), 0);
            // sort indexes based on comparing values in data
            std::stable_sort(idx.begin(), idx.end(),
                             [&data](int i1, int i2) {return data[i1] > data[i2];});
            idx.resize(TOPK);
            result.push_back(idx);
            data += class_num;
        }
    }
}

bm::BMNNTensorPtr Resnet::get_output_tensor(const std::string &name, bm::cvs10FrameInfo& frame_info, float scale=1.0) {
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