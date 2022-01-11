//
// Created by shanglin on 6/3/21.
//

#include "person_detect.hpp"


Person_Detect::Person_Detect(bm::BMNNContextPtr bmctx, int maxBatch):m_bmctx(bmctx),MAX_BATCH(maxBatch)
{
    // the bmodel has only one yolo network.
    auto net_name = m_bmctx->network_name(0);
    m_bmnet = std::make_shared<bm::BMNNNetwork>(m_bmctx->bmrt(), net_name);

    //bmnet_ = std::make_shared<bm::BMNNNetwork>(bmctx_->bmrt(), "SqueezeNet");
    assert(m_bmnet != nullptr);
    assert(m_bmnet->inputTensorNum() == 1);
    auto tensor = m_bmnet->inputTensor(0);

    auto tensor1 = m_bmnet->outputTensor(0);
    //std::cout <<" ***" << tensor1->get_shape()->dims[2] << tensor1->get_shape()->dims[3] << std::endl;
    m_net_h = tensor->get_shape()->dims[2];
    m_net_w = tensor->get_shape()->dims[3];

    float scale = tensor1->get_scale(); 
    //std::cout << "person: " << scale <<" "  << tensor1->get_name() <<std::endl;
    is4N_ = false;
}

Person_Detect::~Person_Detect()
{

}

void Person_Detect::setParams(bool useCustomScale, float customInputScale, float customOutputScale)
{
    m_use_custom_scale = useCustomScale;
    m_input_scale = customInputScale;
    m_output_scale = customOutputScale;
}

int Person_Detect::preprocess(std::vector<bm::FrameInfo2> &frame_infos)
{
    int ret = 0;
    bm_handle_t handle = m_bmctx->handle();
    //calc_resized_HW(1080, 1920, &m_net_h, &m_net_w);

    //std::cout <<"person preprocess****" <<std::endl;
    std::vector<bm::FrameBaseInfo2> vecFrameBaseInfo;
    for(int frameInfoIdx = 0; frameInfoIdx < frame_infos.size(); ++frameInfoIdx) {
        auto &frame_info = frame_infos[frameInfoIdx];
        vecFrameBaseInfo.insert(vecFrameBaseInfo.end(), frame_info.frames.begin(), frame_info.frames.end());
    }

    //Clear the input frames, because we'll re-arrange it later.
    frame_infos.clear();

    int total = vecFrameBaseInfo.size();
    int left = (total % MAX_BATCH == 0 ? MAX_BATCH : total % MAX_BATCH);
    int batch_num = total % MAX_BATCH == 0 ? total / MAX_BATCH : (total / MAX_BATCH + 1);
    for (int batch_idx = 0; batch_idx < batch_num; ++batch_idx) {
        int num = MAX_BATCH;
        int start_idx = batch_idx * MAX_BATCH;
        if (batch_idx == batch_num - 1) {
            // last one
            num = left;
        }

        bm::FrameInfo2 finfo;

        //1. Resize
        bm_image resized_imgs[MAX_BATCH];
        ret = bm::BMImage::create_batch(handle, m_net_h, m_net_w, FORMAT_BGR_PLANAR,
                                        DATA_TYPE_EXT_1N_BYTE, resized_imgs, num, 64);
        assert(BM_SUCCESS == ret);

        for (int i = 0; i < num; ++i) {
            bm_image image1;
            //FrameBaseInfo2 frameBaseInfo;
            bm::BMImage::from_avframe(handle, vecFrameBaseInfo[start_idx + i].avframe, image1, true);

            ret = bm::BMImage::create_batch(handle, image1.height, image1.width, FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE,
                    &vecFrameBaseInfo[start_idx + i].origin_image, 1, 64);
            ret = bmcv_image_vpp_convert(handle, 1, image1, &vecFrameBaseInfo[start_idx+i].origin_image);

            assert(BM_SUCCESS == ret);

            ret = bmcv_image_vpp_convert(handle, 1, image1, &resized_imgs[i]);
            assert(BM_SUCCESS == ret);

            uint8_t *jpeg_data = NULL;
            size_t out_size = 0;
#if USE_QTGUI
            bmcv_image_jpeg_enc(handle, 1, &image1, (void **) &jpeg_data, &out_size);
#endif
            vecFrameBaseInfo[start_idx + i].jpeg_data = std::make_shared<bm::Data>(jpeg_data, out_size);
            vecFrameBaseInfo[start_idx + i].height = image1.height;
            vecFrameBaseInfo[start_idx + i].width = image1.width;

            av_frame_unref(vecFrameBaseInfo[start_idx + i].avframe);
            av_frame_free(&vecFrameBaseInfo[start_idx + i].avframe);

            finfo.frames.push_back(vecFrameBaseInfo[start_idx + i]);
            bm_image_destroy(image1);
            assert(vecFrameBaseInfo[start_idx + i].avframe == nullptr);
            //if (frames[start_idx].chan_id == 0)
            //std::cout << "[" << frames[start_idx].chan_id << "]total index =" << start_idx + i << std::endl;
        }

        //2. Convert to
        bm_image convertto_imgs[MAX_BATCH];

        float R_bias = -123.0f;
        float G_bias = -117.0f;
        float B_bias = -104.0f;
        float alpha, beta;

        bm_image_data_format_ext img_type = DATA_TYPE_EXT_FLOAT32;
        auto tensor = m_bmnet->inputTensor(0);
        if (tensor->get_dtype() == BM_INT8) {
            img_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
            alpha = 0.847682119;
            beta = -0.5;
            img_type = (is4N_) ? (DATA_TYPE_EXT_4N_BYTE_SIGNED)
                               : (DATA_TYPE_EXT_1N_BYTE_SIGNED);
        } else {
            alpha = 1.0;
            beta = 0.0;
            img_type = DATA_TYPE_EXT_FLOAT32;
        }

        ret = bm::BMImage::create_batch(handle, m_net_h, m_net_w, FORMAT_BGR_PLANAR, img_type, convertto_imgs, num, 1, false, true);
        assert(BM_SUCCESS == ret);

        bm_tensor_t input_tensor = *tensor->bm_tensor();
        bm::bm_tensor_reshape_NCHW(handle, &input_tensor, num, 3, m_net_h, m_net_w);

        ret = bm_image_attach_contiguous_mem(num, convertto_imgs, input_tensor.device_mem);
        assert(BM_SUCCESS == ret);

        bmcv_convert_to_attr convert_to_attr;
        convert_to_attr.alpha_0 = alpha;
        convert_to_attr.alpha_1 = alpha;
        convert_to_attr.alpha_2 = alpha;
        //convert_to_attr.beta_0 = beta + B_bias;
        //convert_to_attr.beta_1 = beta + G_bias;
        //convert_to_attr.beta_2 = beta + R_bias;
        convert_to_attr.beta_0  = -104;
        convert_to_attr.beta_1  = -117;
        convert_to_attr.beta_2  = -123;

        ret = bmcv_image_convert_to(m_bmctx->handle(), num, convert_to_attr, resized_imgs, convertto_imgs);
        assert(ret == 0);

        bm_image_dettach_contiguous_mem(num, convertto_imgs);

        bm::NetForward netFwd;
        netFwd.batch_size = num;
        netFwd.input_tensors.push_back(input_tensor);
        for (int l = 0; l < m_bmnet->outputTensorNum(); ++l) {
            bm_tensor_t t;
            netFwd.output_tensors.push_back(t);
        }
        finfo.forwards.push_back(netFwd);

        bm::BMImage::destroy_batch(resized_imgs, num);
        bm::BMImage::destroy_batch(convertto_imgs, num);

        frame_infos.push_back(finfo);
    }
}

int Person_Detect::forward(std::vector<bm::FrameInfo2>& frame_infos)
{

    int ret = 0;
    for(int b = 0; b < frame_infos.size(); ++b) {
#if DUMP_FILE
        bm::BMImage::dump_dev_memory(bmctx_->handle(), frame_infos[b].input_tensors[0].device_mem, "convertto",
                frame_infos[b].frames.size(), m_net_h, m_net_w, false, false);
#endif
        for(int l = 0; l < frame_infos[b].forwards.size(); ++l) {
            ret = m_bmnet->forward(frame_infos[b].forwards[l].input_tensors.data(), frame_infos[b].forwards[l].input_tensors.size(),
                    frame_infos[b].forwards[l].output_tensors.data(), frame_infos[b].forwards[l].output_tensors.size());
            assert(BM_SUCCESS == ret);
        }
    }

    //std::cout <<"person forard****" <<std::endl;
    return 0;
}

int Person_Detect::postprocess(std::vector<bm::FrameInfo2> &frame_infos) {
    
    int ret = 0;
    for(int frameInfoIdx =0; frameInfoIdx < frame_infos.size(); ++frameInfoIdx) {

        // Free AVFrames
        auto &frame_info = frame_infos[frameInfoIdx];

        // extract face detection
        //extract_facebox_cpu(frame_info);
        
        decode_from_output_tensor(frame_info);
        // Free Tensors
        for (auto &ios : frame_info.forwards) {
            for (auto &tensor : ios.input_tensors) {
                bm_free_device(m_bmctx->handle(), tensor.device_mem);
            }

            for (auto &tensor: ios.output_tensors) {
                bm_free_device(m_bmctx->handle(), tensor.device_mem);
            }
        }

        frame_info.forwards.clear();

        if (m_nextInferPipe == nullptr) {
            if (m_pfnDetectFinish != nullptr) {
                m_pfnDetectFinish(frame_info);
            }

            for (int j = 0; j < frame_info.frames.size(); ++j) {

                auto &reff = frame_info.frames[j];
                if (reff.avpkt) {
                    av_packet_unref(reff.avpkt);
                    av_packet_free(&reff.avpkt);
                }

                if (reff.avframe) {
                    av_frame_unref(reff.avframe);
                    av_frame_free(&reff.avframe);
                }
            }
        } else {
            // transfer to next pipe
            m_nextInferPipe->push_frame(&frame_info);

        }

    }
}

bm::BMNNTensorPtr Person_Detect::get_output_tensor(const std::string &name, bm::FrameInfo2 &frame_info, float scale) {
    int idx = 0;
    int k = 0;
    if (!name.empty()) {
        int output_tensor_num = frame_info.forwards[k].output_tensors.size();
        idx = m_bmnet->outputName2Index(name);
        if (idx < 0 || idx > output_tensor_num - 1) {
            std::cout << "ERROR:idx=" << idx << std::endl;
            assert(0);
        }
    }else{

    }
    bm::BMNNTensorPtr tensor = std::make_shared<bm::BMNNTensor>(m_bmctx->handle(), name, scale,
                                                                &frame_info.forwards[k].output_tensors[idx]);
    return tensor;
}

int count = 0;
void Person_Detect::decode_from_output_tensor(bm::FrameInfo2 &frame_info) {
    //bm::BMNNTensorPtr tensorPtr = get_output_tensor("net_output", frame_info, 1.0);
    // int8 and fp32 network's name is different, so we always we index 0.
    auto tensor = m_bmnet->outputTensor(0);
    float scale;
    if (m_use_custom_scale) {
        scale = m_output_scale;
    }else {
        scale = tensor->get_scale();
    }

    //std::cout <<" person detct: " << scale <<std::endl;
    //float scale = 0.0104515;
    bm::BMNNTensorPtr tensorPtr = get_output_tensor(tensor->get_name(), frame_info, scale);
    cv::Size netInputSize;
    //netInputSize.width = frame_info.input_tensors[0].shape.dims[3];
    //netInputSize.height = frame_info.input_tensors[0].shape.dims[2];

    //std::vector<bm::PoseKeyPoints> vct_keypoints;
    cv::Size originSize (frame_info.frames[0].width, frame_info.frames[0].height);
    
    //std::cout << tensor->get_name() << std::endl;
    int n = tensorPtr->get_num();
    int chan_num = tensorPtr->get_shape()->dims[1];
    int net_output_height = tensorPtr->get_shape()->dims[2];
    int net_output_width = tensorPtr->get_shape()->dims[3];
    
    //td::cout << cout << " "  << tensorPtr->get_shape()->dims[0] << " " << chan_num << " " << net_output_height << " " << net_output_width << std::endl;
    std::cout <<"Frame no: " << count <<std::endl;
    count++;
    std::vector<bm::SafetyhatObject> personRects;
    personRects.clear();
    for(int batch_idx = 0;batch_idx < n; ++batch_idx) {
        int batch_byte_size = chan_num*net_output_height*net_output_width;
        float *base = tensorPtr->get_cpu_data() + batch_byte_size * batch_idx;
        for (int out = 0; out < net_output_height; out++) {
            if (base[out*net_output_width+2] >= base_threshold_) { 
                bm::SafetyhatObject rect;
                rect.class_id = base[out*net_output_width+1];
                rect.score = base[out*net_output_width+2];
                if (base[out*net_output_width+3]* frame_info.frames[0].width < 0)
                    rect.x1 = 0;
                else
                    rect.x1 = base[out*net_output_width+3]* frame_info.frames[0].width;

                if (base[out*net_output_width+4] *frame_info.frames[0].height < 0)
                    rect.y1 = 0;
                else
                    rect.y1 = base[out*net_output_width+4] *frame_info.frames[0].height;
                
                if (base[out*net_output_width+5] * frame_info.frames[0].width >= frame_info.frames[0].width)
                    rect.x2 = frame_info.frames[0].width -1;
                else
                    rect.x2 = base[out*net_output_width+5] * frame_info.frames[0].width;

                if (base[out*net_output_width+6]*frame_info.frames[0].height >= frame_info.frames[0].height)
                    rect.y2  = frame_info.frames[0].height -1;
                else
                    rect.y2 = base[out*net_output_width+6]*frame_info.frames[0].height;
                std::cout << "x1: " << rect.x1 <<" " <<rect.y1 <<" " << rect.x2 <<" " <<rect.y2<<" " << rect.class_id << " " << rect.score<<std::endl;
                personRects.push_back(rect);
            }
        }
    }

    frame_info.out_datums.push_back(bm::NetOutputDatum(personRects));
}


