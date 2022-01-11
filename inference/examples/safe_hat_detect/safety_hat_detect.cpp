//
// Created by shanglin on 6/3/21.
//

#include "safety_hat_detect.hpp"


Safety_Hat_Detect::Safety_Hat_Detect(bm::BMNNContextPtr bmctx, int maxBatch):m_bmctx(bmctx),MAX_BATCH(maxBatch)
{
    // the bmodel has only one yolo network.
    auto net_name = m_bmctx->network_name(1);
    m_bmnet = std::make_shared<bm::BMNNNetwork>(m_bmctx->bmrt(), net_name);
    assert(m_bmnet != nullptr);
    assert(m_bmnet->inputTensorNum() == 1);
    auto tensor = m_bmnet->inputTensor(0);

    auto tensor1 = m_bmnet->outputTensor(0);
    //std::cout <<" ***" << tensor1->get_shape()->dims[2] << tensor1->get_shape()->dims[3] << std::endl;
    m_net_h = tensor->get_shape()->dims[2];
    m_net_w = tensor->get_shape()->dims[3];
    //std::cout <<"w ***8 " <<m_net_w << " " << m_net_h <<std::endl;
}

Safety_Hat_Detect::~Safety_Hat_Detect()
{

}

void Safety_Hat_Detect::setParams(bool useCustomScale, float customInputScale, float customOutputScale)
{
    m_use_custom_scale = useCustomScale;
    m_input_scale = customInputScale;
    m_output_scale = customOutputScale;
}

int Safety_Hat_Detect::preprocess(std::vector<bm::FrameInfo2> &frame_infos){
    int ret = 0;
    bm_handle_t handle = m_bmctx->handle();

    //std::cout <<"safe hat preprocess" <<std::endl;
    for(int frameInfoIdx = 0; frameInfoIdx < frame_infos.size(); ++frameInfoIdx) {
        // for total batch N
        auto &frame_info = frame_infos[frameInfoIdx];
        int frameNum = frame_info.frames.size();
        frame_info.forwards.resize(frameNum);
        for (int frameIdx = 0; frameIdx < frameNum; ++frameIdx) {
            // for each frame in batch one, crop images
            //crop images by face rects
            auto &rcs = frame_info.out_datums[frameIdx].safetyhat_objects;
            int face_num = rcs.size();
            //std::cout << "Detected faces: " << rcs.size() << std::endl;
            if (face_num > 0) {
                bmcv_rect_t crop_rects[face_num];
                frame_info.frames[frameIdx].crop_bmimgs.resize(face_num);
                for (int k = 0; k < face_num; k++) {
                    rcs[k].to_bmcv_rect(&crop_rects[k]);
                    ret = bm::BMImage::create_batch(handle, crop_rects[k].crop_h, crop_rects[k].crop_w,
                            FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE, &frame_info.frames[frameIdx].crop_bmimgs[k], 1, 64);
                    assert(BM_SUCCESS == ret);
                }

                //std::cout << "safehat****:"<<frame_infos.size() << " " << frameNum <<" " << frameIdx <<" " << face_num << std::endl;
                ret = bmcv_image_crop(handle, face_num, crop_rects,
                                      frame_info.frames[frameIdx].origin_image,
                                      frame_info.frames[frameIdx].crop_bmimgs.data());
                assert(BM_SUCCESS == ret);

                // convert images to tensors
                int total = face_num;
                int left = (total % MAX_BATCH == 0 ? MAX_BATCH : total % MAX_BATCH);
                int batch_num = total % MAX_BATCH == 0 ? total / MAX_BATCH : (total / MAX_BATCH + 1);
                for (int batch_idx = 0; batch_idx < batch_num; ++batch_idx) {
                    int num = MAX_BATCH;
                    int start_idx = batch_idx * MAX_BATCH;
                    if (batch_idx == batch_num - 1) {
                        // last one
                        num = left;
                    }

                    bm_image resized_imgs[MAX_BATCH];
                    ret = bm::BMImage::create_batch(handle, m_net_h,
                                                    m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE,
                                                    resized_imgs,
                                                    num, 64);
                    assert(BM_SUCCESS == ret);

                    //1. Resize
                    for (int i = 0; i < num; ++i) {
                        ret = bmcv_image_vpp_convert(handle, 1, frame_info.frames[frameIdx].crop_bmimgs[start_idx + i],
                                                     &resized_imgs[i]);
                        assert(BM_SUCCESS == ret);
                    }

                    //2. Convert to
                    bm_image convertto_imgs[MAX_BATCH];
                    float alpha, beta;

                    bm_image_data_format_ext img_type;
                    auto inputTensorPtr = m_bmnet->inputTensor(0);
                    if (inputTensorPtr->get_dtype() == BM_INT8) {
                        img_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
                        alpha = 1;
                        beta = 0;
                        img_type = (DATA_TYPE_EXT_1N_BYTE_SIGNED);
                    } else {
                        alpha = 1.0/255.f;
                        beta = 0;
                        img_type = DATA_TYPE_EXT_FLOAT32;
                    }

                    ret = bm::BMImage::create_batch(handle, m_net_h, m_net_w,
                                                    FORMAT_RGB_PLANAR, img_type, convertto_imgs, num, 1, false, true);
                    assert(BM_SUCCESS == ret);

                    // create input tensor
                    bm_tensor_t input_tensor = *inputTensorPtr->bm_tensor();
                    bm::bm_tensor_reshape_NCHW(handle, &input_tensor, num, 3, m_net_h, m_net_w);

                    ret = bm_image_attach_contiguous_mem(num, convertto_imgs, input_tensor.device_mem);
                    assert(BM_SUCCESS == ret);

                    bmcv_convert_to_attr convert_to_attr;
                    convert_to_attr.alpha_0 = alpha;
                    convert_to_attr.alpha_1 = alpha;
                    convert_to_attr.alpha_2 = alpha;
                    convert_to_attr.beta_0 = beta;
                    convert_to_attr.beta_1 = beta;
                    convert_to_attr.beta_2 = beta;

                    ret = bmcv_image_convert_to(m_bmctx->handle(), num, convert_to_attr, resized_imgs, convertto_imgs);
                    assert(ret == 0);

                    bm_image_dettach_contiguous_mem(num, convertto_imgs);

                    bm::NetForward netFwd;
                    netFwd.batch_size = num;
                    netFwd.input_tensors.push_back(input_tensor);
                    for(int l=0;l < m_bmnet->outputTensorNum(); ++l) {
                        bm_tensor_t t;
                        netFwd.output_tensors.push_back(t);
                    }

                    frame_info.forwards[frameIdx].subnet_ios.push_back(netFwd);

                    bm::BMImage::destroy_batch(resized_imgs, num);
                    bm::BMImage::destroy_batch(convertto_imgs, num);
                }
            }
        }
    }

    return 0;
}

int Safety_Hat_Detect::forward_subnet(std::vector<bm::NetForward> &ios) {
    int ret = 0;
    for(auto &io: ios) {
#if DUMP_FILE
        bm::BMImage::dump_dev_memory(bmctx_->handle(), frame_infos[b].input_tensors[0].device_mem, "convertto",
                    frame_infos[b].frames.size(), m_net_h, m_net_w, false, false);
#endif
        if (io.batch_size > 0) {
            ret = m_bmnet->forward(io.input_tensors.data(),
                                   io.input_tensors.size(),
                                   io.output_tensors.data(),
                                   io.output_tensors.size());
            assert(BM_SUCCESS == ret);
        }

        ret = forward_subnet(io.subnet_ios);
        assert(BM_SUCCESS == ret);
    }

 
    return ret;
}


int Safety_Hat_Detect::forward(std::vector<bm::FrameInfo2>& frame_infos)
{
    int ret = 0;
    for(int infoIdx = 0; infoIdx < frame_infos.size(); ++infoIdx) {
        // for frame_info
        auto &frame_info = frame_infos[infoIdx];
        forward_subnet(frame_info.forwards);
    }

    //std::cout << "safe hat forward"<<std::endl;
    return 0;
}

void Safety_Hat_Detect::free_NetFwds(std::vector<bm::NetForward> &NetFwds) {
    for (auto &ios : NetFwds) {
        for (auto &tensor : ios.input_tensors) {
            bm_free_device(m_bmctx->handle(), tensor.device_mem);
        }

        for (auto &tensor: ios.output_tensors) {
            bm_free_device(m_bmctx->handle(), tensor.device_mem);
        }

        free_NetFwds(ios.subnet_ios);
    }
}


int Safety_Hat_Detect::argmax(const float* data, int num, float *confidence) {
    float max_value = 0.0;
    int max_index = 0;
    for(int i = 0; i < num; ++i) {
        
        if (data[i] > max_value) {
            max_value = data[i];
            max_index = i;
        }
    }

    *confidence = max_value;
    return max_index;
}

int Safety_Hat_Detect::postprocess(std::vector<bm::FrameInfo2> &frame_infos) {
    // for each batch
    //std::cout <<"safe hat post handle" <<std::endl;
    for(int frameInfoIdx = 0; frameInfoIdx < frame_infos.size(); ++frameInfoIdx) {
        auto &frame_info = frame_infos[frameInfoIdx];
        // for each frame
        int frameNum = frame_info.frames.size();
        for (int frameIdx = 0; frameIdx < frameNum; ++frameIdx) {
            int start_idx = 0;
            //frame_info.out_datums[frameIdx].safetyhat_objects.clear();
            // for each landmark batch
            for(int j = 0; j < frame_info.forwards[frameIdx].subnet_ios.size(); ++j) {
                auto &subnet_io = frame_info.forwards[frameIdx].subnet_ios[j];
                int image_n = subnet_io.batch_size;
                bm::BMNNTensorPtr prob_tensor = get_output_tensor("prob", &subnet_io);
                //bm::BMNNTensorPtr points_tensor = get_output_tensor("conv6-2", &subnet_io);
                const float *prob_data = prob_tensor->get_cpu_data();
                //const float *points_data = points_tensor->get_cpu_data();

                // landmark batch size
                for (int k = 0; k < image_n; ++k) {
                    //bm::SafetyhatObject safetyhat_info;

                    int index = 0;
                    float max_value = 0;
                    
                    index = argmax(prob_data+57, 4, &max_value);
                    //std::cout <<"index: " << index << " "<<std::setprecision(6)<<max_value <<std::endl;
                    //safetyhat_info.index = index;
                    //safetyhat_info.confidence = max_value;
                    frame_info.out_datums[frameIdx].safetyhat_objects[j].index = index;
                    frame_info.out_datums[frameIdx].safetyhat_objects[j].confidence = max_value;
                    

                }

                start_idx += image_n;
            }
        } // end of frameIdx

        std::cout <<"index: ";
        for (int kk=0 ;  kk < frame_info.out_datums[0].safetyhat_objects.size(); kk++)
        {
            
            std::cout << " " <<frame_info.out_datums[0].safetyhat_objects[kk].index;
        }

        std::cout << std::endl;

        std::cout <<"confi: ";
        for (int kk=0 ;  kk < frame_info.out_datums[0].safetyhat_objects.size(); kk++)
        {
            
            std::cout << " " <<std::setprecision(6) << frame_info.out_datums[0].safetyhat_objects[kk].confidence;
        }

        std::cout << std::endl;

        /*
        std::cout <<"index: " << frame_info.out_datums[0].safetyhat_objects[0].index  << " " \
             << frame_info.out_datums[0].safetyhat_objects[1].index <<" " \
             << frame_info.out_datums[0].safetyhat_objects[2].index <<" " \
             << frame_info.out_datums[0].safetyhat_objects[3].index <<" " \
             << frame_info.out_datums[0].safetyhat_objects[4].index <<" " \
             << frame_info.out_datums[0].safetyhat_objects[5].index<<std::endl;

        std::cout <<"confi: " << std::setprecision(6) <<frame_info.out_datums[0].safetyhat_objects[0].confidence  << " " \
             << frame_info.out_datums[0].safetyhat_objects[1].confidence <<" " \
             << frame_info.out_datums[0].safetyhat_objects[2].confidence <<" " \
             << frame_info.out_datums[0].safetyhat_objects[3].confidence <<" " \
             << frame_info.out_datums[0].safetyhat_objects[4].confidence <<" " \
             << frame_info.out_datums[0].safetyhat_objects[5].confidence<<std::endl;
        */
        //free input and output tensors
        free_NetFwds(frame_info.forwards);

        if (m_nextInferPipe) {
            m_nextInferPipe->push_frame(&frame_info);
        } else {
            if (m_pfnDetectFinish != nullptr) {
                m_pfnDetectFinish(frame_info);
            }
        }

        for (int j = 0; j < frame_info.frames.size(); ++j) {

            bm_image_destroy(frame_info.frames[j].origin_image);
            frame_info.frames[j].jpeg_data = nullptr;
            for(auto simg : frame_info.frames[j].crop_bmimgs) {
                bm_image_destroy(simg);
            }

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

    }
}

bm::BMNNTensorPtr Safety_Hat_Detect::get_output_tensor(const std::string &name, bm::NetForward *inferIO, float scale)
{
    for (auto t : inferIO->output_tensors) {
        int idx = m_bmnet->outputName2Index(name);
        int output_tensor_num = m_bmnet->outputTensorNum();
        if (idx < 0 && idx > output_tensor_num-1) {
            std::cout << "ERROR:idx=" << idx << std::endl;
            assert(0);
        }
        bm::BMNNTensorPtr tensor = std::make_shared<bm::BMNNTensor>(m_bmctx->handle(), name, scale,
                                                                    &inferIO->output_tensors[idx]);
        return tensor;
    }

    return nullptr;
}

