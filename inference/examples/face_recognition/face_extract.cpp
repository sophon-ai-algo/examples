//
// Created by yuan on 6/9/21.
//

#include "face_extract.h"

FaceExtract::FaceExtract(bm::BMNNContextPtr bmctx, int maxBatch):m_bmctx(bmctx), MAX_BATCH(maxBatch)
{
    m_bmnet = std::make_shared<bm::BMNNNetwork>(m_bmctx->bmrt(), "r50i-model");
    assert(m_bmnet != nullptr);
    m_alpha_int8 = 1.003921316;
    m_beta_int8  = -127.5 * 1.003921316;
    m_alpha_fp32 = 0.0078125;
    m_beta_fp32  = -127.5 * 0.0078125;

    auto shape = m_bmnet->inputTensor(0)->get_shape();
    // for NCHW
    m_inputSize.height = shape->dims[2];
    m_inputSize.width = shape->dims[3];
}

FaceExtract::~FaceExtract()
{

}

int FaceExtract::preprocess(std::vector <bm::FrameInfo2> &frame_infos)
{
    int ret = 0;
    bm_handle_t handle = m_bmctx->handle();

    for(int frameInfoIdx = 0; frameInfoIdx < frame_infos.size(); ++frameInfoIdx) {
        auto &frame_info = frame_infos[frameInfoIdx];
        int frameNum = frame_info.frames.size();
        assert(frame_info.forwards.size() == 0);

        int total_face_num = 0;
        std::vector<bm_image> crop_images;
        //1. calculate total faces
        for (int frameIdx = 0; frameIdx < frameNum; ++frameIdx) {
            // for each frame in batch one, crop images
            //crop images by face rects
            auto &rcs = frame_info.out_datums[frameIdx].obj_rects;
            int face_num = rcs.size();

            if (face_num > 0) {
                bmcv_rect_t crop_rects[face_num];
                frame_info.frames[frameIdx].crop_bmimgs.resize(face_num);
                for (int k = 0; k < face_num; k++) {
                    rcs[k].to_bmcv_rect(&crop_rects[k]);
                    ret = bm::BMImage::create_batch(handle, crop_rects[k].crop_h, crop_rects[k].crop_w,
                                                    FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE,
                                                    &frame_info.frames[frameIdx].crop_bmimgs[k], 1, 64);
                    assert(BM_SUCCESS == ret);
                }

                ret = bmcv_image_crop(handle, face_num, crop_rects,
                                      frame_info.frames[frameIdx].origin_image,
                                      frame_info.frames[frameIdx].crop_bmimgs.data());
                assert(BM_SUCCESS == ret);

                // gather all crop images
                crop_images.insert(crop_images.end(), frame_info.frames[frameIdx].crop_bmimgs.begin(),
                                   frame_info.frames[frameIdx].crop_bmimgs.end());
            }
            total_face_num += face_num;
        }

        // to 4batch
        int left = (total_face_num % MAX_BATCH == 0 ? MAX_BATCH : total_face_num % MAX_BATCH);
        int batch_num = total_face_num % MAX_BATCH == 0 ? total_face_num / MAX_BATCH : (total_face_num / MAX_BATCH + 1);
        for (int batch_idx = 0; batch_idx < batch_num; ++batch_idx) {
            int num = MAX_BATCH;
            int start_idx = batch_idx * MAX_BATCH;
            if (batch_idx == batch_num - 1) {
                // last one
                num = left;
            }

            bm_image resized_images[MAX_BATCH];
            ret = bm::BMImage::create_batch(handle, m_inputSize.height,
                                            m_inputSize.width, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE,
                                            resized_images,
                                            num, 64);
            assert(BM_SUCCESS == ret);

            //1. Resize
            for (int i = 0; i < num; ++i) {
                ret = bmcv_image_vpp_convert(handle, 1, crop_images[start_idx + i],
                                             &resized_images[i]);
                assert(BM_SUCCESS == ret);
            }

            //2. Convert to
            bm_image convertto_imgs[MAX_BATCH];
            float alpha, beta;

            bm_image_data_format_ext img_type;
            auto inputTensorPtr = m_bmnet->inputTensor(0);
            if (inputTensorPtr->get_dtype() == BM_INT8) {
                alpha = m_alpha_int8;
                beta = m_alpha_int8;
                img_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
            } else {
                alpha = m_alpha_fp32;
                beta = m_alpha_fp32;
                img_type = DATA_TYPE_EXT_FLOAT32;
            }

            ret = bm::BMImage::create_batch(handle, m_inputSize.height, m_inputSize.width,
                                            FORMAT_RGB_PLANAR, img_type, convertto_imgs, num, 1, false, true);
            assert(BM_SUCCESS == ret);

            // create input tensor
            bm_tensor_t input_tensor = *inputTensorPtr->bm_tensor();
            bm::bm_tensor_reshape_NCHW(handle, &input_tensor, num, 3, m_inputSize.height, m_inputSize.width);

            ret = bm_image_attach_contiguous_mem(num, convertto_imgs, input_tensor.device_mem);
            assert(BM_SUCCESS == ret);

            bmcv_convert_to_attr convert_to_attr;
            convert_to_attr.alpha_0 = alpha;
            convert_to_attr.alpha_1 = alpha;
            convert_to_attr.alpha_2 = alpha;
            convert_to_attr.beta_0 = beta;
            convert_to_attr.beta_1 = beta;
            convert_to_attr.beta_2 = beta;

            ret = bmcv_image_convert_to(m_bmctx->handle(), num, convert_to_attr, resized_images, convertto_imgs);
            assert(ret == 0);

            bm_image_dettach_contiguous_mem(num, convertto_imgs);

            bm::NetForward io;
            io.batch_size = num;
            io.input_tensors.push_back(input_tensor);
            for (int l = 0; l < m_bmnet->outputTensorNum(); ++l) {
                bm_tensor_t t;
                io.output_tensors.push_back(t);
            }

            frame_info.forwards.push_back(io);

            bm::BMImage::destroy_batch(resized_images, num);
            bm::BMImage::destroy_batch(convertto_imgs, num);
        } // end of batch_idx
    } // end of frameinfo_idx

    return 0;
}

int FaceExtract::forward(std::vector <bm::FrameInfo2> &frame_infos)
{
    int ret = 0;
    for (auto &frame_info : frame_infos) {
        for (auto &fwd : frame_info.forwards) {
            if (fwd.batch_size > 0) {
                ret = m_bmnet->forward(fwd.input_tensors.data(),
                                       fwd.input_tensors.size(),
                                       fwd.output_tensors.data(),
                                       fwd.output_tensors.size());
                assert(BM_SUCCESS == ret);
            }
        }
    }

    return ret;
}

int FaceExtract::postprocess(std::vector <bm::FrameInfo2> &frame_infos)
{
    // for each batch
    for(int frameInfoIdx = 0; frameInfoIdx < frame_infos.size(); ++frameInfoIdx) {
        auto &frame_info = frame_infos[frameInfoIdx];
        // for each frame
        int start_idx = 0;
        for(int fwd_idx = 0; fwd_idx < frame_info.forwards.size(); ++fwd_idx) {
            // for each landmark batch
            int image_n = frame_info.forwards[fwd_idx].batch_size;
            if (image_n == 0) continue;
            bm::BMNNTensorPtr fc1_tensor = get_output_tensor("fc1_scale", &frame_info.forwards[fwd_idx]);
            const float *out_data = fc1_tensor->get_cpu_data();
            int out_c = fc1_tensor->get_shape()->dims[1];

            for(int n = 0; n < image_n; ++n) {
                int frame_idx =0;
                int oc_idx;
                get_complex_idx(start_idx+n, frame_info.out_datums, &frame_idx, &oc_idx);
                bm::ObjectFeature feature;
                const float *data_start = out_data + out_c * image_n;
                feature.insert(feature.end(), data_start, data_start + out_c);
                frame_info.out_datums[frame_idx].face_features.push_back(feature);
            }

            start_idx += image_n;

        } // end of frameIdx

        //free input and output tensors
        free_fwds(frame_info.forwards);

        if (m_nextInferPipe) {
            m_nextInferPipe->push_frame(&frame_info);
        }else{
            if (m_pfnDetectFinish) {
                m_pfnDetectFinish(frame_info);
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
}

int FaceExtract::get_complex_idx(int idx, std::vector<bm::NetOutputDatum> out, int *p_frameIdx, int *prc_idx)
{
    int frame_idx = 0;
    int rc_idx = 0;
    for(int i = 0;i < out.size(); ++i) {
        if (idx < out[i].obj_rects.size()) {
            frame_idx = i;
            rc_idx = idx;
            break;
        }

        idx -= out[i].obj_rects.size();
    }

    if (p_frameIdx) *p_frameIdx = frame_idx;
    if (prc_idx) *prc_idx = rc_idx;
    return 0;
}

void FaceExtract::free_fwds(std::vector<bm::NetForward> &fwds)
{
    for (auto &ios : fwds) {
        for (auto &tensor : ios.input_tensors) {
            bm_free_device(m_bmctx->handle(), tensor.device_mem);
        }

        for (auto &tensor: ios.output_tensors) {
            bm_free_device(m_bmctx->handle(), tensor.device_mem);
        }
    }
}

bm::BMNNTensorPtr FaceExtract::get_output_tensor(const std::string &name, bm::NetForward *inferIO, float scale)
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