//
// Created by yuan on 2/23/21.
//

#include "face_detector.h"
#include <algorithm>
#include "bm_wrapper.hpp"

#define DUMP_FILE 0
#define DYNAMIC_SIZE 0

static inline bool compareBBox(const bm::NetOutputObject &a, const bm::NetOutputObject &b) {
    return a.score > b.score;
}

FaceDetector::FaceDetector(bm::BMNNContextPtr bmctx, int max_batch)
{
    bmctx_ = bmctx;
    anchor_ratios_.push_back(1.0f);
    anchor_scales_.push_back(1);
    anchor_scales_.push_back(2);
    anchor_num_ = 2;
    is4N_ = false;

    bmnet_ = std::make_shared<bm::BMNNNetwork>(bmctx_->bmrt(), "SqueezeNet");
    assert(bmnet_ != nullptr);

    auto tensor = bmnet_->inputTensor(0);
    m_net_h = 400; // static net
    m_net_w = 711; // static net


}

FaceDetector::~FaceDetector()
{

}

void FaceDetector::calc_resized_HW(int image_h, int image_w, int *p_h, int *p_w) {
    int im_size_min = std::min(image_h, image_w);
    int im_size_max = std::max(image_h, image_w);
    im_scale_  = target_size_ / im_size_min;
    if (im_scale_ * im_size_max > max_size_) {
        im_scale_ = max_size_ / im_size_max;
    }

#if DYNAMIC_SIZE

    img_x_scale_ = im_scale_;
    img_y_scale_ =  im_scale_;

    *p_h = (int)(image_h * im_scale_ + 0.5);
    *p_w = (int)(image_w * im_scale_ + 0.5);
#else

    float vw, vh;
    if (image_h > image_w) {
        vh = 711.0;
        vw = 400.0;
    }else{
        vh = 400.0;
        vw = 711.0;
    }

    *p_h = vh;
    *p_w = vw;

    img_x_scale_ = vw / image_w;
    img_y_scale_ =  vh / image_h;
#endif
}


int FaceDetector::preprocess(std::vector<bm::FrameInfo2> &frame_infos)
{
#if 1
    int ret = 0;
    bm_handle_t handle = bmctx_->handle();
    calc_resized_HW(1080, 1920, &m_net_h, &m_net_w);

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
        ret = bm::BMImage::create_batch(handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR,
                                        DATA_TYPE_EXT_1N_BYTE, resized_imgs, num, 64);
        assert(BM_SUCCESS == ret);

        for (int i = 0; i < num; ++i) {
            bm_image image1;
            //FrameBaseInfo frameBaseInfo;
            bm::BMImage::from_avframe(handle, vecFrameBaseInfo[start_idx + i].avframe, image1, true);

            ret = bm::BMImage::create_batch(handle, image1.height, image1.width, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE,
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
        auto tensor = bmnet_->inputTensor(0);
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
        convert_to_attr.beta_0 = beta + B_bias;
        convert_to_attr.beta_1 = beta + G_bias;
        convert_to_attr.beta_2 = beta + R_bias;

        ret = bmcv_image_convert_to(bmctx_->handle(), num, convert_to_attr, resized_imgs, convertto_imgs);
        assert(ret == 0);

        bm_image_dettach_contiguous_mem(num, convertto_imgs);

        bm::NetForward netfwd;
        netfwd.batch_size = num;
        netfwd.input_tensors.push_back(input_tensor);
        for (int l = 0; l < bmnet_->outputTensorNum(); ++l) {
            bm_tensor_t t;
            netfwd.output_tensors.push_back(t);
        }
        finfo.forwards.push_back(netfwd);

        bm::BMImage::destroy_batch(resized_imgs, num);
        bm::BMImage::destroy_batch(convertto_imgs, num);

        frame_infos.push_back(finfo);
    }


#else
    FrameInfo frame_info;
    for(int i = 0;i < frames.size(); ++i) {
        FrameBaseInfo finfo;
        finfo.avframe = frames[i].avframe;
        finfo.avpkt = frames[i].avpkt;
        frame_info.frames.push_back(finfo);
    }

    frame_infos.push_back(frame_info);

#endif


    return 0;
}

int FaceDetector::forward(std::vector<bm::FrameInfo2>& frame_infos)
{
    int ret = 0;
    for(int b = 0; b < frame_infos.size(); ++b) {
        for(int l = 0; l < frame_infos[b].forwards.size(); ++l) {
            ret = bmnet_->forward(frame_infos[b].forwards[l].input_tensors.data(),
                    frame_infos[b].forwards[l].input_tensors.size(),
                    frame_infos[b].forwards[l].output_tensors.data(),
                    frame_infos[b].forwards[l].output_tensors.size());
            assert(BM_SUCCESS == ret);
        }
    }

    return 0;
}

int FaceDetector::postprocess(std::vector<bm::FrameInfo2> &frame_infos)
{
    int ret = 0;
    for(int frameInfoIdx =0; frameInfoIdx < frame_infos.size(); ++frameInfoIdx) {

        // Free AVFrames
        auto &frame_info = frame_infos[frameInfoIdx];

        // extract face detection
        extract_facebox_cpu(frame_info);

        // Free Tensors
        for (auto &ios : frame_info.forwards) {
            for (auto &tensor : ios.input_tensors) {
                bm_free_device(bmctx_->handle(), tensor.device_mem);
            }

            for (auto &tensor: ios.output_tensors) {
                bm_free_device(bmctx_->handle(), tensor.device_mem);
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
#if 1
            // transfer to next pipe
            m_nextInferPipe->push_frame(&frame_info);
#else
            if (m_pfnDetectFinish != nullptr) {
                m_pfnDetectFinish(frame_info);
            }


            for (int j = 0; j < frame_info.frames.size(); ++j) {

                bm_image_destroy(frame_info.frames[j].origin_image);
                frame_info.frames[j].jpeg_data = nullptr;

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
#endif
        }

    }
}

bm::BMNNTensorPtr FaceDetector::get_output_tensor(const std::string &name, bm::FrameInfo2& frame_info, float scale, int k) {
    int output_tensor_num = frame_info.forwards[k].output_tensors.size();
    int idx = bmnet_->outputName2Index(name);
    if (idx < 0 && idx > output_tensor_num-1) {
        std::cout << "ERROR:idx=" << idx << std::endl;
        assert(0);
    }
    bm::BMNNTensorPtr tensor = std::make_shared<bm::BMNNTensor>(bmctx_->handle(), name, scale,
                                  &frame_info.forwards[k].output_tensors[idx]);
    return tensor;
}

int FaceDetector::extract_facebox_cpu(bm::FrameInfo2 &frame_info)
{
    int image_n = frame_info.frames.size();

    float m3_scale_to_float = 0.00587051f;
    float m2_scale_to_float = 0.00527f;
    float m1_scale_to_float = 0.00376741f;

    assert(frame_info.forwards.size() == 1);

    bm::BMNNTensorPtr m3_cls_tensor = get_output_tensor("m3@ssh_cls_prob_reshape_output", frame_info);
    bm::BMNNTensorPtr m3_bbox_tensor = get_output_tensor("m3@ssh_bbox_pred_output", frame_info, m3_scale_to_float);
    bm::BMNNTensorPtr m2_cls_tensor = get_output_tensor("m2@ssh_cls_prob_reshape_output", frame_info);
    bm::BMNNTensorPtr m2_bbox_tensor = get_output_tensor("m2@ssh_bbox_pred_output", frame_info, m2_scale_to_float);
    bm::BMNNTensorPtr m1_cls_tensor = get_output_tensor("m1@ssh_cls_prob_reshape_output", frame_info);
    bm::BMNNTensorPtr m1_bbox_tensor = get_output_tensor("m1@ssh_bbox_pred_output", frame_info, m1_scale_to_float);

    // NCHW
    int m3_c = m3_cls_tensor->get_shape()->dims[1];
    int m3_w = m3_cls_tensor->get_shape()->dims[3];
    int m3_h = m3_cls_tensor->get_shape()->dims[2];

    int m2_c = m2_cls_tensor->get_shape()->dims[1];
    int m2_w = m2_cls_tensor->get_shape()->dims[3];
    int m2_h = m2_cls_tensor->get_shape()->dims[2];

    int m1_c = m1_cls_tensor->get_shape()->dims[1];
    int m1_w = m1_cls_tensor->get_shape()->dims[3];
    int m1_h = m1_cls_tensor->get_shape()->dims[2];

    int b3_c = m3_bbox_tensor->get_shape()->dims[1];
    int b3_w = m3_bbox_tensor->get_shape()->dims[3];
    int b3_h = m3_bbox_tensor->get_shape()->dims[2];

    int b2_c = m2_bbox_tensor->get_shape()->dims[1];
    int b2_w = m2_bbox_tensor->get_shape()->dims[3];
    int b2_h = m2_bbox_tensor->get_shape()->dims[2];

    int b1_c = m1_bbox_tensor->get_shape()->dims[1];
    int b1_w = m1_bbox_tensor->get_shape()->dims[3];
    int b1_h = m1_bbox_tensor->get_shape()->dims[2];

    const float *m3_scores      = (float*)m3_cls_tensor->get_cpu_data();
    const float *m2_scores      = (float*)m2_cls_tensor->get_cpu_data();
    const float *m1_scores      = (float*)m1_cls_tensor->get_cpu_data();
    const float *m3_bbox_deltas = (float*)m3_bbox_tensor->get_cpu_data();
    const float *m2_bbox_deltas = (float*)m2_bbox_tensor->get_cpu_data();
    const float *m1_bbox_deltas = (float*)m1_bbox_tensor->get_cpu_data();

    for (int n = 0; n < image_n; n++) {
        int                   width  = frame_info.frames[n].width;
        int                   height = frame_info.frames[n].height;
        std::vector<bm::NetOutputObject> proposals;
        proposals.clear();
        generate_proposal(m3_scores + (m3_c * n + anchor_num_) * m3_h * m3_w,
                          m3_bbox_deltas + b3_c * b3_h * b3_w * n,
                          16.0,
                          4,
                          m3_w,
                          m3_h,
                          width,
                          height,
                          proposals);
        generate_proposal(m2_scores + (m2_c * n + anchor_num_) * m2_h * m2_w,
                          m2_bbox_deltas + b2_c * b2_h * b2_w * n,
                          4.0,
                          2,
                          m2_w,
                          m2_h,
                          width,
                          height,
                          proposals);
        generate_proposal(m1_scores + (m1_c * n + anchor_num_) * m1_h * m1_w,
                          m1_bbox_deltas + b1_c * b1_h * b1_w * n,
                          1.0,
                          1,
                          m1_w,
                          m1_h,
                          width,
                          height,
                          proposals);
        std::vector<bm::NetOutputObject> nmsProposals;
        nmsProposals.clear();
        nms(proposals, nmsProposals);

        std::vector<bm::NetOutputObject> faceRects;
        faceRects.clear();
        for (size_t i = 0; i < nmsProposals.size(); ++i) {
            bm::NetOutputObject rect = nmsProposals[i];
            if (rect.score >= 0.7)
                faceRects.push_back(rect);
        }

        frame_info.out_datums.push_back(bm::NetOutputDatum(faceRects));
        //std::cout << "Image idx=" << n << " final predict " << faceRects.size() << " bboxes" << std::endl;
    }

    return 0;
}

void FaceDetector::generate_proposal(const float *          scores,
                                        const float *          bbox_deltas,
                                        const float            scale_factor,
                                        const int              feat_factor,
                                        const int              feat_w,
                                        const int              feat_h,
                                        const int              width,
                                        const int              height,
                                        std::vector<bm::NetOutputObject> &proposals)
{
    std::vector<bm::NetOutputObject> m_proposals;
    float                 anchor_cx   = (base_size_ - 1) * 0.5;
    float                 anchor_cy   = (base_size_ - 1) * 0.5;
    int                   feat_stride = feat_stride_ * feat_factor;

    for (int s = 0; s < anchor_scales_.size(); ++s) {
        float scale = anchor_scales_[s] * scale_factor;
        for (int h = 0; h < feat_h; ++h) {
            for (int w = 0; w < feat_w; ++w) {
                int      delta_index = h * feat_w + w;
                bm::NetOutputObject facerect;
                facerect.score = scores[s * feat_w * feat_h + delta_index];
                if (facerect.score <= base_threshold_)
                    continue;
                float anchor_size = scale * base_size_;
                float bbox_x1 =
                        anchor_cx - (anchor_size - 1) * 0.5 + w * feat_stride;
                float bbox_y1 =
                        anchor_cy - (anchor_size - 1) * 0.5 + h * feat_stride;
                float bbox_x2 =
                        anchor_cx + (anchor_size - 1) * 0.5 + w * feat_stride;
                float bbox_y2 =
                        anchor_cy + (anchor_size - 1) * 0.5 + h * feat_stride;

                float bbox_w  = bbox_x2 - bbox_x1 + 1;
                float bbox_h  = bbox_y2 - bbox_y1 + 1;
                float bbox_cx = bbox_x1 + 0.5 * bbox_w;
                float bbox_cy = bbox_y1 + 0.5 * bbox_h;
                float dx =
                        bbox_deltas[(s * 4 + 0) * feat_h * feat_w + delta_index];
                float dy =
                        bbox_deltas[(s * 4 + 1) * feat_h * feat_w + delta_index];
                float dw =
                        bbox_deltas[(s * 4 + 2) * feat_h * feat_w + delta_index];
                float dh =
                        bbox_deltas[(s * 4 + 3) * feat_h * feat_w + delta_index];
                float pred_cx = dx * bbox_w + bbox_cx;
                float pred_cy = dy * bbox_h + bbox_cy;
                float pred_w  = std::exp(dw) * bbox_w;
                float pred_h  = std::exp(dh) * bbox_h;
                facerect.x1 =
                        std::max(std::min(static_cast<double>(width - 1),
                                          (pred_cx - 0.5 * pred_w) / im_scale_),
                                 0.0);
                facerect.y1 =
                        std::max(std::min(static_cast<double>(height - 1),
                                          (pred_cy - 0.5 * pred_h) / im_scale_),
                                 0.0);
                facerect.x2 =
                        std::max(std::min(static_cast<double>(width - 1),
                                          (pred_cx + 0.5 * pred_w) / im_scale_),
                                 0.0);
                facerect.y2 =
                        std::max(std::min(static_cast<double>(height - 1),
                                          (pred_cy + 0.5 * pred_h) / im_scale_),
                                 0.0);
                if ((facerect.x2 - facerect.x1 + 1 < min_size_) ||
                    (facerect.y2 - facerect.y1 + 1 < min_size_))
                    continue;
                m_proposals.push_back(facerect);
            }
        }
    }
    std::sort(m_proposals.begin(), m_proposals.end(), compareBBox);

    int keep = m_proposals.size();
    if (per_nms_topn_ < keep)
        keep = per_nms_topn_;

    if (keep > 0) {
        proposals.insert(
                proposals.end(), m_proposals.begin(), m_proposals.begin() + keep);
    }
}


void FaceDetector::nms(const std::vector<bm::NetOutputObject> &proposals,
                          std::vector<bm::NetOutputObject>&      nmsProposals)
{
    if (proposals.empty()) {
        nmsProposals.clear();
        return;
    }
    std::vector<bm::NetOutputObject> bboxes = proposals;
    std::sort(bboxes.begin(), bboxes.end(), compareBBox);

    int              select_idx = 0;
    int              num_bbox   = bboxes.size();
    std::vector<int> mask_merged(num_bbox, 0);
    bool             all_merged = false;
    while (!all_merged) {
        while (select_idx < num_bbox && 1 == mask_merged[select_idx])
            ++select_idx;

        if (select_idx == num_bbox) {
            all_merged = true;
            continue;
        }
        nmsProposals.push_back(bboxes[select_idx]);
        mask_merged[select_idx] = 1;
        bm::NetOutputObject select_bbox    = bboxes[select_idx];
        float    area1          = (select_bbox.x2 - select_bbox.x1 + 1) *
                                  (select_bbox.y2 - select_bbox.y1 + 1);
        ++select_idx;
        for (int i = select_idx; i < num_bbox; ++i) {
            if (mask_merged[i] == 1)
                continue;
            bm::NetOutputObject &bbox_i = bboxes[i];
            float     x      = std::max(select_bbox.x1, bbox_i.x1);
            float     y      = std::max(select_bbox.y1, bbox_i.y1);
            float     w      = std::min(select_bbox.x2, bbox_i.x2) - x + 1;
            float     h      = std::min(select_bbox.y2, bbox_i.y2) - y + 1;
            if (w <= 0 || h <= 0)
                continue;
            float area2 =
                    (bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1);
            float area_intersect = w * h;
            // Union method
            if (area_intersect / (area1 + area2 - area_intersect) >
                nms_threshold_)
                mask_merged[i] = 1;
        }
    }
}