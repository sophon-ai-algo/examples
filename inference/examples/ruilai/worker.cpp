//
// Created by yuan on 3/11/21.
//

#include "worker.h"
#include "stream_sei.h"
#include "bmutility_image.h"

OneCardInferApp::OneCardInferApp(AppStatis& statis,bm::VideoUIAppPtr gui, bm::TimerQueuePtr tq, bm::BMNNHandlePtr handle,
                                 int start_index, int num, int resize_queue_num, int skip)
  : m_detectorDelegate(nullptr), m_channel_num(num), m_appStatis(statis), m_callbackQueue(bm::TimerQueue::create()),
    m_img_result_cb_func(nullptr) {

    m_guiReceiver = gui;
    m_timeQueue = tq;
    m_channel_start = start_index;
    m_skipN = skip;
    m_dev_id = handle->dev_id();
    m_handle = handle->handle();

    m_v320ClassifyPipes.resize(3);

    m_resizeQueue = std::make_shared<BlockingQueue<bm::CropFrameInfo>>("resize", 0, 32);
    m_resizeWorkerPool.init(m_resizeQueue.get(), resize_queue_num, 4, 4);
    m_resizeWorkerPool.startWork([this](std::vector<bm::CropFrameInfo> &items) {
//        assert(items.size() == 4);
//        std::vector<bm_image> resized_image_224;
//        std::vector<bm_image> resized_image_320;
//        this->unifyResizeProcess(items, resized_image_224, resized_image_320);
//        for (int i = 0; i < items.size(); ++i) {
//            bm_image_destroy(items[i].crop_img);
//        }

        // 224x224 Input
        bm::ResizeFrameInfo rfi_224;
        bm::ResizeFrameInfo rfi_320;
        for (int i = 0; i < items.size(); ++i) {
            rfi_224.v_seq.push_back(items[i].seq);
            rfi_224.v_resized_imgs.push_back(items[i].crop_img_224);
            //rfi_320.v_resized_imgs.push_back(items[i].crop_img_320);
        }
        //rfi_224.v_resized_imgs.swap(resized_image_224);
        m_224ClassifyPipe.push_frame(&rfi_224);
        //m_v320ClassifyPipes[0].push_frame(&rfi_320);

        // 320x320 Input
//        for(int i = 0; i < m_v320ClassifyPipes.size() - 1; ++i) {
//            bm::ResizeFrameInfo rfi_320;
//            rfi_320.v_resized_imgs.resize(resized_image_320.size());
//            bm::BMImage::bm_images_clone(
//                    m_handle, resized_image_320.data(), resized_image_320.size(), rfi_320.v_resized_imgs.data(), 64);
//            m_v320ClassifyPipes[i].push_frame(&rfi_320);
//        }
//        if (m_v320ClassifyPipes.size() > 0) {
//            bm::ResizeFrameInfo rfi_320;
//            rfi_320.v_resized_imgs.swap(resized_image_320);
//            m_v320ClassifyPipes[m_v320ClassifyPipes.size() - 1].push_frame(&rfi_320);
//        }
    });
}

void OneCardInferApp::start(const std::vector<std::string>& urls, Config& config)
{
    m_detectorDelegate->set_detected_callback([this](bm::FrameInfo &frame_info) {
        int ret = 0;
        int total_face_num = 0;
        std::vector<bm::CropFrameInfo> total_crop_images;
        for (int frameIdx = 0; frameIdx < frame_info.out_datums.size(); ++frameIdx) {
            auto &rcs = frame_info.out_datums[frameIdx].obj_rects;
            int face_num = rcs.size();
            //std::cout << "Detected faces: " << rcs.size() << std::endl;
            if (face_num > 0) {
                bmcv_rect_t crop_rects[face_num];
                bmcv_padding_atrr_t padding_attr[face_num];
                std::vector<bm_image> crop_images_224;
                std::vector<bm_image> crop_images_320;
                crop_images_224.resize(face_num);
                crop_images_320.resize(face_num);

                for (int k = 0; k < face_num; k++) {
                    rcs[k].to_bmcv_rect(&crop_rects[k], frame_info.frames[frameIdx].original.width, frame_info.frames[frameIdx].height);

                    padding_attr[k].padding_r    = 128;
                    padding_attr[k].padding_g    = 128;
                    padding_attr[k].padding_b    = 128;
                    padding_attr[k].if_memset    = 0;
                    padding_attr[k].dst_crop_stx = 0;
                    padding_attr[k].dst_crop_sty = 0;
                    padding_attr[k].dst_crop_h   = crop_rects[k].crop_h;
                    padding_attr[k].dst_crop_w   = crop_rects[k].crop_w;

//                    ret = bm::BMImage::create_batch(m_handle, crop_rects[k].crop_h, crop_rects[k].crop_w,
//                                                    FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE,
//                                                    &crop_images[k], 1, 64);
                    ret = bm::BMImage::create_batch(m_handle, 224, 224,
                                                    FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE,
                                                    &crop_images_224[k], 1, 64);
                    assert(BM_SUCCESS == ret);

//                    ret = bm::BMImage::create_batch(m_handle, 224, 224,
//                                                    FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE,
//                                                    &crop_images_320[k], 1, 64);
//                    assert(BM_SUCCESS == ret);
                }
                bm::BMPerf p1("crop", 10);

                // crop faces
//                ret = bmcv_image_vpp_convert_padding(m_handle, face_num,
//                                                     frame_info.frames[frameIdx].original,
//                                                     crop_images.data(),
//                                                     padding_attr,
//                                                     crop_rects);
                 ret = bmcv_image_vpp_convert(m_handle, face_num,
                                              frame_info.frames[frameIdx].original,
                                              crop_images_224.data(),
                                              crop_rects);
                 assert(BM_SUCCESS == ret);
//                 ret = bmcv_image_vpp_convert(m_handle, face_num,
//                                             frame_info.frames[frameIdx].original,
//                                             crop_images_320.data(),
//                                             crop_rects);
                // ret = bmcv_image_crop(handle, face_num, crop_rects,
                //                       frame_info.frames[frameIdx].original,
                //                       crop_images.data());
                p1.end();

                assert(BM_SUCCESS == ret);

                // gather all crop images
                for (int i = 0; i < face_num; ++i) {
                    bm::CropFrameInfo cfi;
                    cfi.chan_id  = frame_info.frames[frameIdx].chan_id;
                    cfi.seq      = frame_info.frames[frameIdx].seq;
                    cfi.crop_img_224 = crop_images_224[i];
                    //cfi.crop_img_320 = crop_images_320[i];
//                    char vv[256];
//                    static int ii = 0;
//                    snprintf(vv, 256, "output_%d.bmp", );
//                    call(bm_image_write_to_bmp, crop_images_224[i], vv);
//                    total_crop_images.push_back(cfi);
                    m_resizeQueue->push(cfi);
                }
            }
            total_face_num += face_num;
        }
        //m_resizeQueue->push(total_crop_images);

        
        for (int i = 0; i < frame_info.frames.size(); ++i) {
            int ch = frame_info.frames[i].chan_id;
            m_appStatis.m_chan_statis[ch]++;
            m_appStatis.m_statis_lock.lock();
            m_appStatis.m_total_statis++;
            m_appStatis.m_statis_lock.unlock();
        }
    });

    m_classifyDelegate224->set_classified_callback([this](bm::ResizeFrameInfo& frame_info){
        bm::BMNNTensor output_tensor(m_handle, "", 1.0, &frame_info.output_tensors[0]);
        float *data = output_tensor.get_cpu_data();
        auto output_shape = output_tensor.get_shape();
        int batch_size = output_shape->dims[0];
        int class_num = output_shape->dims[1];
        int total = batch_size * class_num;
        std::map<uint64_t, float> score_record;
        for (int i = 0; i < frame_info.v_resized_imgs.size(); ++i) {
            uint64_t seq = frame_info.v_seq[i];
            auto iter = score_record.find(seq);
            if (iter == score_record.end()) {
                score_record[seq] = data[i*2 + 1];
            } else if (iter->second < data[i*2 + 1]) {
                iter->second = data[i*2 + 1];
            }
        }
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        for (auto iter = score_record.begin(); iter != score_record.end(); ++iter) {
            uint64_t seq = iter->first;
            if (m_image_score_record.find(seq) == m_image_score_record.end()) {
                m_image_score_record[seq] = iter->second;
                if (m_img_result_cb_func != nullptr) {
                    m_callbackQueue->create_timer(10, [this, seq]() {
                        std::cout << "result callback.." << std::endl;
                        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
                        m_img_result_cb_func(seq, m_image_score_record[seq] > 0.5, m_image_score_record[seq]);
                    }, 0, nullptr);
                }

            } else {
                m_image_score_record[seq] = iter->second;
            }
        }

    });

    ruilai::DetectorParam param;
    loadConfig<ruilai::DetectorParam>(param, config);
    // detector init
    m_inferPipe.init(param, m_detectorDelegate);
    // classify init
    initClassifyPipes(config);


    for(int i = 0; i < m_channel_num; ++i) {
        int ch = m_channel_start + i;
        std::cout << "push id=" << ch << std::endl;
        TChannelPtr pchan = std::make_shared<TChannel>();
        pchan->decoder = new bm::StreamDecoder(ch);
        pchan->channel_id = ch;

        std::string media_file;
        AVDictionary *opts = NULL;
        av_dict_set_int(&opts, "sophon_idx", m_dev_id, 0);
        //av_dict_set(&opts, "output_format", "101", 18);
        av_dict_set(&opts, "extra_frame_buffer_num", "10", 0);

        // pchan->decoder->set_avformat_opend_callback([this, pchan](AVFormatContext *ifmt) {
        // });

        // pchan->decoder->set_avformat_closed_callback([this, pchan]() {
        // });

        pchan->decoder->open_stream(urls[i % urls.size()], true, opts);
        av_dict_free(&opts);
        pchan->decoder->set_decoded_frame_callback([this, pchan, ch](const AVPacket* pkt, const AVFrame *frame){
            bm::FrameBaseInfo fbi;
            fbi.avframe = av_frame_alloc();
            fbi.avpkt = av_packet_alloc();
            av_frame_ref(fbi.avframe, frame);
            av_packet_ref(fbi.avpkt, pkt);
            fbi.seq = pchan->seq++;
            if (m_skipN > 0) {
                if (fbi.seq % m_skipN != 0) fbi.skip = true;
            }
            fbi.chan_id = ch;
#ifdef DEBUG
            if (ch == 0) std::cout << "decoded frame " << std::endl;
#endif
            m_detectorDelegate->decode_process(fbi);
            m_inferPipe.push_frame(&fbi);
        });

        m_chans[ch] = pchan;
    }
}

void OneCardInferApp::unifyResizeProcess(std::vector<bm::CropFrameInfo> &items,
                                         std::vector<bm_image>& resized_image_224,
                                         std::vector<bm_image>& resized_image_320) {
    int num = items.size();
    // Alloc resized image & input image
    resized_image_224.resize(num);
    resized_image_320.resize(num);
    
    const size_t align = 64;
    const bool alloc_mem = true;

    call(
        bm::BMImage::create_batch,
        m_handle, 224, 224,
        FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE,
        resized_image_224.data(), num,
        align, alloc_mem, true, 1);
//    call(
//        bm::BMImage::create_batch,
//        m_handle, 224, 224,
//        FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE,
//        resized_image_320.data(), num,
//        align, alloc_mem, true, 1);
    bm::BMPerf p1("resize", 1);

    for (int i = 0; i < num; ++i) {
        bm_image image = items[i].crop_img_224;
        // Resize image
        call(bmcv_image_vpp_basic, m_handle, 1, &image, &resized_image_224[i]);
        //call(bmcv_image_vpp_basic, m_handle, 1, &image, &resized_image_320[i]);
    }
    p1.end();
}

void OneCardInferApp::initClassifyPipes(Config& config) {
    ruilai::ClassifyParam param;
    loadConfig<ruilai::ClassifyParam>(param, config);
    param.batch_num = 1;

    // 224 model
    m_224ClassifyPipe.init(param, m_classifyDelegate224);

    // 320 model
    //assert(m_vClassifyDelegate320.size() == m_v320ClassifyPipes.size());
    for (int i = 0; i < m_vClassifyDelegate320.size(); ++i) {
        m_v320ClassifyPipes[i].init(param, m_vClassifyDelegate320[i]);
    }

}