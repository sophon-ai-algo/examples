#include "macros.h"
#include "retinaface.h"
#include "inference.h"
#include "bm_wrapper.hpp"

#ifdef USE_EXIV2
#include <exiv2/exiv2.hpp>
#include <opencv2/opencv.hpp>
#endif

#include <glog/logging.h>

#define align_with(v, n) ((int(v) + n - 1) / n * n)

const size_t vpp_limit = 4096;

struct RetinafaceImpl {
    bm::BMNNContextPtr ctx;
    bm::BMNNNetworkPtr net;
    bool is_dynamic;
    bool keep_original;
    size_t batch_size, width, height, output_num;
    float aspect_ratio;
    float nms_threshold;
    float conf_threshold;
    bmcv_convert_to_attr convert_attr;
    bm_data_type_t dtype;
    bm_image_data_format_ext img_type;
    size_t target_size;
    std::shared_ptr<bm::DeviceMemoryPool> input_pool;
    std::shared_ptr<bm::DeviceMemoryPool> resized_pool;
    std::shared_ptr<bm::DeviceMemoryPool> tensor_pool;

    bm_image opencv_decode_and_resize(const uint8_t *bin, size_t size);
};

Retinaface::Retinaface(
    bm::BMNNContextPtr ctx,
    bool keep_original,
    float nms_threshold,
    float conf_threshold,
    std::string net_name,
    bm::Watch *w) : w_(w)
{
    impl_ = new RetinafaceImpl;
    impl_->keep_original = keep_original;
    impl_->conf_threshold = conf_threshold;
    impl_->nms_threshold = nms_threshold;
    impl_->ctx = ctx;

    bm_handle_t handle = impl_->ctx->handle();
    impl_->input_pool = std::make_shared<bm::DeviceMemoryPool>(handle);
    impl_->resized_pool = std::make_shared<bm::DeviceMemoryPool>(handle);
    impl_->tensor_pool = std::make_shared<bm::DeviceMemoryPool>(handle);

    if (net_name.empty())
    {
        net_name = ctx->network_name(0);
    }
    impl_->net = std::make_shared<bm::BMNNNetwork>(ctx->bmrt(), net_name);
    impl_->is_dynamic = impl_->net->is_dynamic();
    impl_->output_num = impl_->net->outputTensorNum();
    auto input_tensor = impl_->net->inputTensor(0);
    bm_shape_t input_shape = *input_tensor->get_shape();
    if (input_shape.num_dims != 4)
    {
        LOG(ERROR) << "invalid input dims " << input_shape.num_dims;
        throw std::runtime_error("invalid model");
    }
    impl_->batch_size = input_shape.dims[0];
    impl_->height = input_shape.dims[2];
    impl_->width = input_shape.dims[3];
    impl_->aspect_ratio = impl_->width * 1. / impl_->height;
    impl_->dtype = impl_->net->get_input_dtype(0);
    impl_->img_type = impl_->dtype == BM_INT8 ?
        DATA_TYPE_EXT_1N_BYTE_SIGNED :
        DATA_TYPE_EXT_FLOAT32;

    float mean[3] = {104, 117, 123};
    float scale[3] = {1, 1, 1};
    float input_scale = input_tensor->get_scale();
    memset(&impl_->convert_attr, 0, sizeof(bmcv_convert_to_attr));
    impl_->convert_attr.alpha_0 = input_scale * scale[0];
    impl_->convert_attr.alpha_1 = input_scale * scale[1];
    impl_->convert_attr.alpha_2 = input_scale * scale[2];
    impl_->convert_attr.beta_0 = -input_scale * scale[0] * mean[0];
    impl_->convert_attr.beta_1 = -input_scale * scale[1] * mean[1];
    impl_->convert_attr.beta_2 = -input_scale * scale[2] * mean[2];

}

Retinaface::~Retinaface()
{
    delete impl_;
}

bm_image RetinafaceImpl::opencv_decode_and_resize(const uint8_t *bin, size_t size)
{
    cv::Mat image, out_image;
    std::vector<uint8_t> input(bin, bin + size);
    //image = cv::imdecode(input, cv::IMREAD_RETRY_SOFTDEC | cv::IMREAD_COLOR);
    image = cv::imdecode(input, cv::IMREAD_COLOR);
    bm_image bimage;
    cv::bmcv::toBMI(image, &bimage);
    return bimage;
}

bm_image Retinaface::read_image(bm::FrameBaseInfo &frame)
{
    bm_handle_t handle = impl_->ctx->handle();
    bm_image image;
    if (frame.filename.empty())
    {
        LOG(FATAL) << "empty filename";
        throw std::runtime_error("invalid argument");
    }
    frame.jpeg_data = bm::read_binary(frame.filename);
    void *data_ptr = frame.jpeg_data->ptr<uint8_t>();
    size_t num = 1, data_size = frame.jpeg_data->size();
#ifdef USE_EXIV2
    uint8_t *bin = frame.jpeg_data->ptr<uint8_t>();
    Exiv2::Image::AutoPtr exiv = Exiv2::ImageFactory::open(bin, data_size);
    exiv->readMetadata();
    size_t w = exiv->pixelWidth();
    size_t h = exiv->pixelHeight();

    frame.original_width = w;
    frame.original_height = h;
    if (w > vpp_limit || h > vpp_limit)
    {
        LOG(INFO) << "use OpenCV to process " << frame.filename;
        image = impl_->opencv_decode_and_resize(bin, data_size);
    } else {
        int strides[3];
        const bm_image_format_ext format = FORMAT_YUV420P;
        strides[0] = align_with(w, 32);
        strides[1] = strides[2] = strides[0] / 2;
        const bm_image_data_format_ext dtype = DATA_TYPE_EXT_1N_BYTE;
        const int num = 1;
        const int align = 32;
        const int heap = 1;
        call(
            bm_image_create, handle,
            align_with(h, 2),
            align_with(w, 2),
            format, dtype, &image, strides);
#endif
    call(
        bmcv_image_jpeg_dec,
        handle,
        &data_ptr, &data_size,
        num, &image);
#ifdef USE_EXIV2
    }
#endif
    frame.jpeg_data.reset();
    frame.original = image;
    return image;
}

void Retinaface::decode_process(bm::FrameBaseInfo &frame)
{
    bm_handle_t handle = impl_->ctx->handle();
    call(bm::BMImage::from_avframe, handle, frame.avframe, frame.original);
    av_frame_unref(frame.avframe);
    av_frame_free(&frame.avframe);
}

int Retinaface::preprocess(
    std::vector<bm::FrameBaseInfo> &frames,
    std::vector<bm::FrameInfo> &frame_infos)
{
    if (w_) w_->mark("preprocess");
    bm_handle_t handle = impl_->ctx->handle();
    std::vector<bm_image> resized_images(impl_->batch_size);
    const size_t align = 32;
    const size_t heap = 1;
    impl_->resized_pool->alloc(
        impl_->batch_size, resized_images.data(),
        impl_->height, impl_->width,
        FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE,
        align, heap);
    for (int i = 0; i < frames.size(); i += impl_->batch_size)
    {
        bm::FrameInfo frame_info;
        int num = std::min<int>(frames.size() - i, impl_->batch_size);
        int infer_batch_size = impl_->is_dynamic ? num : impl_->batch_size;
        std::vector<bm_image> input_images(infer_batch_size);
        const size_t align = 1;
        const int heap = 0;
        impl_->input_pool->alloc(
            infer_batch_size, input_images.data(),
            impl_->height, impl_->width,
            FORMAT_BGR_PLANAR, impl_->img_type,
            align, heap);
        bm_tensor_t input_tensor = *impl_->net->inputTensor(0)->bm_tensor();
        call(
            bm_image_get_contiguous_device_mem, infer_batch_size,
            input_images.data(), &input_tensor.device_mem);
        input_tensor.shape.dims[0] = infer_batch_size;
        frame_info.input_tensors.push_back(input_tensor);
        for (int j = 0; j < num; ++j)
        {
            auto &frame = frames[i + j];
            bm_image image;
            if (frame.original.width) {
                image = frame.original;
            } else if (!frame.filename.empty())
            {
                image = this->read_image(frame);
            } else if (frame.avframe) {
                call(bm::BMImage::from_avframe, handle, frame.avframe, image);
                av_frame_unref(frame.avframe);
                av_frame_free(&frame.avframe);
                frame.original = image;
            } else {
                LOG(FATAL) << "No input";
            }
            frame.width = image.width;
            frame.height = image.height;
            bmcv_padding_atrr_t padding_attr;
            memset(&padding_attr, 0, sizeof(bmcv_padding_atrr_t));
            padding_attr.padding_r = 128;
            padding_attr.padding_g = 128;
            padding_attr.padding_b = 128;
            padding_attr.if_memset = 0;
            float image_aspect_ratio = image.width * 1. / image.height;
            if (image_aspect_ratio > impl_->aspect_ratio)
            {
                padding_attr.dst_crop_w = impl_->width;
                padding_attr.dst_crop_h = round(impl_->width / image_aspect_ratio);
#ifdef CENTER_ROI
                padding_attr.dst_crop_sty = (impl_->height - padding_attr.dst_crop_h) / 2;
                frame.y_offset = padding_attr.dst_crop_sty * 1. / impl_->height;
#endif
                frame.y_scale = padding_attr.dst_crop_h * 1. / impl_->height;
            } else {
                padding_attr.dst_crop_h = impl_->height;
                padding_attr.dst_crop_w = round(impl_->height * image_aspect_ratio);
#ifdef CENTER_ROI
                padding_attr.dst_crop_stx = (impl_->width - padding_attr.dst_crop_w) / 2;
                frame.x_offset = padding_attr.dst_crop_stx * 1. / impl_->width;
#endif
                frame.x_scale = padding_attr.dst_crop_w * 1. / impl_->width;
            }
            const int vpp_num = 1;
            bmcv_rect_t crop_rect{0, 0, image.width, image.height};
            if (w_) w_->mark("vpp");
            call(
                bmcv_image_vpp_convert_padding,
                handle, vpp_num,
                image, &resized_images[j],
                &padding_attr, &crop_rect);
            if (w_) w_->mark("vpp");
            if (!impl_->keep_original)
            {
                call(bm_image_destroy, image);
            }
            frame_info.frames.push_back(frame);
        }
        call(
            bmcv_image_convert_to,
            handle, num, impl_->convert_attr, resized_images.data(), input_images.data());
        frame_infos.push_back(std::move(frame_info));
    }
    impl_->resized_pool->free(resized_images.size(), resized_images.data());
    if (w_) w_->mark("preprocess");
}

int Retinaface::forward(std::vector<bm::FrameInfo> &frame_infos)
{
    if (w_) w_->mark("forward");
    for (int b = 0; b < frame_infos.size(); ++b)
    {
        auto &frame_info = frame_infos[b];
        frame_info.output_tensors.resize(impl_->output_num);
        for (int i = 0; i < impl_->output_num; ++i)
        {
            frame_info.output_tensors[i] = *impl_->net->outputTensor(i)->bm_tensor();
            impl_->tensor_pool->alloc(frame_info.output_tensors[i]);
        }
#if 0
        bm_handle_t handle = impl_->ctx->handle();
        auto &input_tensor = frame_info.input_tensors[0];
        auto bin = bm::read_binary("input.bin");
        if (bmrt_tensor_bytesize(&input_tensor) % bin->size())
        {
            LOG(ERROR) << "invalid input.bin";
        }
        std::vector<uint8_t> tensor_bin(bin->size());
        call(
            bm_memcpy_d2s_partial_offset,
            handle, tensor_bin.data(), input_tensor.device_mem,
            bin->size(), 0);
        const size_t num_per_row = 40;
        auto tensor_data = reinterpret_cast<const float *>(tensor_bin.data());
        auto file_data = bin->ptr<float>();
        std::stringstream ss;
        ss << std::endl;
        ss << std::setfill('0') << std::fixed << std::setprecision(2);
        for (int i = 0; i < bin->size() / 4; ++i)
        {
            ss << tensor_data[i] - file_data[i];
            if (i && i % num_per_row == 0)
            {
                ss << std::endl;
            } else {
                ss << " ";
            }
        }
        LOG(INFO) << ss.str();
        call(
            bm_memcpy_s2d_partial_offset,
            handle, input_tensor.device_mem, bin->ptr<uint8_t>(),
            bin->size(), 0);
#endif
        call(
            impl_->net->forward_user_mem,
            frame_info.input_tensors.data(),
            frame_info.input_tensors.size(),
            frame_info.output_tensors.data(),
            frame_info.output_tensors.size());
    }
    if (w_) w_->mark("forward");
}

int Retinaface::postprocess(std::vector<bm::FrameInfo> &frame_infos)
{
    if (w_) w_->mark("postprocess");
    for (auto &frame_info : frame_infos)
    {
        // extract face detection
        this->extract_facebox_cpu(frame_info);

        if (m_pfnDetectFinish != nullptr) {
            m_pfnDetectFinish(frame_info);
        }

        for(int j = 0; j < frame_info.frames.size(); ++j) {

            auto& reff = frame_info.frames[j];
            if (reff.avpkt) {
                av_packet_unref(reff.avpkt);
                av_packet_free(&reff.avpkt);
            }

            if (reff.avframe) {
                av_frame_unref(reff.avframe);
                av_frame_free(&reff.avframe);
            }

            if (impl_->keep_original)
                call(bm_image_destroy, reff.original);
        }

        // Free Tensors
        for(auto& tensor : frame_info.input_tensors) {
            impl_->input_pool->free(tensor.device_mem);
        }

        for(auto &tensor : frame_info.output_tensors) {
            impl_->tensor_pool->free(tensor);
        }

    }
    if (w_) w_->mark("postprocess");
}

bm::BMNNTensorPtr Retinaface::get_output_tensor(
    const std::string &name,
    bm::FrameInfo &frame_info)
{
    int output_tensor_num = frame_info.output_tensors.size();
    int idx = impl_->net->outputName2Index(name);
    if (idx < 0 && idx + 1> output_tensor_num) {
        LOG(ERROR) << "invalid index " << idx;
        throw std::runtime_error("invalid index");
    }
    auto &tensor = frame_info.output_tensors[idx];
    if (tensor.dtype != BM_FLOAT32)
    {
        LOG(ERROR) << "Output tensor not of FLOAT32\n"
                   << "Please compile .bmodel with -output_as_fp32 option.";
        throw std::runtime_error("invalid model");
    }
    const float scale = 1.; // Dummy
    return std::make_shared<bm::BMNNTensor>(
        impl_->ctx->handle(), name, scale, &tensor);
}

static inline bool compareBBox(
    const bm::NetOutputObject &a, const bm::NetOutputObject &b) {
    return a.score > b.score;
}

static std::vector<bm::NetOutputObject> nms(
    std::vector<bm::NetOutputObject> &bboxes,
    float nms_threshold)
{
    std::vector<bm::NetOutputObject> nmsProposals;
    if (bboxes.empty()) {
        return nmsProposals;
    }
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
        float    area1          = (select_bbox.x2 - select_bbox.x1) *
                                  (select_bbox.y2 - select_bbox.y1);
        ++select_idx;
        for (int i = select_idx; i < num_bbox; ++i) {
            if (mask_merged[i] == 1)
                continue;
            bm::NetOutputObject &bbox_i = bboxes[i];
            float     x      = std::max(select_bbox.x1, bbox_i.x1);
            float     y      = std::max(select_bbox.y1, bbox_i.y1);
            float     w      = std::min(select_bbox.x2, bbox_i.x2) - x;
            float     h      = std::min(select_bbox.y2, bbox_i.y2) - y;
            if (w <= 0 || h <= 0)
                continue;
            float area2 = (bbox_i.x2 - bbox_i.x1) * (bbox_i.y2 - bbox_i.y1);
            float area_intersect = w * h;
            // Union method
            if (area_intersect / (area1 + area2 - area_intersect) >
                nms_threshold)
                mask_merged[i] = 1;
        }
    }
    return nmsProposals;
}

std::vector<bm::NetOutputObject> Retinaface::parse_boxes(
    size_t input_width,
    size_t input_height,
    const float *cls_data,
    const float *land_data,
    const float *loc_data,
    const bm::FrameBaseInfo &frame)
{
    const size_t num_layer = 3;
    const size_t steps[] = {8, 16, 32};
    const size_t num_anchor = 2;
    const size_t anchor_sizes[][2] = {
        {16, 32},
        {64, 128},
        {256, 512}};
    const float variances[] = {0.1, 0.2};

    size_t index = 0, min_size;
    std::vector<bm::NetOutputObject> boxes;
    const float *loc, *land;
    float x, y, w, h, conf;
    float anchor_w, anchor_h, anchor_x, anchor_y;
    bm::NetOutputObject obj;
    for (int il = 0; il < num_layer; ++il)
    {
        size_t feature_width = (input_width + steps[il] - 1) / steps[il];
        size_t feature_height = (input_height + steps[il] - 1) / steps[il];
        for (int iy = 0; iy < feature_height; ++iy)
        {
            for (int ix = 0; ix < feature_width; ++ix)
            {
                for (int ia = 0; ia < num_anchor; ++ia)
                {
                    conf = cls_data[index * 2 + 1];
                    if (conf < impl_->conf_threshold)
                        goto cond;
                    min_size = anchor_sizes[il][ia];
                    anchor_x = (ix + 0.5) * steps[il] / input_width;
                    anchor_y = (iy + 0.5) * steps[il] / input_height;
                    anchor_w = min_size * 1. / input_width;
                    anchor_h = min_size * 1. / input_height;
                    obj.score = conf;
                    loc = loc_data + index * 4;
                    w = exp(loc[2] * variances[1]) * anchor_w;
                    h = exp(loc[3] * variances[1]) * anchor_h;
                    x = anchor_x + loc[0] * variances[0] * anchor_w;
                    y = anchor_y + loc[1] * variances[0] * anchor_h;
                    obj.x1 = x - w / 2;
                    obj.x2 = x + w / 2;
                    obj.y1 = y - h / 2;
                    obj.y2 = y + h / 2;
                    land = land_data + index * 10;
                    for (int i = 0; i < 5; ++i)
                    {
                        obj.landmark.x[i] = anchor_x +
                            land[i * 2] * variances[0] * anchor_w;
                        obj.landmark.y[i] = anchor_y +
                            land[i * 2 + 1] * variances[0] * anchor_h;
                        obj.landmark.score = 1.;
                    }
                    boxes.push_back(obj);
cond:
                    ++index;
                }
            }
        }
    }
    auto objs = nms(boxes, impl_->nms_threshold);
    for (auto &obj : objs)
    {
        obj.x1 = round((obj.x1 - frame.x_offset) / frame.x_scale * frame.width);
        obj.x2 = round((obj.x2 - frame.x_offset) / frame.x_scale * frame.width);
        obj.y1 = round((obj.y1 - frame.y_offset) / frame.y_scale * frame.height);
        obj.y2 = round((obj.y2 - frame.y_offset) / frame.y_scale * frame.height);
        for (int i = 0; i < 5; ++i)
        {
            auto &x = obj.landmark.x[i];
            auto &y = obj.landmark.y[i];
            x = (x - frame.x_offset) / frame.x_scale * frame.width;
            y = (y - frame.y_offset) / frame.y_scale * frame.height;
        }
    }
    return objs;
}

void Retinaface::extract_facebox_cpu(
    bm::FrameInfo &frame_info)
{
    auto land_tensor = this->get_output_tensor("land", frame_info);
    auto cls_tensor  = this->get_output_tensor("cls", frame_info);
    auto loc_tensor  = this->get_output_tensor("loc", frame_info);
    const float *land_data = land_tensor->get_cpu_data();
    const float *cls_data  = cls_tensor->get_cpu_data();
    const float *loc_data  = loc_tensor->get_cpu_data();
    size_t land_step = land_tensor->total() / land_tensor->get_num();
    size_t cls_step = cls_tensor->total() / cls_tensor->get_num();
    size_t loc_step = loc_tensor->total() / loc_tensor->get_num();
    const auto &input_shape = frame_info.input_tensors[0].shape;
    size_t input_width = input_shape.dims[3];
    size_t input_height = input_shape.dims[2];
    for (int i = 0; i < frame_info.frames.size(); ++i)
    {
        frame_info.out_datums.emplace_back(
            this->parse_boxes(
                input_width,
                input_height,
                cls_data + i * cls_step,
                land_data + i * land_step,
                loc_data + i * loc_step,
                frame_info.frames[i]));
    }
}

RetinafaceEval::RetinafaceEval(
    bm::BMNNContextPtr bmctx,
    size_t target_size,
    bool keep_original,
    float nms_threshold,
    float conf_threshold,
    std::string net_name)
    : Retinaface(bmctx, keep_original, nms_threshold, conf_threshold, net_name)
{
    impl_->target_size = target_size;
}

int RetinafaceEval::preprocess(
    std::vector<bm::FrameBaseInfo> &frames,
    std::vector<bm::FrameInfo> &frame_infos)
{
    bm_handle_t handle = impl_->ctx->handle();
    for (int i = 0; i < frames.size(); ++i)
    {
        auto &frame = frames[i];
        bm::FrameInfo frame_info;

        bm_image image;
        if (frame.filename.empty())
        {
            LOG(ERROR) << "frame filename empty";
            throw std::runtime_error("input error");
        }

        // Decode image
        frame.jpeg_data = bm::read_binary(frame.filename);
        void *data_ptr = frame.jpeg_data->ptr<uint8_t>();
        size_t num = 1, data_size = frame.jpeg_data->size();
#ifdef USE_EXIV2
        uint8_t *bin = frame.jpeg_data->ptr<uint8_t>();
        Exiv2::Image::AutoPtr exiv = Exiv2::ImageFactory::open(bin, data_size);
        exiv->readMetadata();
        size_t w = exiv->pixelWidth();
        size_t h = exiv->pixelHeight();
        frame.original_width = w;
        frame.original_height = h;
        if (w > vpp_limit || h > vpp_limit)
        {
            LOG(INFO) << "use OpenCV to process " << frame.filename;
            image = impl_->opencv_decode_and_resize(bin, data_size);
        } else {
            int strides[3];
            const bm_image_format_ext format = FORMAT_YUV420P;
            strides[0] = align_with(w, 32);
            strides[1] = strides[2] = strides[0] / 2;
            const bm_image_data_format_ext dtype = DATA_TYPE_EXT_1N_BYTE;
            call(
                bm_image_create, handle,
                align_with(h, 2),
                align_with(w, 2),
                format, dtype, &image, strides);
#endif
            call(
                bmcv_image_jpeg_dec,
                handle,
                &data_ptr, &data_size,
                num, &image);
#ifdef USE_EXIV2
        }
#endif
        frame.jpeg_data.reset();

        // Calculate input size
        frame.width = image.width;
        frame.height = image.height;
        float min_size = std::min<float>(frame.width, frame.height);
        float max_size = std::max<float>(frame.width, frame.height);
        auto max_input = std::min(impl_->width, impl_->height);
        float factor = impl_->target_size / min_size;
        if (factor * max_size > max_input)
        {
            factor = max_input / max_size;
        }
        size_t input_width = round(factor * frame.width);
        size_t input_height = round(factor * frame.height);

        // Alloc resized image & input image
        bm_image resized_image;
        const size_t align = 64;
        const bool alloc_mem = true;
        call(
            bm::BMImage::create_batch,
            handle, input_height, input_width,
            FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE,
            &resized_image, num,
            align, alloc_mem);
        bm_image input_image;
        const size_t input_align = 1;
        const bool contiguous = true;
        const int heap_mask = 1;
        call(
            bm::BMImage::create_batch,
            handle, input_height, input_width,
            FORMAT_BGR_PLANAR, impl_->img_type,
            &input_image, num,
            input_align, alloc_mem, contiguous, heap_mask);

        // Resize image
        call(bmcv_image_vpp_basic, handle, num, &image, &resized_image);
        if (impl_->keep_original)
        {
            frame.original = image;
        } else {
            call(bm_image_destroy, image);
        }

        // Make input tensor
        bm_tensor_t input_tensor = *impl_->net->inputTensor(0)->bm_tensor();
        call(
            bm_image_get_contiguous_device_mem, num,
            &input_image, &input_tensor.device_mem);
        input_tensor.shape.dims[0] = num;
        input_tensor.shape.dims[2] = input_height;
        input_tensor.shape.dims[3] = input_width;
        frame_info.input_tensors.push_back(input_tensor);

        // Call convert to
        // Scale input and sub mean
        call(
            bmcv_image_convert_to,
            handle, num, impl_->convert_attr, &resized_image, &input_image);

        // Save frame
        // Destroy resized image
        // Destroy input image with release input memory
        frame_info.frames.push_back(frame);
        frame_infos.push_back(std::move(frame_info));
        call(bm_image_dettach_contiguous_mem, num, &input_image);
        call(
            bm::BMImage::destroy_batch,
            &resized_image, num);
        call(bm_image_destroy, input_image);
    }
}

