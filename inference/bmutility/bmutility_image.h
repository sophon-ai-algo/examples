/*==========================================================================
 * Copyright 2016-2022 by Sophgo Technologies Inc. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
============================================================================*/
#ifndef BMUTILITY_IMAGE_H
#define BMUTILITY_IMAGE_H

#ifdef __cplusplus
extern "C" {
#include "libavutil/avutil.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
}
#endif

#include "opencv2/opencv.hpp"
#include "bmcv_api_ext.h"

namespace bm {
///////////////////////////////////////////////////////////////////////////
#define BM_MEM_DDR0 1
#define BM_MEM_DDR1 2
#define BM_MEM_DDR2 4

// BMCV_IMAGE
struct BMImage {
  static inline bm_status_t create_batch(bm_handle_t handle,
                                         int img_h,
                                         int img_w,
                                         bm_image_format_ext img_format,
                                         bm_image_data_format_ext data_type,
                                         bm_image *image,
                                         int batch_num,
                                         int align = 1, bool bPreAllocMem = true,
                                         bool bContinuious = false, int mask = 6) {

    // init images
    int data_size = 1;
    if (data_type == DATA_TYPE_EXT_FLOAT32) {
      data_size = 4;
    }

    int stride[3] = {0};
    int img_w_real = img_w * data_size;
    if (FORMAT_RGB_PLANAR == img_format ||
        FORMAT_RGB_PACKED == img_format ||
        FORMAT_BGR_PLANAR == img_format ||
        FORMAT_BGR_PACKED == img_format) {
      stride[0] = FFALIGN(img_w_real, align);
    } else if (FORMAT_YUV420P == img_format) {
      stride[0] = FFALIGN(img_w_real, align);
      stride[1] = stride[2] = FFALIGN(img_w_real >> 1, align);
    } else if (FORMAT_NV12 == img_format || FORMAT_NV21 == img_format) {
      stride[0] = FFALIGN(img_w_real, align);
      stride[1] = FFALIGN(img_w_real >> 1, align);
    } else {
      assert(0);
    }

    int ret = 0;
    for (int i = 0; i < batch_num; i++) {
      bm_image_create(handle, img_h, img_w, img_format, data_type, &image[i], stride);
      if (bPreAllocMem && !bContinuious) {
        ret = bm_image_alloc_dev_mem_heap_mask(image[i], mask);
        assert(BM_SUCCESS == ret);
      }
    }

    if (bPreAllocMem && bContinuious) {
      ret = bm_image_alloc_contiguous_mem_heap_mask(batch_num, image, mask);
      assert(BM_SUCCESS == ret);
    }

    return BM_SUCCESS;
  }

  static inline bm_status_t destroy_batch(bm_image *images, int batch_num, bool bContinuious = false) {
    if (bContinuious) {
      bm_image_free_contiguous_mem(batch_num, images);
    }

    // deinit bm image
    for (int i = 0; i < batch_num; i++) {
      if (BM_SUCCESS != bm_image_destroy(images[i])) {
        printf("bm_image_destroy failed!\n");
      }
    }

    return BM_SUCCESS;
  }

  static inline int map_bmformat_to_avformat(int bmformat) {
    int format;
    switch (bmformat) {
      case FORMAT_YUV420P: format = AV_PIX_FMT_YUV420P;
        break;
      case FORMAT_YUV422P: format = AV_PIX_FMT_YUV422P;
        break;
      case FORMAT_YUV444P: format = AV_PIX_FMT_YUV444P;
        break;
      case FORMAT_NV12: format = AV_PIX_FMT_NV12;
        break;
      case FORMAT_NV16: format = AV_PIX_FMT_NV16;
        break;
      case FORMAT_GRAY: format = AV_PIX_FMT_GRAY8;
        break;
      case FORMAT_RGBP_SEPARATE: format = AV_PIX_FMT_GBRP;
        break;
      default: printf("unsupported image format %d\n", bmformat);
        return -1;
    }
    return format;
  }

  static inline int map_avformat_to_bmformat(int avformat) {
    int format;
    switch (avformat) {
      case AV_PIX_FMT_YUV420P: format = FORMAT_YUV420P;
        break;
      case AV_PIX_FMT_YUV422P: format = FORMAT_YUV422P;
        break;
      case AV_PIX_FMT_YUV444P: format = FORMAT_YUV444P;
        break;
      case AV_PIX_FMT_NV12: format = FORMAT_NV12;
        break;
      case AV_PIX_FMT_NV16: format = FORMAT_NV16;
        break;
      case AV_PIX_FMT_GRAY8: format = FORMAT_GRAY;
        break;
      case AV_PIX_FMT_GBRP: format = FORMAT_RGBP_SEPARATE;
        break;
      default: printf("unsupported av_pix_format %d\n", avformat);
        assert(0);
        return -1;
    }

    return format;
  }

  static inline int convert_yuv420p_software(const AVFrame *src, AVFrame** p_dst)
  {
    AVFrame *dst = av_frame_alloc();
    dst->width = src->width;
    dst->height = src->height;
    dst->format = AV_PIX_FMT_YUV420P;

    av_frame_get_buffer(dst, 64);
    SwsContext *ctx = sws_getContext(src->width, src->height, (AVPixelFormat)src->format, dst->width, dst->height,
                                     (AVPixelFormat)dst->format,SWS_BICUBIC, 0, NULL, NULL);
    sws_scale(ctx, src->data,  src->linesize, 0, src->height, dst->data, dst->linesize);
    sws_freeContext(ctx);
    *p_dst = dst;
    return 0;
  }

  static inline bm_status_t avframe_to_bm_image(bm_handle_t &bm_handle, const AVFrame *ifp, bm_image &out) {

    int plane = 0;
    int data_five_denominator = -1;
    int data_six_denominator = -1;
    AVFrame *tmp_yuv420p=NULL;
    AVFrame *pIn = (AVFrame*)ifp;

    if (ifp->data[4] != NULL) {
      switch (ifp->format) {
        case AV_PIX_FMT_GRAY8:plane = 1;
          data_five_denominator = -1;
          data_six_denominator = -1;
          break;
        case AV_PIX_FMT_YUV420P:plane = 3;
          data_five_denominator = 4;
          data_six_denominator = 4;
          break;
        case AV_PIX_FMT_NV12:plane = 2;
          data_five_denominator = 2;
          data_six_denominator = -1;
          break;
        case AV_PIX_FMT_YUV422P:plane = 3;
          data_five_denominator = 2;
          data_six_denominator = 2;
          break;
        case AV_PIX_FMT_NV16:plane = 2;
          data_five_denominator = 2;
          data_six_denominator = -1;
          break;
        case AV_PIX_FMT_YUV444P:
        case AV_PIX_FMT_GBRP:plane = 3;
          data_five_denominator = 1;
          data_six_denominator = 1;
          break;
        default:
          printf("unsupported format, only gray,nv12,yuv420p,nv16,yuv422p horizontal,yuv444p,rgbp supported\n");
          assert(0);
          break;
      }

      if (pIn->channel_layout == 101) {/* COMPRESSED NV12 FORMAT */
        if ((0 == pIn->height) || (0 == pIn->width) || \
         (0 == pIn->linesize[4]) || (0 == pIn->linesize[5]) || (0 == pIn->linesize[6]) || (0 == pIn->linesize[7]) || \
         (0 == pIn->data[4]) || (0 == pIn->data[5]) || (0 == pIn->data[6]) || (0 == pIn->data[7])) {
          printf("bm_image_from_frame: get yuv failed!!");
          return BM_ERR_PARAM;
        }
        bm_image cmp_bmimg;
        bm_image_create(bm_handle,
                        pIn->height,
                        pIn->width,
                        FORMAT_COMPRESSED,
                        DATA_TYPE_EXT_1N_BYTE,
                        &cmp_bmimg);

        bm_device_mem_t input_addr[4];
        int size = pIn->height * pIn->linesize[4];
        input_addr[0] = bm_mem_from_device((unsigned long long) pIn->data[6], size);
        size = (pIn->height / 2) * pIn->linesize[5];
        input_addr[1] = bm_mem_from_device((unsigned long long) pIn->data[4], size);
        size = pIn->linesize[6];
        input_addr[2] = bm_mem_from_device((unsigned long long) pIn->data[7], size);
        size = pIn->linesize[7];
        input_addr[3] = bm_mem_from_device((unsigned long long) pIn->data[5], size);
        bm_image_attach(cmp_bmimg, input_addr);
        bm_image_create(bm_handle,
                        pIn->height,
                        pIn->width,
                        FORMAT_YUV420P,
                        DATA_TYPE_EXT_1N_BYTE,
                        &out);
        //bm_image_dev_mem_alloc(out);
        bm_image_alloc_dev_mem_heap_mask(out, 4);
        bmcv_rect_t crop_rect = {0, 0, pIn->width, pIn->height};
        bmcv_image_vpp_convert(bm_handle, 1, cmp_bmimg, &out, &crop_rect);
        bm_image_destroy(cmp_bmimg);
      } else {
        int stride[3];
        bm_image_format_ext bm_format;
        bm_device_mem_t input_addr[3] = {0};
        if (plane == 1) {
          if ((0 == pIn->height) || (0 == pIn->width) || (0 == pIn->linesize[4]) || (0 == pIn->data[4])) {
            return BM_ERR_PARAM;
          }
          stride[0] = pIn->linesize[4];
        } else if (plane == 2) {
          if ((0 == pIn->height) || (0 == pIn->width) || \
                (0 == pIn->linesize[4]) || (0 == pIn->linesize[5]) || \
                (0 == pIn->data[4]) || (0 == pIn->data[5])) {
            return BM_ERR_PARAM;
          }

          stride[0] = pIn->linesize[4];
          stride[1] = pIn->linesize[5];
        } else if (plane == 3) {
          if ((0 == pIn->height) || (0 == pIn->width) || \
                (0 == pIn->linesize[4]) || (0 == pIn->linesize[5]) || (0 == pIn->linesize[6]) || \
                (0 == pIn->data[4]) || (0 == pIn->data[5]) || (0 == pIn->data[6])) {
            return BM_ERR_PARAM;
          }

          stride[0] = pIn->linesize[4];
          stride[1] = pIn->linesize[5];
          stride[2] = pIn->linesize[6];
        }

        bm_format = (bm_image_format_ext) map_avformat_to_bmformat(pIn->format);
        bm_image_create(bm_handle,
                        pIn->height,
                        pIn->width,
                        bm_format,
                        DATA_TYPE_EXT_1N_BYTE,
                        &out,
                        stride);

        int size = pIn->height * stride[0];
        input_addr[0] = bm_mem_from_device((unsigned long long) pIn->data[4], size);
        if (data_five_denominator != -1) {
          size = pIn->height * stride[1] / data_five_denominator;
          input_addr[1] = bm_mem_from_device((unsigned long long) pIn->data[5], size);
        }
        if (data_six_denominator != -1) {
          size = pIn->height * stride[2] / data_six_denominator;
          input_addr[2] = bm_mem_from_device((unsigned long long) pIn->data[6], size);
        }
        bm_image_attach(out, input_addr);
      }
      return BM_SUCCESS;
    }

    // software
    bm_status_t ret;
    int strides[3];
    bm_device_mem_t input_addr[3] = {0};
    convert_yuv420p_software(ifp, &tmp_yuv420p);
    strides[0] = tmp_yuv420p->linesize[0];
    strides[1] = tmp_yuv420p->linesize[1];
    strides[2] = tmp_yuv420p->linesize[2];
    bm_image_create(bm_handle,
                    pIn->height,
                    pIn->width,
                    FORMAT_YUV420P,
                    DATA_TYPE_EXT_1N_BYTE,
                    &out,
                    strides);
    ret = bm_image_alloc_dev_mem_heap_mask(out, BM_MEM_DDR1 | BM_MEM_DDR2);
    assert(BM_SUCCESS == ret);
    ret = bm_image_copy_host_to_device(out, (void**)tmp_yuv420p->data);
    assert(BM_SUCCESS == ret);
    if (tmp_yuv420p) {
      av_frame_free(&tmp_yuv420p);
    }
    return ret;
  }

  static inline bm_status_t from_avframe(bm_handle_t bm_handle,
                                         const AVFrame *pAVFrame,
                                         bm_image &out, bool bToYUV420p = false) {
    bm_status_t ret;
    const AVFrame &in = *pAVFrame;
    if (in.format != AV_PIX_FMT_NV12) {
      bm_image in_bmimage;
      ret = avframe_to_bm_image(bm_handle, pAVFrame, in_bmimage);
      assert(BM_SUCCESS == ret);
      ret = bm_image_create(bm_handle,
                            in.height,
                            in.width,
                            FORMAT_YUV420P,
                            DATA_TYPE_EXT_1N_BYTE,
                            &out);
      assert(BM_SUCCESS == ret);
      ret = bm_image_alloc_dev_mem(out, BMCV_HEAP1_ID);
      assert(BM_SUCCESS == ret);
      int ret = bmcv_image_vpp_convert(bm_handle, 1, in_bmimage, &out);
      assert(BM_SUCCESS == ret);
      bm_image_destroy(in_bmimage);

      return BM_SUCCESS;
    }

    if (in.channel_layout == 101) { /* COMPRESSED NV12 FORMAT */
      /* sanity check */
      if ((0 == in.height) || (0 == in.width) || \
    (0 == in.linesize[4]) || (0 == in.linesize[5]) || (0 == in.linesize[6]) || (0 == in.linesize[7]) || \
    (0 == in.data[4]) || (0 == in.data[5]) || (0 == in.data[6]) || (0 == in.data[7])) {
        std::cout << "bm_image_from_frame: get yuv failed!!" << std::endl;
        return BM_ERR_PARAM;
      }
      bm_image cmp_bmimg;
      ret = bm_image_create(bm_handle,
                            in.height,
                            in.width,
                            FORMAT_COMPRESSED,
                            DATA_TYPE_EXT_1N_BYTE,
                            &cmp_bmimg);
      assert(BM_SUCCESS == ret);
      /* calculate physical address of avframe */
      bm_device_mem_t input_addr[4];
      int size = in.height * in.linesize[4];
      input_addr[0] = bm_mem_from_device((unsigned long long) in.data[6], size);
      size = (in.height / 2) * in.linesize[5];
      input_addr[1] = bm_mem_from_device((unsigned long long) in.data[4], size);
      size = in.linesize[6];
      input_addr[2] = bm_mem_from_device((unsigned long long) in.data[7], size);
      size = in.linesize[7];
      input_addr[3] = bm_mem_from_device((unsigned long long) in.data[5], size);
      ret = bm_image_attach(cmp_bmimg, input_addr);
      assert(BM_SUCCESS == ret);

      if (!bToYUV420p) {
        out = cmp_bmimg;
      } else {
        ret = bm_image_create(bm_handle,
                              in.height,
                              in.width,
                              FORMAT_YUV420P,
                              DATA_TYPE_EXT_1N_BYTE,
                              &out);
        assert(BM_SUCCESS == ret);
        ret = bm_image_alloc_dev_mem(out, BMCV_HEAP1_ID);
        assert(BM_SUCCESS == ret);
        int ret = bmcv_image_vpp_convert(bm_handle, 1, cmp_bmimg, &out);
        assert(BM_SUCCESS == ret);
        bm_image_destroy(cmp_bmimg);
      }

    } else { /* UNCOMPRESSED NV12 FORMAT */
      /* sanity check */
      if ((0 == in.height) || (0 == in.width) || \
    (0 == in.linesize[4]) || (0 == in.linesize[5]) || \
    (0 == in.data[4]) || (0 == in.data[5])) {
        std::cout << "bm_image_from_frame: get yuv failed!!" << std::endl;
        return BM_ERR_PARAM;
      }

      /* create bm_image with YUV-nv12 format */
      bm_image cmp_bmimg;
      int stride[2];
      stride[0] = in.linesize[4];
      stride[1] = in.linesize[5];
      bm_image_create(bm_handle,
                      in.height,
                      in.width,
                      FORMAT_NV12,
                      DATA_TYPE_EXT_1N_BYTE,
                      &cmp_bmimg,
                      stride);

      /* calculate physical address of yuv mat */
      bm_device_mem_t input_addr[2];
      int size = in.height * stride[0];
      input_addr[0] = bm_mem_from_device((unsigned long long) in.data[4], size);
      size = in.height * stride[1];
      input_addr[1] = bm_mem_from_device((unsigned long long) in.data[5], size);

      /* attach memory from mat to bm_image */
      bm_image_attach(cmp_bmimg, input_addr);

      if (!bToYUV420p) {
        out = cmp_bmimg;
      } else {
        ret = bm_image_create(bm_handle,
                              in.height,
                              in.width,
                              FORMAT_YUV420P,
                              DATA_TYPE_EXT_1N_BYTE,
                              &out);
        assert(BM_SUCCESS == ret);
        ret = bm_image_alloc_dev_mem(out, BMCV_HEAP1_ID);
        assert(BM_SUCCESS == ret);
        int ret = bmcv_image_vpp_convert(bm_handle, 1, cmp_bmimg, &out);
        assert(BM_SUCCESS == ret);
        bm_image_destroy(cmp_bmimg);
      }
    }

    return BM_SUCCESS;
  }

  static uint8_t *jpeg_enc(bm_handle_t handle, AVFrame *frame) {
    bm_image yuv_img;
    int ret = from_avframe(handle, frame, yuv_img, true);
    assert(BM_SUCCESS == ret);
    uint8_t *jpeg = nullptr;
    size_t out_size = 0;
    ret = bmcv_image_jpeg_enc(handle, 1, &yuv_img, (void **) &jpeg, &out_size);
    assert(BM_SUCCESS == ret);
    bm_image_destroy(yuv_img);
    return jpeg;
  }

  static unsigned int face_align(unsigned int n, unsigned align) {
    return (n + (align - 1)) & (~(align - 1));
  }

  static void BGRPlanarToPacked(unsigned char *inout, int N, int H, int W) {
    unsigned char *temp = new unsigned char[H * W * 3];
    for (int n = 0; n < N; n++) {
      unsigned char *start = inout + 3 * H * W * n;
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          temp[3 * (h * W + w)] = start[(h * W + w)];
          temp[3 * (h * W + w) + 1] = start[(h * W + w) + H * W];
          temp[3 * (h * W + w) + 2] = start[(h * W + w) + 2 * H * W];
        }
      }
      memcpy(start, temp, H * W * 3);
    }
    delete[] temp;
  }

  static void convert_4N_2_1N(unsigned char *inout, int N, int C, int H, int W) {
    unsigned char *temp_buf = new unsigned char[4 * C * H * W];
    for (int i = 0; i < face_align(N, 4) / 4; i++) {
      memcpy(temp_buf, inout + 4 * C * H * W * i, 4 * C * H * W);
      for (int loop = 0; loop < C * H * W; loop++) {
        inout[i * 4 * C * H * W + loop] = temp_buf[4 * loop];
        inout[i * 4 * C * H * W + 1 * C * H * W + loop] = temp_buf[4 * loop + 1];
        inout[i * 4 * C * H * W + 2 * C * H * W + loop] = temp_buf[4 * loop + 2];
        inout[i * 4 * C * H * W + 3 * C * H * W + loop] = temp_buf[4 * loop + 3];
      }
    }
    delete[] temp_buf;
  }

  static void interleave_fp32(float *inout, int N, int H, int W) {
    float *temp = new float[H * W * 3];
    for (int n = 0; n < N; n++) {
      float *start = inout + 3 * H * W * n;
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          temp[3 * (h * W + w)] = start[(h * W + w)];
          temp[3 * (h * W + w) + 1] = start[(h * W + w) + H * W];
          temp[3 * (h * W + w) + 2] = start[(h * W + w) + 2 * H * W];
        }
      }
      memcpy(start, temp, H * W * 3 * sizeof(float));
    }
    delete[] temp;
  }

  static void dump_dev_memory(bm_handle_t bm_handle,
                              bm_device_mem_t dev_mem,
                              char *fn,
                              int n,
                              int h,
                              int w,
                              int b_fp32,
                              int b_4N) {
    cv::Mat img;
    int c = 3;
    int tensor_size = face_align(n, 4) * c * h * w;
    int c_size = c * h * w;
    int element_size = 4;
    unsigned char *s = new unsigned char[tensor_size * element_size];
    if (bm_mem_get_type(dev_mem) == BM_MEM_TYPE_DEVICE) {
      bm_memcpy_d2s(bm_handle, (void *) s, dev_mem);
    } else {
      int element_size = b_fp32 ? 4 : 1;
      memcpy(s,
             bm_mem_get_system_addr(dev_mem),
             n * c * h * w * element_size);
    }
    if (b_4N) {
      convert_4N_2_1N(s, n, c, h, w);
    }
    if (b_fp32) {
      interleave_fp32((float *) s, n, h, w);
    } else {
      BGRPlanarToPacked(s, n, h, w);
    }
    for (int i = 0; i < n; i++) {
      char fname[256];
      sprintf(fname, "%s_%d.png", fn, i);
      if (b_fp32) {
        img.create(h, w, CV_32FC3);
        memcpy(img.data, (float *) s + c_size * i, c_size * 4);
        cv::Mat img2;
        img.convertTo(img2, CV_8UC3);
        cv::imwrite(fn, img2);
      } else {
        cv::Mat img(h, w, CV_8UC3);
        memcpy(img.data, s + c_size * i, c_size);
        cv::imwrite(fname, img);
      }
    }
    delete[]s;
  }
};

static void *get_bm_image_addr(const bm_image &image) {
  bm_device_mem_t mem[3];
  auto ret = bm_image_get_device_mem(image, mem);
  assert(BM_SUCCESS == ret);
  auto addr = reinterpret_cast<void *>(bm_mem_get_device_addr(mem[0]));
  return addr;
}
}

#endif //!BMUTILITY_IMAGE_H
