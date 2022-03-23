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
//
// Created by yuan on 3/10/21.
//

#ifndef INFERENCE_FRAMEWORK_BMUTILITY_TYPES_H
#define INFERENCE_FRAMEWORK_BMUTILITY_TYPES_H

#include <vector>
#include <functional>

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <assert.h>

#ifdef __linux__
#include <netinet/in.h>
#endif

#ifdef WIN32
#include <WinSock2.h>
#include <Windows.h>
#endif

#include <memory>

#if USE_FFMPEG
#include "stream_decode.h"
#endif

#include <opencv2/opencv.hpp>

#include "bmcv_api_ext.h"
#include "bmruntime_interface.h"

namespace bm {

    class NoCopyable {
    protected:
        NoCopyable() = default;

        ~NoCopyable() = default;

        NoCopyable(const NoCopyable &) = delete;

        NoCopyable &operator=(const NoCopyable &rhs) = delete;
    };
    class ByteBuffer : public NoCopyable {
        char *m_bytes;
        int m_back_offset;
        int m_front_offset;
        size_t m_size;

        bool check_buffer(int len) {
            if (m_back_offset + len > m_size) {
                char *pNew = nullptr;
                pNew = (char*)realloc(m_bytes, m_size + 1024);
                assert(pNew != nullptr);
                if (pNew == nullptr) {
                    return false;
                }else{
                    m_bytes = pNew;
                    m_size += 1024;
                }
            }

            return true;
        }

        bool check_buffer2(int len) {
            if (m_back_offset - len < 0) {
               return false;
            }

            return true;
        }

        int push_internal(int8_t *p, int len) {
            if (!check_buffer(len)) return -1;
            memcpy((uint8_t*)m_bytes + m_back_offset, p, len);
            m_back_offset += len;
            return 0;
        }

        int pop_internal(int8_t *p, int len) {
            if (check_buffer2(len) !=0) return -1;
            memcpy(p, &m_bytes[m_back_offset], len);
            m_back_offset-=len;
            return 0;
        }

        int pop_front_internal(int8_t *p, int len) {
            if (m_front_offset + len > m_back_offset) return -1;
            memcpy(p, &m_bytes[m_front_offset], len);
            m_front_offset+=len;
            return 0;
        }

        uint64_t bm_htonll(uint64_t val)
        {
            return (((uint64_t) htonl(val)) << 32) + htonl(val >> 32);
        }

        uint64_t bm_ntohll(uint64_t val)
        {
            return (((uint64_t) ntohl(val)) << 32) + ntohl(val >> 32);
        }

        std::function<void(void*)> m_freeFunc;

    public:
        ByteBuffer(size_t size = 1024):m_size(size) {
            m_bytes = new char[size];
            assert(m_bytes != nullptr);
            m_front_offset = 0;
            m_back_offset = 0;

            m_freeFunc = [this](void*p) {
                delete [] m_bytes;
            };

        }

        ByteBuffer(char *buf, int size, std::function<void(void*)> free_func = nullptr){
            m_bytes = buf;
            m_front_offset = 0;
            m_back_offset = size;
            m_size = size;
            m_freeFunc = free_func;
        }

        ~ByteBuffer() {
            if (m_freeFunc) m_freeFunc(m_bytes);
        }



        int push_back(int8_t b){
            return push_internal(&b, sizeof(b));
        }

        int push_back(uint8_t b) {
            return push_internal((int8_t *)&b, sizeof(b));
        }

        int push_back(int16_t b) {
            b = htons(b);
            int8_t  *p = (int8_t*)&b;
            return push_internal(p, sizeof(b));
        }

        int push_back(uint16_t b)
        {
            b = htons(b);
            int8_t  *p = (int8_t*)&b;
            return push_internal(p, sizeof(b));
        }

        int push_back(int32_t b)
        {
            b = htonl(b);
            int8_t  *p = (int8_t*)&b;
            return push_internal(p, sizeof(b));
        }

        int push_back(uint32_t b)
        {
            b = htonl(b);
            int8_t  *p = (int8_t*)&b;
            return push_internal(p, sizeof(b));
        }

        int push_back(int64_t b)
        {
            b = bm_htonll(b);
            int8_t  *p = (int8_t*)&b;
            return push_internal(p, sizeof(b));
        }

        int push_back(uint64_t b)
        {
            b = bm_htonll(b);
            int8_t  *p = (int8_t*)&b;
            return push_internal(p, sizeof(b));
        }

        int push_back(float f)
        {
            int8_t  *p = (int8_t*)&f;
            return push_internal(p, sizeof(f));
        }

        int push_back(double d)
        {
            int8_t  *p = (int8_t*)&d;
            return push_internal(p, sizeof(d));
        }

        int pop(int8_t &val) {
            return pop_internal(&val, sizeof(val));
        }

        int pop(int16_t &val) {
            int16_t t;
            if (pop_internal((int8_t *)&t, sizeof(val)) != 0) {
                return -1;
            }

            val = ntohs(t);
            return 0;
        }

        int pop(int32_t &val) {
            int32_t t;
            if (pop_internal((int8_t *)&t, sizeof(val)) != 0) {
                return -1;
            }

            val = ntohl(t);
            return 0;
        }
        int pop(int64_t &val) {
            int64_t t;
            if (pop_internal((int8_t *)&t, sizeof(val)) != 0) {
                return -1;
            }

            val = bm_ntohll(t);
            return 0;
        }

        int pop(uint8_t &val) {
            return pop_internal((int8_t*)&val, sizeof(val));
        }

        int pop(uint16_t &val) {
            uint16_t t;
            if (pop_internal((int8_t *)&t, sizeof(val)) != 0) {
                return -1;
            }

            val = ntohs(t);
            return 0;
        }

        int pop(uint32_t &val) {
            uint32_t t;
            if (pop_internal((int8_t *)&t, sizeof(val)) != 0) {
                return -1;
            }

            val = ntohl(t);
            return 0;
        }

        int pop(uint64_t &val) {
            uint64_t t;
            if (pop_internal((int8_t *)&t, sizeof(val)) != 0) {
                return -1;
            }

            val = bm_ntohll(t);
            return 0;
        }

        int pop(float &val) {
            return pop_internal((int8_t *)&val, sizeof(val));
        }

        int pop(double &val) {
            return pop_internal((int8_t *)&val, sizeof(val));
        }

        int pop_front(int8_t &val) {
            return pop_front_internal(&val, sizeof(val));
        }

        int pop_front(int16_t &val) {
            int16_t t;
            if (pop_front_internal((int8_t *)&t, sizeof(val)) != 0) {
                return -1;
            }

            val = ntohs(t);
            return 0;
        }

        int pop_front(int32_t &val) {
            int32_t t;
            if (pop_front_internal((int8_t *)&t, sizeof(val)) != 0) {
                return -1;
            }

            val = ntohl(t);
            return 0;
        }
        int pop_front(int64_t &val) {
            int64_t t;
            if (pop_front_internal((int8_t *)&t, sizeof(val)) != 0) {
                return -1;
            }

            val = bm_ntohll(t);
            return 0;
        }

        int pop_front(uint8_t &val) {
            return pop_front_internal((int8_t*)&val, sizeof(val));
        }

        int pop_front(uint16_t &val) {
            uint16_t t;
            if (pop_front_internal((int8_t *)&t, sizeof(val)) != 0) {
                return -1;
            }

            val = ntohs(t);
            return 0;
        }

        int pop_front(uint32_t &val) {
            uint32_t t;
            if (pop_front_internal((int8_t *)&t, sizeof(val)) != 0) {
                return -1;
            }

            val = ntohl(t);
            return 0;
        }

        int pop_front(uint64_t &val) {
            uint64_t t;
            if (pop_front_internal((int8_t *)&t, sizeof(val)) != 0) {
                return -1;
            }

            val = bm_ntohll(t);
            return 0;
        }

        int pop_front(float &val) {
            return pop_front_internal((int8_t *)&val, sizeof(val));
        }

        int pop_front(double &val) {
            return pop_front_internal((int8_t *)&val, sizeof(val));
        }

        char *data() {
            return m_bytes;
        }

        int size() {
            return m_back_offset - m_front_offset;
        }

    };

    struct Data : public NoCopyable {
        size_t dsize;
        uint8_t *data;

        Data() : dsize(0), data(nullptr){
        }

        Data(uint8_t* p, size_t sz, bool copy=false) : dsize(sz) {
            if (copy) {
                data = new uint8_t[dsize];
                memcpy(data, p,  dsize);
            }else{
                data = p;
            }
        }

        virtual ~Data() {
            if (data)  delete[] data;
        }

        int size() {
            return dsize;
        }

        template <typename T>
        T* ptr() {
            return reinterpret_cast<T*>(data);
        }

    };

    using DataPtr = std::shared_ptr<Data>;
    struct LandmarkObject {
        float x[5];
        float y[5];
        float score;
    };
    using LandmarkObjects =std::vector<LandmarkObject>;

    struct NetOutputObject {
        float x1, y1, x2, y2;
        float score;
        int class_id;
        int track_id;
        LandmarkObject landmark;

        NetOutputObject():x1(0.0), y1(0.0), x2(0.0) ,y2(0.0), class_id(0), track_id(0) {}
        NetOutputObject(float x1, float y1, float width, float height) {
            this->x1 = x1;
            this->y1 = y1;
            this->x2 = x1 + width;
            this->y2 = y1 + height;

            this->track_id = 0;
            this->class_id = 0;
        }

        int width() {
            return x2-x1;
        }

        int height() {
            return y2-y1;
        }

        void to_bmcv_rect(bmcv_rect_t *rect){
            rect->start_y =y1;
            rect->start_x = x1;
            rect->crop_w = width();
            rect->crop_h = height();
        }
    };
    using NetOutputObjects =std::vector<NetOutputObject>;
    using ObjectFeature = std::vector<float>;

    struct PoseKeyPoints{
        enum EModelType {
            BODY_25 = 0,
            COCO_18 = 1
        };
        std::vector<float> keypoints;
        std::vector<int> shape;
        int width, height;
        EModelType modeltype;
    };

    struct SafetyhatObject {
        float x1, y1, x2, y2;
        float score;
        int class_id;
        int index;
        float confidence;

        int width() {
            return x2-x1;
        }

        int height() {
            return y2-y1;
        }

        void to_bmcv_rect(bmcv_rect_t *rect){
            rect->start_y =y1;
            rect->start_x = x1;
            rect->crop_w = width();
            rect->crop_h = height();
        }
       
    };
    using SafetyhatObjects = std::vector<SafetyhatObject>;

    struct NetOutputDatum {
        enum NetClassType {
            Box=0,
            Pose,
            FaceRecognition,
            SaftyhatRecogniton,
        };
        NetClassType type;

        NetOutputObjects obj_rects;
        PoseKeyPoints pose_keypoints;
        LandmarkObjects landmark_objects;
        std::vector<ObjectFeature> face_features;
        NetOutputObjects track_rects;
        SafetyhatObjects safetyhat_objects;

        NetOutputDatum(PoseKeyPoints& o) {
            pose_keypoints = o;
            type = Pose;
        }

        NetOutputDatum(const NetOutputObjects &o) {
            type = Box;
            obj_rects = o;
        }

        NetOutputDatum() {
            type = Box;
        }

        NetOutputDatum(SafetyhatObjects& o) {
            safetyhat_objects = o;
            type = SaftyhatRecogniton;
        }

        std::shared_ptr<ByteBuffer> toByteBuffer() {
            std::shared_ptr<ByteBuffer> buf = std::make_shared<ByteBuffer>();
            buf->push_back((int32_t)type);
            if (Box == type) {
                buf->push_back((uint32_t)obj_rects.size());
                for(auto o: obj_rects) {
                    buf->push_back(o.x1);
                    buf->push_back(o.y1);
                    buf->push_back(o.x2);
                    buf->push_back(o.y2);
                    buf->push_back(o.score);
                    buf->push_back(o.class_id);
                }
            } else if(Pose == type) {
                buf->push_back(pose_keypoints.height);
                buf->push_back(pose_keypoints.width);
                buf->push_back((uint32_t)pose_keypoints.shape.size());
                for(int i = 0;i < pose_keypoints.shape.size(); ++i) {
                    buf->push_back(pose_keypoints.shape[i]);
                }

                buf->push_back((uint32_t)pose_keypoints.keypoints.size());
                for(int i = 0;i < pose_keypoints.keypoints.size(); ++i) {
                    buf->push_back(pose_keypoints.keypoints[i]);
                }
            } else if(SaftyhatRecogniton == type) {
                buf->push_back((uint32_t)safetyhat_objects.size());
                for(auto o: safetyhat_objects) {
                    buf->push_back(o.x1);
                    buf->push_back(o.y1);
                    buf->push_back(o.x2);
                    buf->push_back(o.y2);
                    buf->push_back(o.score);
                    buf->push_back(o.class_id);
                    buf->push_back(o.index);
                    buf->push_back(o.confidence);
                }
            } else {
                printf("Unsupport type=%d\n", type);
                assert(0);
            }

            return buf;
        }

        void fromByteBuffer(ByteBuffer *buf) {
            int32_t itype;
            buf->pop_front(itype);
            this->type = (NetClassType)itype;
            if (Box == type) {
                uint32_t size = 0;
                buf->pop_front(size);
                for(int i = 0; i < size; ++i) {
                    NetOutputObject o;
                    buf->pop_front(o.x1);
                    buf->pop_front(o.y1);
                    buf->pop_front(o.x2);
                    buf->pop_front(o.y2);
                    buf->pop_front(o.score);
                    buf->pop_front(o.class_id);
                    obj_rects.push_back(o);
                }
            } else if(Pose == type) {
                buf->pop_front(pose_keypoints.height);
                buf->pop_front(pose_keypoints.width);
                uint32_t size=0;
                buf->pop_front(size);
                for(int i = 0;i < size; ++i) {
                    int dim = 0;
                    buf->pop_front(dim);
                    pose_keypoints.shape.push_back(dim);
                }

                buf->pop_front(size);
                pose_keypoints.keypoints.resize(size);
                for(int i = 0;i < size; ++i) {
                    buf->pop_front(pose_keypoints.keypoints[i]);
                }
            }else{
                printf("Unsupport type=%d\n", type);
                assert(0);
            }

        }
    };

#if USE_FFMPEG
    struct FrameBaseInfo {
        int chan_id;
        uint64_t seq;
        AVPacket *avpkt;
        AVFrame *avframe;
        std::string filename;
        bm::DataPtr jpeg_data;
        cv::Mat cvimg;
        float x_offset = 0, y_offset = 0;
        float x_scale = 1, y_scale = 1;
        bm_image original, resized;
        int width, height, original_width, original_height;
        bool skip;

        FrameBaseInfo() : chan_id(0), seq(0), avpkt(nullptr), avframe(nullptr), jpeg_data(nullptr), skip(false) {
            memset(&resized, 0, sizeof(bm_image));
            memset(&original, 0, sizeof(bm_image));
        }

        void destroy () {
            if (avpkt != nullptr) {
                av_packet_unref(avpkt);
                av_packet_free(&avpkt);
                avpkt = nullptr;
            }
            if (avframe != nullptr) {
                av_frame_unref(avframe);
                av_frame_free(&avframe);
                avframe = nullptr;
            }
            bm_image_destroy(original);
            bm_image_destroy(resized);
            cvimg.release();
            jpeg_data.reset();
        }
        static void FrameBaseInfoDestroyFn(bm::FrameBaseInfo& obj) {
            obj.destroy();
        }
    };

    struct FrameInfo {
        //AVFrame based
        std::vector<FrameBaseInfo> frames;
        std::vector<bm_tensor_t> input_tensors;
        std::vector<bm_tensor_t> output_tensors;
        std::vector<bm::NetOutputDatum> out_datums;
        bm_handle_t handle;

        void destroy() {
            for (auto& f : frames) {
                f.destroy();
            }
            for (auto& f : frames) {
                f.destroy();
            }
            // Free Tensors
            for(auto& tensor : input_tensors) {
                bm_free_device(handle, tensor.device_mem);
            }

            for(auto& tensor: output_tensors) {
                bm_free_device(handle, tensor.device_mem);
            }
        }
        static void FrameInfoDestroyFn(bm::FrameInfo& obj) {
            obj.destroy();
        }
    };

#endif

}

#endif //INFERENCE_FRAMEWORK_BMUTILITY_TYPES_H
