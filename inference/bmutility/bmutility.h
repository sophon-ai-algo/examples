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
// Created by yuan on 1/22/21.
//

#ifndef YOLOV5_DEMO_BMNN_UTILS_H
#define YOLOV5_DEMO_BMNN_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <assert.h>

#include <iostream>
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <thread>
#include <fstream>
#include <numeric>

#ifdef __linux__
#include <sys/time.h>
#endif //

#include "bmruntime_interface.h"
#include "bmutility_types.h"
#include "bmutility_timer.h"
#include "bmutility_image.h"
#include "bmutility_string.h"

static std::ostream &operator<<(std::ostream &out, const bm_shape_t &shape)
{
    if (shape.num_dims <= 0)
    {
        return out << "[empty shape]";
    }
    std::stringstream ss;
    for (int i = 0; i < shape.num_dims; ++i)
        ss << shape.dims[i] << "x";
    auto s = ss.str();
    s = s.substr(0, s.size() - 1);
    return out << s;
}

static std::ostream &operator<<(std::ostream &out, const bm_net_info_t &net_info)
{
    std::stringstream ss;
    ss << "============" << net_info.name << "============" << std::endl;
    if (net_info.is_dynamic)
    {
        ss << "dynamic" << std::endl;
    }
    for (int ii = 0; ii < net_info.input_num; ++ii)
    {
        ss << "input  " << ii;
        for (int is = 0; is < net_info.stage_num; ++is)
        {
            const bm_stage_info_s &stage = net_info.stages[is];
            const bm_shape_t &shape = stage.input_shapes[ii];
            ss << " " << shape;
        }
        if (net_info.input_dtypes[ii] != BM_FLOAT32)
            ss << " " << net_info.input_scales[ii];
        ss << std::endl;
    }
    for (int ii = 0; ii < net_info.output_num; ++ii)
    {
        ss << "output " << ii;
        for (int is = 0; is < net_info.stage_num; ++is)
        {
            const bm_stage_info_s &stage = net_info.stages[is];
            const bm_shape_t &shape = stage.output_shapes[ii];
            ss << " " << shape;
        }
        if (net_info.input_dtypes[ii] != BM_FLOAT32)
            ss << " " << net_info.output_scales[ii];
        ss << std::endl;
    }
    return out << ss.str();
}

namespace bm {

    static int bm_tensor_reshape_NCHW(bm_handle_t handle, bm_tensor_t *tensor, int n, int c, int h, int w) {
        tensor->shape.num_dims=4;
        tensor->shape.dims[0] = n;
        tensor->shape.dims[1] = c;
        tensor->shape.dims[2] = h;
        tensor->shape.dims[3] = w;

        int size = bmrt_tensor_bytesize(tensor);
        int ret = bm_malloc_device_byte_heap_mask(handle, &tensor->device_mem, 7, size);
        assert(BM_SUCCESS == ret);
        return ret;
    }

    static int bm_tensor_reshape(bm_handle_t handle, bm_tensor_t *tensor, const bm_shape_t* shape) {
        tensor->shape = *shape;
        int size = bmrt_tensor_bytesize(tensor);
        int ret = bm_malloc_device_byte_heap_mask(handle, &tensor->device_mem, 7, size);
        assert(BM_SUCCESS == ret);
        return ret;
    }

    static bm::DataPtr read_binary(const std::string &fn)
    {
        std::ifstream in(fn, std::ifstream::binary | std::ifstream::ate);
        int size = in.tellg();
        if (size < 0)
        {
            std::cerr << "failed to read " << fn << std::endl;
            throw std::runtime_error("io failed");
        }
        auto data = std::make_shared<bm::Data>(new uint8_t[size], size);
        in.seekg(0);
        in.read(data->ptr<char>(), size);
        return data;
    }

    class BMNNTensor {
        /**
         *  members from bm_tensor {
         *  bm_data_type_t dtype;
            bm_shape_t shape;
            bm_device_mem_t device_mem;
            bm_store_mode_t st_mode;
            }
         */
        bm_handle_t m_handle;

        std::string m_name;
        float *m_cpu_data;
        float m_scale;
        // user must free tensor's device memory by himself.
        bm_tensor_t *m_tensor;
        int m_tensor_elem_count {0};
        int m_tensor_size {0};

        std::vector<int> m_shape;
        int m_dtype {-1};

        bool update_shape() {
            bool changed = false;
            if (m_tensor->shape.num_dims == m_shape.size()) {
                for (int i = 0; i < m_tensor->shape.num_dims; ++i) {
                    if (m_shape[i] != m_tensor->shape.dims[i]) {
                        changed = true;
                        break;
                    }
                }
            }else{
                changed = true;
            }

            if (changed) {
                m_shape.assign(m_tensor->shape.dims,
                               m_tensor->shape.dims + m_tensor->shape.num_dims);
                m_tensor_elem_count = std::accumulate(m_shape.begin(), m_shape.end(), 1,
                                                 std::multiplies<int>());
            }
        }

        void update_dtype() {
            if (m_dtype != m_tensor->dtype) {
                switch (m_tensor->dtype) {
                    case BM_FLOAT32:
                    case BM_UINT32:
                    case BM_INT32:
                        m_tensor_size = m_tensor_elem_count << 2;
                        break;
                    case BM_INT16:
                    case BM_UINT16:
                    case BM_FLOAT16:
                        m_tensor_size = m_tensor_elem_count << 1;
                        break;
                    default:
                        m_tensor_size = m_tensor_elem_count;
                }
            }
        }

    public:
        BMNNTensor(bm_handle_t handle, const std::string& name, float scale,
                   bm_tensor_t *tensor) : m_handle(handle), m_name(name),
                                          m_cpu_data(nullptr), m_scale(scale),
                                          m_tensor(tensor)
                                          {

                                          }

        virtual ~BMNNTensor() {
            if (m_cpu_data != NULL) {
                delete [] m_cpu_data;
                m_cpu_data = NULL;
            }
        }

        //int set_device_mem(bm_device_mem_t *mem) {
        //    this->m_tensor->device_mem = *mem;
        //    return 0;
        //}

        const bm_device_mem_t get_device_mem() {
            return this->m_tensor->device_mem;
        }

        int get_count() {
            //m_tensor maybe changed, so change
            update_shape();
            return m_tensor_elem_count;
        }

        int get_size() {
            update_shape();
            update_dtype();
            return m_tensor_size;
        }

        float *get_cpu_data() {
            if (m_cpu_data == NULL) {
                bm_status_t ret;
                float *pFP32 = nullptr;
                int count = bmrt_shape_count(&m_tensor->shape);
                if (m_tensor->dtype == BM_FLOAT32) {
                    pFP32 = new float[count];
                    assert(pFP32 != nullptr);
                    ret = bm_memcpy_d2s_partial(m_handle, pFP32, m_tensor->device_mem, count * sizeof(float));
                    assert(BM_SUCCESS ==ret);
                }else if (BM_INT8 == m_tensor->dtype) {
                    int tensor_size = bmrt_tensor_bytesize(m_tensor);
                    int8_t *pU8 = new int8_t[tensor_size];
                    assert(pU8 != nullptr);
                    pFP32 = new float[count];
                    assert(pFP32 != nullptr);
                    ret = bm_memcpy_d2s_partial(m_handle, pU8, m_tensor->device_mem, tensor_size);
                    assert(BM_SUCCESS ==ret);
                    for(int i = 0;i < count; ++ i) {
                        pFP32[i] = pU8[i] * m_scale;
                    }
                    delete [] pU8;
                }else{
                    std::cout << "NOT support dtype=" << m_tensor->dtype << std::endl;
                }

                m_cpu_data = pFP32;
            }

            return m_cpu_data;
        }

        const bm_shape_t *get_shape() {
            return &m_tensor->shape;
        }

        bm_data_type_t get_dtype() {
            return m_tensor->dtype;
        }

        float get_scale() {
            return m_scale;
        }

        size_t total() {
            return bmrt_shape_count(&m_tensor->shape);
        }

        int get_num() {
            return m_tensor->shape.dims[0];
        }

        const std::string& get_name() {
            return m_name;
        }

        bm_tensor_t *bm_tensor() {
            return m_tensor;
        }

        //int reshape_nchw(int n, int c, int h, int w) {
        //    return bm_tensor_reshape_NCHW(m_handle, m_tensor, n, c, h, w);
        //}

        //int reshape_nhwc(int n, int c, int h, int w) {
        //    return bm_tensor_reshape_NHWC(m_handle, m_tensor, n, c, h, w);
        //}
    };

    using BMNNTensorPtr = std::shared_ptr<BMNNTensor>;


    class BMNNNetwork : public NoCopyable {
        const bm_net_info_t *m_netinfo;
        bm_tensor_t *m_inputTensors;
        bm_tensor_t *m_outputTensors;
        bm_handle_t m_handle;
        void *m_bmrt;
        std::vector<std::vector<bm_shape_t>> m_input_shapes;

        std::unordered_map<std::string, int> m_mapInputName2Index;
        std::unordered_map<std::string, int> m_mapOutputName2Index;

    public:
        static std::shared_ptr<BMNNNetwork> create(void *bmrt, const std::string& name)
        {
            return std::make_shared<BMNNNetwork>(bmrt, name);
        }

        friend std::ostream &operator<<(std::ostream &out, const BMNNNetwork &net)
        {
            return out << *net.m_netinfo;
        }

        BMNNNetwork(void *bmrt, const std::string &name) : m_bmrt(bmrt) {
            m_handle = static_cast<bm_handle_t>(bmrt_get_bm_handle(bmrt));
            m_netinfo = bmrt_get_network_info(bmrt, name.c_str());
            m_inputTensors = new bm_tensor_t[m_netinfo->input_num];
            m_outputTensors = new bm_tensor_t[m_netinfo->output_num];
            for (int i = 0; i < m_netinfo->input_num; ++i) {
                std::vector<bm_shape_t> shapes;
                for (int is = 0; is < m_netinfo->stage_num; ++is)
                    shapes.push_back(m_netinfo->stages[is].input_shapes[i]);
                m_input_shapes.push_back(std::move(shapes));

                m_inputTensors[i].dtype = m_netinfo->input_dtypes[i];
                m_inputTensors[i].shape = m_netinfo->stages[0].input_shapes[i];
                m_inputTensors[i].st_mode = BM_STORE_1N;
                m_inputTensors[i].device_mem = bm_mem_null();
                m_mapInputName2Index[m_netinfo->input_names[i]] = i;
            }

            for (int i = 0; i < m_netinfo->output_num; ++i) {
                m_outputTensors[i].dtype = m_netinfo->output_dtypes[i];
                m_outputTensors[i].shape = m_netinfo->stages[0].output_shapes[i];
                m_outputTensors[i].st_mode = BM_STORE_1N;
                m_outputTensors[i].device_mem = bm_mem_null();
                m_mapOutputName2Index[m_netinfo->output_names[i]] = i;
            }

            assert(m_netinfo->stage_num >= 1);
        }

        bool is_dynamic() const
        {
            return m_netinfo->is_dynamic;
        }

        // Get input shapes by input index
        // Returns shapes of all stages
        const std::vector<bm_shape_t> &get_input_shape(size_t index) const
        {
            assert(index < m_netinfo->input_num);
            return m_input_shapes[index];
        }

        bm_data_type_t get_input_dtype(size_t index) const {
            assert(index < m_netinfo->input_num);
            return m_netinfo->input_dtypes[index];
        }

        float get_input_scale(size_t index) const {
            assert(index < m_netinfo->input_num);
            return m_netinfo->input_scales[index];
        }

        float get_output_scale(size_t index) const {
            assert(index < m_netinfo->output_num);
            return m_netinfo->output_scales[index];
        }

        ~BMNNNetwork() {
            //Free input tensors
            delete [] m_inputTensors;
            //Free output tensors
            for(int i = 0; i < m_netinfo->output_num; ++i) {
                if (m_outputTensors[i].device_mem.size != 0) {
                    bm_free_device(m_handle, m_outputTensors[i].device_mem);
                }
            }
            delete []m_outputTensors;
        }

        int inputTensorNum() {
            return m_netinfo->input_num;
        }

        int inputName2Index(const std::string& name) {
            if (m_mapInputName2Index.find(name) != m_mapInputName2Index.end()) {
                return m_mapInputName2Index[name];
            }

            return -1;
        }

        int outputName2Index(const std::string& name) {
            if (m_mapOutputName2Index.find(name) != m_mapOutputName2Index.end()) {
                return m_mapOutputName2Index[name];
            }

            return -1;
        }

        std::shared_ptr<BMNNTensor> inputTensor(const std::string& name) {
            int index = outputName2Index(name);
            if (-1 == index) return nullptr;
            return std::make_shared<BMNNTensor>(m_handle, m_netinfo->input_names[index],
                                                m_netinfo->input_scales[index],
                                                &m_inputTensors[index]);
        }

        std::shared_ptr<BMNNTensor> inputTensor(int index) {
            assert(index < m_netinfo->input_num);
            return std::make_shared<BMNNTensor>(m_handle, m_netinfo->input_names[index],
                                                m_netinfo->input_scales[index],
                                                &m_inputTensors[index]);
        }

        int outputTensorNum() {
            return m_netinfo->output_num;
        }

        std::shared_ptr<BMNNTensor> outputTensor(int index) {
            assert(index < m_netinfo->output_num);
            return std::make_shared<BMNNTensor>(m_handle, m_netinfo->output_names[index],
                                                m_netinfo->output_scales[index], &m_outputTensors[index]);
        }

        int forward() {
            bool user_mem = false; // if false, bmrt will alloc mem every time.
            if (m_outputTensors->device_mem.size != 0) {
                // if true, bmrt don't alloc mem again.
                user_mem = true;
            }

            bool ok = bmrt_launch_tensor_ex(m_bmrt, m_netinfo->name, m_inputTensors, m_netinfo->input_num,
                                         m_outputTensors, m_netinfo->output_num, user_mem, true);
            if (!ok) {
                std::cout << "bm_launch_tensor() failed=" << std::endl;
                return -1;
            }

#if 0
            for(int i = 0;i < m_netinfo->output_num; ++i) {
                auto tensor = m_outputTensors[i];
                // dump
                std::cout << "output_tensor [" << i << "] size=" << bmrt_tensor_device_size(&tensor) << std::endl;
            }
#endif

            return 0;
        }

        int forward(const bm_tensor_t *input_tensors, int input_num, bm_tensor_t *output_tensors, int output_num)
        {
            bool ok = bmrt_launch_tensor_ex(m_bmrt, m_netinfo->name, input_tensors, input_num,
                    output_tensors, output_num, false, false);
            if (!ok) {
                std::cout << "bm_launch_tensor_ex() failed=" << std::endl;
                return -1;
            }

            return 0;
        }

        int forward_user_mem(
            const bm_tensor_t *input_tensors,
            int input_num,
            bm_tensor_t *output_tensors,
            int output_num)
        {
            bool ok = bmrt_launch_tensor_ex(m_bmrt, m_netinfo->name, input_tensors, input_num,
                    output_tensors, output_num, true, false);
            if (!ok) {
                std::cout << "bm_launch_tensor_ex() failed=" << std::endl;
                return -1;
            }

            return 0;
        }
    };

    using BMNNNetworkPtr=std::shared_ptr<BMNNNetwork>;

    class BMNNHandle : public NoCopyable {
        bm_handle_t m_handle;
        int m_dev_id;

    public:
        static std::shared_ptr<BMNNHandle> create(int dev_id) {
            return std::make_shared<BMNNHandle>(dev_id);
        }
        BMNNHandle(int dev_id = 0) : m_dev_id(dev_id) {
            int ret = bm_dev_request(&m_handle, dev_id);
            assert(BM_SUCCESS == ret);
        }

        ~BMNNHandle() {
            bm_dev_free(m_handle);
        }

        bm_handle_t handle() {
            return m_handle;
        }

        int dev_id() {
            return m_dev_id;
        }
    };

    using BMNNHandlePtr = std::shared_ptr<BMNNHandle>;

    class BMNNContext : public NoCopyable {
        BMNNHandlePtr m_handlePtr;
        void *m_bmrt;
        std::vector<std::string> m_network_names;

    public:
        static std::shared_ptr<BMNNContext> create(BMNNHandlePtr handle, const std::string& bmodel_file)
        {
            return std::make_shared<BMNNContext>(handle, bmodel_file);
        }

        BMNNContext(BMNNHandlePtr handle, const std::string& bmodel_file) : m_handlePtr(handle) {
            bm_handle_t hdev = m_handlePtr->handle();
            m_bmrt = bmrt_create(hdev);
            if (NULL == m_bmrt) {
                std::cout << "bmrt_create() failed!" << std::endl;
                exit(-1);
            }

            if (!bmrt_load_bmodel(m_bmrt, bmodel_file.c_str())) {
                std::cout << "load bmodel(" << bmodel_file << ") failed" << std::endl;
            }

            load_network_names();
        }

        ~BMNNContext() {
            if (m_bmrt != NULL) {
                bmrt_destroy(m_bmrt);
                m_bmrt = NULL;
            }
        }

        bm_handle_t handle() {
            return m_handlePtr->handle();
        }

        int dev_id() {
            return m_handlePtr->dev_id();
        }

        void *bmrt() {
            return m_bmrt;
        }

        void load_network_names() {
            const char **names;
            int num;
            num = bmrt_get_network_number(m_bmrt);
            bmrt_get_network_names(m_bmrt, &names);
            for (int i = 0; i < num; ++i) {
                m_network_names.push_back(names[i]);
            }

            free(names);
        }

        std::string network_name(int index) {
            if (index >= (int) m_network_names.size()) {
                return "Invalid index";
            }

            return m_network_names[index];
        }

        std::shared_ptr<BMNNNetwork> network(const std::string &net_name) {
            return std::make_shared<BMNNNetwork>(m_bmrt, net_name);
        }

        std::shared_ptr<BMNNNetwork> network(int net_index) {
            assert(net_index < (int) m_network_names.size());
            return std::make_shared<BMNNNetwork>(m_bmrt, m_network_names[net_index]);
        }


    };

    using BMNNContextPtr = std::shared_ptr<BMNNContext>;

} // end of namespace bm

#endif //YOLOV5_DEMO_BMNN_UTILS_H
