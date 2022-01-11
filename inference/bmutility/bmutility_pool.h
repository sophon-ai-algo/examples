#pragma once
#include "bmutility.h"

namespace bm {
    class DeviceMemoryPool {
    private:
        std::mutex mut_;
        static const size_t max_ddr = 3;
        bm_handle_t handle_;
        std::vector<bm_device_mem_t> registry_[max_ddr];
        std::unordered_map<unsigned long long, bm_device_mem_t> history_;
        bool try_alloc_from_registry(bm_device_mem_t &, size_t size, size_t heap);

    public:
        DeviceMemoryPool(bm_handle_t);
        ~DeviceMemoryPool();
        bm_device_mem_t alloc(size_t size, size_t heap);
        void free(bm_device_mem_t mem);

        void alloc(
            size_t num, bm_image *images,
            size_t height, size_t width,
            bm_image_format_ext img_format,
            bm_image_data_format_ext data_type,
            int align, size_t heap);
        void free(size_t num, bm_image *images);

        void alloc(bm_tensor_t &tensor);
        void free(bm_tensor_t tensor);
    };
}

