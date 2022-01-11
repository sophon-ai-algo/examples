#include "bmutility_pool.h"

#define call(fn, ...) \
    do { \
        auto ret = fn(__VA_ARGS__); \
        if (ret != BM_SUCCESS) \
        { \
            std::cerr << #fn << " failed " << ret << std::endl; \
            throw std::runtime_error("api error"); \
        } \
    } while (false);

#define align_with(v, n) ((int(v) + n - 1) / n * n)

namespace bm {
    DeviceMemoryPool::DeviceMemoryPool(bm_handle_t handle)
        : handle_(handle)
    {
    }

    DeviceMemoryPool::~DeviceMemoryPool()
    {
        std::lock_guard<decltype(mut_)> lock(mut_);
        for (auto &p : history_)
        {
            bm_free_device(handle_, p.second);
        }
    }

    bool DeviceMemoryPool::try_alloc_from_registry(
        bm_device_mem_t &mem, size_t size, size_t heap)
    {
        if (registry_[heap].empty())
            return false;
        auto &registry = registry_[heap];
        for (int i = 0; i < registry.size(); ++i)
        {
            auto &m = registry[i];
            if (m.size >= size)
            {
                mem = m;
                registry.erase(registry.begin() + i);
                return true;
            }
        }
        return false;
    }

    bm_device_mem_t DeviceMemoryPool::alloc(size_t size, size_t heap)
    {
        unsigned long long addr;
        if (heap > max_ddr)
        {
            std::cerr << "invalid heap id " << heap << std::endl;
            throw std::runtime_error("invalid argument");
        }
        std::lock_guard<decltype(mut_)> lock(mut_);
        bm_device_mem_t mem;
        if (this->try_alloc_from_registry(mem, size, heap))
            goto ret;
        call(
            bm_malloc_device_byte_heap, handle_,
            &mem, heap, size);
ret:
        addr = bm_mem_get_device_addr(mem);
        history_[addr] = mem;
        return mem;
    }

    void DeviceMemoryPool::alloc(
        size_t num, bm_image *images,
        size_t height, size_t width,
        bm_image_format_ext img_format,
        bm_image_data_format_ext data_type,
        int align, size_t heap)
    {
        int data_size = 1;
        if (data_type == DATA_TYPE_EXT_FLOAT32) {
            data_size = 4;
        }
        int stride[3] = {0};
        size_t mem_size;
        int img_w_real = width * data_size;
        if (FORMAT_RGB_PLANAR == img_format ||
            FORMAT_RGB_PACKED == img_format ||
            FORMAT_BGR_PLANAR == img_format ||
            FORMAT_BGR_PACKED == img_format) {
            stride[0] = align_with(width, align) * data_size;
            mem_size = stride[0] * height * 3 * data_size;
        } else if (FORMAT_YUV420P == img_format) {
            stride[0] = align_with(width, align) * data_size;
            stride[1] = stride[2] = align_with(width >> 1, align) * data_size;
            mem_size = stride[0] * height * 3 / 2 * data_size;
        } else if (FORMAT_NV12 == img_format || FORMAT_NV21 == img_format) {
            stride[0] = align_with(width, align) * data_size;
            stride[1] = align_with(width >> 1, align) * data_size;
            mem_size = stride[0] * height * 3 / 2 * data_size;
        } else {
            std::cerr << "not supported image format "
                      << img_format << std::endl;
            throw std::runtime_error("invalid argument");
        }

        for (int i = 0; i < num; ++i)
            call(
                bm_image_create,
                handle_, height, width,
                img_format, data_type,
                &images[i], stride);
        auto mem = this->alloc(mem_size * num, heap);
        call(bm_image_attach_contiguous_mem, num, images, mem);
    }

    void DeviceMemoryPool::free(size_t num, bm_image *images)
    {
        bm_device_mem_t mem;
        call(bm_image_get_contiguous_device_mem, num, images, &mem);
        this->free(mem);
        for (int i = 0; i < num; ++i)
            call(bm_image_destroy, images[i]);
    }

    void DeviceMemoryPool::alloc(bm_tensor_t &tensor)
    {
        const size_t heap = 0;
        size_t size = bmrt_shape_count(&tensor.shape);
        if (tensor.dtype == BM_FLOAT32)
            size *= 4;
        tensor.device_mem = this->alloc(size, heap);
        auto addr = bm_mem_get_device_addr(tensor.device_mem);
    }

    void DeviceMemoryPool::free(bm_tensor_t tensor)
    {
        auto addr = bm_mem_get_device_addr(tensor.device_mem);
        this->free(tensor.device_mem);
    }

    void DeviceMemoryPool::free(bm_device_mem_t mem)
    {
        std::lock_guard<decltype(mut_)> lock(mut_);
        auto addr = bm_mem_get_device_addr(mem);
        if (history_.find(addr) == history_.end())
        {
            std::cerr << "failed to find addr "
                      << reinterpret_cast<void *>(addr) << std::endl;
            throw std::runtime_error("invalid argument");
        }
        auto old_mem = history_[addr];
        history_.erase(addr);
        unsigned heap;
        call(bm_get_gmem_heap_id, handle_, &old_mem, &heap);
        registry_[heap].push_back(old_mem);
    }
}

