#include <assert.h>
#include <map>
#include <vector>
#include "bmcv_api_ext.h"
#define FFALIGN(x, a) (((x)+(a)-1)&~((a)-1))

int main(int argc, char* argv[]) {
    std::map<uint64_t, float> a;
    a[1] = 1.f;
    a[2] = 2.f;
    std::map<uint64_t, float> b;
    b[2] = 22.f;
    b[3] = 33.f;
    a.insert(b.begin(), b.end());
    bm_handle_t handle;
    int ret = bm_dev_request(&handle, 0);
    assert(ret == BM_SUCCESS);

    int data_size = 1;
    int img_w     = 224;
    int img_h     = 224;
    int align     = 64;
    int batch_num = 4;
    int mask      = 4;

    bm_image_format_ext img_format = FORMAT_BGR_PLANAR;
    bm_image_data_format_ext data_type = DATA_TYPE_EXT_1N_BYTE;
    std::vector<bm_image> vImages;
    vImages.resize(batch_num);
    bm_image *image = vImages.data();

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

    for (int i = 0; i < batch_num; i++) {
        bm_image_create(handle, img_h, img_w, img_format, data_type, &image[i], stride);
    }

    ret = bm_image_alloc_contiguous_mem_heap_mask(batch_num, image, mask);
    assert(BM_SUCCESS == ret);

    return 0;
}