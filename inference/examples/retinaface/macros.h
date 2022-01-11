#pragma once

#define call(fn, ...) \
    do { \
        auto ret = fn(__VA_ARGS__); \
        if (ret != BM_SUCCESS) \
        { \
            LOG(ERROR) << #fn << " failed " << ret; \
            throw std::runtime_error("api error"); \
        } \
    } while (false);
