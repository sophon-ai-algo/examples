//
// Created by yuan on 3/25/21.
//

#ifndef INFERENCE_FRAMEWORK_BMUTILITY_STRING_H
#define INFERENCE_FRAMEWORK_BMUTILITY_STRING_H
#include <string>
#include <vector>
#include "bmutility_types.h"

namespace bm {
    std::vector<std::string> split(std::string str, std::string pattern);
    bool start_with(const std::string &str, const std::string &head);
    std::string file_name_from_path(const std::string& path, bool hasExt);
    std::string file_ext_from_path(const std::string& str);
    std::string format(const char *pszFmt, ...);

    std::string base64_enc(const void *data, size_t sz);
    std::string base64_dec(const void *data, size_t sz);

}
#endif //INFERENCE_FRAMEWORK_BMUTILITY_STRING_H
