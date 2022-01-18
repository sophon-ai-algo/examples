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
