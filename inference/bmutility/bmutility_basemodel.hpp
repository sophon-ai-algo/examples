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
#ifndef _INFERENCEFRAMEWORK_BNUTILITY_BASEMODEL_H_
#define _INFERENCEFRAMEWORK_BNUTILITY_BASEMODEL_H_
#include <memory>
#include <vector>

#define DEFINE_BASEMODEL_THRES_SETTER_GETTER(T, t)    \
inline void set_##t(T val) { m_##t##_thres = val;  }  \
inline T    get_##t(T val) { return m_##t##_thres; }


namespace bm {

class BaseModel : public std::enable_shared_from_this<BaseModel> {
public:
    DEFINE_BASEMODEL_THRES_SETTER_GETTER(float, cls)
    DEFINE_BASEMODEL_THRES_SETTER_GETTER(float, nms)
    DEFINE_BASEMODEL_THRES_SETTER_GETTER(float, obj)
    inline int getBatch() { return m_max_batch; }
protected:
    BaseModel()           = default;
    virtual ~BaseModel()  = default;

protected:
    float m_cls_thres {0.5};
    float m_nms_thres {0.5};
    float m_obj_thres {0.5};
    int   m_max_batch {1};
    std::vector<std::string> m_class_names;

};
} // namespace bm
#endif // _INFERENCEFRAMEWORK_BNUTILITY_BASEMODEL_H_