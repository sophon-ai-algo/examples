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
// Created by yuan on 2/26/21.
//

#ifndef INFERENCE_FRAMEWORK_BMUTILITY_TIMER_H
#define INFERENCE_FRAMEWORK_BMUTILITY_TIMER_H
#include <iostream>
#include <memory>
#include <assert.h>
#include <chrono>
#include <functional>
#include <algorithm>
#include <queue>
#include <map>
#include <mutex>

#include <time.h>
#ifdef WIN32
#  include <WinSock2.h>
#  include <windows.h>
#else
#  include <sys/time.h>
#endif
#ifdef WIN32
static int
gettimeofday(struct timeval* tp, void* tzp)
{
    time_t clock;
    struct tm tm;
    SYSTEMTIME wtm;
    GetLocalTime(&wtm);
    tm.tm_year = wtm.wYear - 1900;
    tm.tm_mon = wtm.wMonth - 1;
    tm.tm_mday = wtm.wDay;
    tm.tm_hour = wtm.wHour;
    tm.tm_min = wtm.wMinute;
    tm.tm_sec = wtm.wSecond;
    tm.tm_isdst = -1;
    clock = mktime(&tm);
    tp->tv_sec = clock;
    tp->tv_usec = wtm.wMilliseconds * 1000;
    return (0);
}
#endif

namespace bm {

    uint64_t gettime_sec();
    uint64_t gettime_msec();
    uint64_t gettime_usec();
    void msleep(int msec);
    void usleep(int usec);
    std::string timeToString(time_t seconds);

    class TimerQueue {
    public:
        static std::shared_ptr<TimerQueue> create();
        virtual ~TimerQueue() {
            std::cout << "TimerQueue dtor" << std::endl;
        };

        virtual int create_timer(uint32_t delay_msec, std::function<void()> func, int repeat, uint64_t *p_timer_id) = 0;
        virtual int delete_timer(uint64_t timer_id) = 0;
        virtual size_t count() = 0;
        virtual int run_loop() = 0;
        virtual int stop() = 0;
    };

    using TimerQueuePtr = std::shared_ptr<TimerQueue>;

    class StatTool {
    public:
        static std::shared_ptr<StatTool> create(int range=5);
        virtual ~StatTool(){};

        virtual void update(uint64_t currentStatis) = 0;
        virtual void reset() = 0;
        virtual double getkbps() = 0;
        virtual double getSpeed() = 0;
    };

    using StatToolPtr = std::shared_ptr<StatTool>;

    class BMPerf {
        int64_t start_us_;
        std::string tag_;
        int threshold_{50};
    public:
        BMPerf() {}
        BMPerf(const std::string &name, int threshold = 0) {
            begin(name, threshold);
        }
        ~BMPerf() {}

        void begin(const std::string &name, int threshold = 0) {
            tag_ = name;
            auto n = std::chrono::steady_clock::now();
            start_us_ = n.time_since_epoch().count();
            threshold_ = threshold;
        }

        void end() {
            auto n = std::chrono::steady_clock::now().time_since_epoch().count();
            auto delta = (n - start_us_) / 1000;
            if (delta < threshold_ * 1000) {
                //printf("%s used:%d us\n", tag_.c_str(), delta);
            } else {
                printf("WARN:%s used:%d ms > %d ms\n", tag_.c_str(), (int)delta/1000, (int)threshold_);
            }
        }
    };
}


#endif //INFERENCE_FRAMEWORK_BMUTILITY_TIMER_H
