#pragma once
#include "bmutility.h"

namespace bm {
    class Watch;
}
static std::ostream &operator<<(std::ostream &out, const bm::Watch &w);

namespace bm {
    class Watch {
    private:
        mutable std::mutex mut_;
        struct record {
            bool set = false;
            std::chrono::steady_clock::time_point mark;
            std::vector<std::chrono::nanoseconds> laps;
        };
        std::unordered_map<std::string, record> records_;
    
    public:
        void mark(const std::string &key) {
            auto now = std::chrono::steady_clock::now();
            std::lock_guard<decltype(mut_)> lock(mut_);
            auto &r = records_[key];
            if (r.set)
            {
                r.laps.push_back(now - r.mark);
            } else {
                r.mark = now;
            }
            r.set = !r.set;
        }
        friend std::ostream &::operator<<(std::ostream &out, const Watch &);
    };
}

static std::ostream &operator<<(std::ostream &out, const bm::Watch &w)
{
    std::stringstream ss;
    ss << "============Profile=================\n";
    std::lock_guard<std::mutex> lock(w.mut_);
    for (const auto &p : w.records_)
    {
        const auto &r = p.second;
        std::chrono::nanoseconds acc(0);
        for (auto dur : r.laps)
            acc += dur;
        ss << "[" << p.first << "] "
           << acc.count() / 1.e6 / r.laps.size()
           << "ms" << std::endl;
    }
    return out << ss.str();
}

