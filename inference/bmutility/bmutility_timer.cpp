//
// Created by yuan on 2/26/21.
//

#include "bmutility_timer.h"
#include <algorithm>
#include <thread>
#include <memory.h>
#include <atomic>

namespace bm {

#define RTC_RETURN_EXP_IF_FAIL(cond, exp) if (!(cond)) { fprintf(stderr, "Assert failed: %s in %s:%d\n", #cond, __FUNCTION__, __LINE__); exp;}

#define rtc_container_of(ptr, type, member)  ((type *) ((char *) (ptr) - offsetof(type, member)))
    uint64_t gettime_msec() {
        auto tnow = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(tnow.time_since_epoch()).count();
    }

    uint64_t gettime_usec() {
        auto tnow = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(tnow.time_since_epoch()).count();
    }

    uint64_t gettime_sec() {
        auto tnow = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::seconds>(tnow.time_since_epoch()).count();
    }



    void msleep(int msec) {
        std::this_thread::sleep_for(std::chrono::milliseconds(msec));
    }

    void usleep(int usec) {
        std::this_thread::sleep_for(std::chrono::microseconds(usec));
    }

    std::string timeToString(time_t sec) {
        struct tm* tm1;
        tm1 = std::localtime(&sec);
        char strtmp[64];
        sprintf(strtmp, "%d-%.2d-%.2d:%.2d:%.2d:%.2d", tm1->tm_year+1900,
                tm1->tm_mon+1, tm1->tm_mday, tm1->tm_hour,
                tm1->tm_min, tm1->tm_sec);
        return strtmp;
    }

    struct BMTimer {
        std::function<void()> lamdaCb;
        uint64_t timeout;
        uint64_t delay_msec;
        int repeat;
        uint64_t start_id;
    };
    using BMTimerPtr=std::shared_ptr<BMTimer>;

    template<typename T>
    class MinHeap : public std::priority_queue<T, std::vector<T>> {
    public:
        bool remove(const T &value) {
            auto it = std::find(this->c.begin(), this->c.end(), value);
            if (it != this->c.end()) {
                this->c.erase(it);
                std::make_heap(this->c.begin(), this->c.end(), this->comp);
                return true;
            } else {
                return false;
            }
        }
    };

    class BMTimerQueue: public TimerQueue {
        int generate_timer(uint32_t delay_msec, std::function<void()> func, int repeat, uint64_t *p_timer_id);
        std::map<uint64_t, BMTimerPtr> m_mapTimers;
        MinHeap<BMTimerPtr> m_QTimers;
        uint64_t m_nTimerSN;
        std::mutex m_mLock;
        bool m_isRunning;
        bool m_stopped;

    public:
        BMTimerQueue():m_nTimerSN(0), m_isRunning(false), m_stopped(true) {
            std::cout << "BMTimerQueue ctor" << std::endl;

        }

        ~BMTimerQueue()
        {
            m_isRunning = false;
            while(!m_stopped) msleep(10);
            std::cout << "BMTimerQueue dtor" << std::endl;
        }

        virtual int create_timer(uint32_t delay_msec, std::function<void()> func, int repeat, uint64_t *p_timer_id) override
        {
            RTC_RETURN_EXP_IF_FAIL(func != nullptr , return -1);
            BMTimerPtr timer = std::make_shared<BMTimer>();
            if (NULL == timer)
            {
                return -1;
            }

            timer->lamdaCb = func;
            timer->timeout = gettime_msec() + delay_msec;
            timer->delay_msec = delay_msec;
            timer->start_id = m_nTimerSN ++;
            timer->repeat = repeat;

            std::unique_lock<std::mutex> locker(m_mLock);

            // Add to Heap
            m_QTimers.push(timer);

            // Add to HashTable
            m_mapTimers[timer->start_id] = timer;
            if (p_timer_id)
            {
                *p_timer_id = timer->start_id;
            }

            return 0;
        }

        virtual int delete_timer(uint64_t timer_id) override {
            std::unique_lock<std::mutex> locker(m_mLock);
            if (m_mapTimers.find(timer_id) == m_mapTimers.end())
            {
                std::cout << "delete_timer(),can't find timer = " << timer_id << std::endl;
                return 0;
            }

            auto timer = m_mapTimers[timer_id];
            m_QTimers.remove(timer);
            m_mapTimers.erase(timer_id);
            timer = nullptr;
            return 0;
        }

        virtual size_t count() override {
            return m_QTimers.size();
        }

        virtual int run_loop() override {
            m_isRunning = true;
            m_stopped = false;
            while (m_isRunning)
            {
                auto timeNow = gettime_msec();
                m_mLock.lock();
                if (m_mapTimers.size() == 0)
                {
                    m_mLock.unlock();
                    msleep(1);
                    continue;
                }

                uint64_t timer_id;
                auto timer = m_QTimers.top();
                timer_id = timer->start_id;
                if (timeNow < timer->timeout)
                {
                    m_mLock.unlock();
                    usleep(1); //sleep 1 million second
                    continue;
                }

                m_mLock.unlock();

                if (timer->lamdaCb != nullptr) {
                    timer->lamdaCb();
                }

                m_mLock.lock();

                if (m_mapTimers.find(timer_id) != m_mapTimers.end()) {

                    if (timer->repeat) {
                        // repeated timer
                        timer->timeout += timer->delay_msec;
                        m_QTimers.push(timer);
                    }
                    else {
                        // oneshot timer
                        m_mapTimers.erase(timer_id);
                    }
                }
                else {
                    // timer is deleted, not existed any more.
                }

                m_mLock.unlock();
            }
            std::cout << "rtc_timer_queue exit!" << std::endl;
            m_stopped= true;
            return 1;
        }

        virtual int stop() override {
            m_isRunning = false;
            return 0;
        }
    };

    std::shared_ptr<TimerQueue> TimerQueue::create() {
        return std::make_shared<BMTimerQueue>();
    }


    class BMStatTool: public StatTool {
        struct statis_layer {
            uint64_t bytes;
            uint64_t time_msec;
        };

        statis_layer *m_layers;
        int m_current_index;
        uint32_t m_total_layers;
        uint32_t m_record_count;

    public:
        BMStatTool(int range=5):m_current_index(0),m_record_count(0) {
            m_total_layers = range;
            m_layers = new statis_layer[range];
            assert(NULL != m_layers);
        }
        virtual ~BMStatTool(){
            delete []m_layers;
        };

        virtual void update(uint64_t currentStatis) override {
            uint32_t current_index = m_current_index;
            m_layers[current_index].time_msec = gettime_msec();
            m_layers[current_index].bytes = currentStatis;

            current_index = (current_index+1) % m_total_layers;
            m_current_index = current_index;

            if (m_record_count < m_total_layers)
            {
                m_record_count++;
            }
        }

        virtual void reset() override {
            m_current_index = 0;
            m_record_count = 0;

            memset(m_layers, 0, sizeof(m_layers[0]) * m_total_layers);
        }

        virtual double getkbps() override {
            return getSpeed()*8*0.001;
        }

        virtual double getSpeed() override {
            uint32_t currentIndex = 0;
            uint32_t newest, oldest;
            double bps = 0.0;
            uint64_t time_diff = 0, byte_diff;

            currentIndex = m_current_index;
            if (m_record_count < m_total_layers)
            {
                newest = currentIndex > 0 ? currentIndex - 1 : 0;
                oldest = 0;
            }
            else
            {
                newest = (m_total_layers + (currentIndex - 1)) % m_total_layers;
                oldest = currentIndex;
            }


            time_diff = m_layers[newest].time_msec - m_layers[oldest].time_msec;
            byte_diff = m_layers[newest].bytes - m_layers[oldest].bytes;

            bps = (double)(byte_diff) * 1000 / (time_diff);
            return bps;
        }
    };

    std::shared_ptr<StatTool> StatTool::create(int range) {
        return std::make_shared<BMStatTool>(range);
    }

}