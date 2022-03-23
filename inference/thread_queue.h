//
// Created by yuan on 2/22/21.
//

#ifndef INFERENCE_FRAMEWORK_THREAD_QUEUE_H
#define INFERENCE_FRAMEWORK_THREAD_QUEUE_H

#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <functional>
#include <thread>
#ifdef __linux__
#include <sys/time.h>
#endif 
#include <pthread.h>
#include "bmutility_timer.h"

template <typename T>
class BlockingQueue {
private:
    size_t size_impl() const
    {
        return m_type == 0 ? m_queue.size() : m_vec.size();
    }

    void wait_and_push_one(T &&data) {
        if (m_limit > 0 && this->size_impl() >= m_limit && !m_stop) {
            std::cout << "WARNING: " << m_name << " queue_size(" << this->size_impl() << ") > "
                      << m_limit << std::endl;
            // flow control by dropping
            if (m_drop_fn != nullptr) {
                this->drop_half_();
                std::cout << m_name << " queue_size after dropping, size: " << this->size_impl() << std::endl;;
            } else {
            // blocking
                do 
                {
                    pthread_cond_wait(&m_pop_cond, &m_qmtx);
                } while (m_limit > 0 && this->size_impl() >= m_limit && !m_stop);
            }
        }

        if (m_type == 0)
        {
            m_queue.push(std::move(data));
        } else {
            m_vec.push_back(std::move(data));
        }
    }

public:
    BlockingQueue(const std::string& name="" ,int type=0, int limit = 0)
        : m_stop(false), m_limit(limit), m_drop_fn(nullptr) {
        m_name = name;
        m_type = type;
        pthread_mutex_init(&m_qmtx, NULL);
        pthread_cond_init(&m_condv, NULL);
        pthread_cond_init(&m_pop_cond, NULL);
    }

    ~BlockingQueue() {
        pthread_mutex_lock(&m_qmtx);

        std::cout<<"destroy "<<m_name<<",size:"<<m_queue.size()+m_vec.size()<<std::endl;
        m_vec.clear();
        std::queue<T> empty;
        m_queue.swap(empty);
        pthread_mutex_unlock(&m_qmtx);
    }

    void stop() {
        pthread_mutex_lock(&m_qmtx);
        m_stop = true;
        std::cout<< "stop blocking queue:" << m_name << std::endl;
        pthread_cond_broadcast(&m_pop_cond);
        pthread_cond_broadcast(&m_condv);
        pthread_mutex_unlock(&m_qmtx);
    }

    int push(T &data) {
        pthread_mutex_lock(&m_qmtx);

        this->wait_and_push_one(std::move(data));
        int num = this->size_impl();

        pthread_mutex_unlock(&m_qmtx);
        pthread_cond_signal(&m_condv);
        return num;
    }

    int push(std::vector<T> &datas) {
        int num;
        pthread_mutex_lock(&m_qmtx);

        for (auto &data : datas)
        {
            this->wait_and_push_one(std::move(data));
            if (m_stop) goto err;
        }
        num = this->size_impl();

        pthread_cond_signal(&m_condv);
        pthread_mutex_unlock(&m_qmtx);
        return num;

err:
        pthread_mutex_unlock(&m_qmtx);
        return 0;
    }
    int pop_front(std::vector<T>& objs, int min_num, int max_num, long wait_ms = 0, bool* is_timeout=nullptr) {
        bool timeout = false;

        struct timespec to;
        struct timeval now;
        gettimeofday(&now, NULL);
        double ms0 = now.tv_sec*1000+now.tv_usec/1000.0;
        //std::cout<<m_name<<",pop:"<<now.tv_usec/1000.0;
        if (wait_ms == 0) {
            to.tv_sec = now.tv_sec + 9999999;
            to.tv_nsec = now.tv_usec * 1000UL;
        } else {
            int nsec = now.tv_usec * 1000 + (wait_ms % 1000) * 1000000;
            to.tv_sec = now.tv_sec + nsec / 1000000000 + wait_ms / 1000;
            to.tv_nsec = nsec%1000000000;//(now.tv_usec + wait_ms * 1000UL) * 1000UL;
        }
        pthread_mutex_lock(&m_qmtx);
        while ((m_type?m_vec.size() < min_num:m_queue.size() < min_num) && !m_stop) {
#ifdef BLOCKING_QUEUE_PERF
            m_timer.tic();
#endif

            // pthread_timestruc_t to;
            int err = pthread_cond_timedwait(&m_condv, &m_qmtx, &to);
            if (err == ETIMEDOUT) {
                timeout = true;
                break;
            }
#ifdef BLOCKING_QUEUE_PERF
            m_timer.toc();
    if (m_timer.total_time_ > 1) {
      m_timer.summary();
    }
#endif
        }

        if (!timeout) {
            if(m_type == 0){
                int oc = 0;
                while(oc < max_num && m_queue.size() > 0) {
                    auto o = std::move(m_queue.front());
                    m_queue.pop();
                    objs.push_back(o);
                    oc++;
                }
            }else{
                int oc = 0;
                while(oc < max_num && m_vec.size() > 0) {
                    auto o = std::move(m_vec[0]);
                    m_vec.erase(m_vec.begin());
                    objs.push_back(o);
                    oc++;
                }
            }
        }
        pthread_cond_broadcast(&m_pop_cond);
        pthread_mutex_unlock(&m_qmtx);

        if (m_stop) {
            return 0;
        }

        if (is_timeout) {
            *is_timeout = true;
            return -1;
        }

        return 0;
    }

    size_t size() {
        size_t queue_size;
        pthread_mutex_lock(&m_qmtx);
        queue_size = this->size_impl();
        pthread_mutex_unlock(&m_qmtx);
        return queue_size;
    }

    int set_drop_fn(std::function<void(T& obj)> fn) {
        m_drop_fn = fn;
        return m_limit;
    }

    void drop_half_() {
        if (m_type == 0) {
            std::queue<T> temp;
            size_t num = m_queue.size();
            for(size_t i = 0; i < num; i++) {
                auto elem = m_queue.front();
                if (i % 2 == 0) {
                    temp.push(elem); 
                }
                else{
                    m_drop_fn(elem);
                }
                m_queue.pop();
            }
            m_queue.swap(temp);
        }
        else {
            std::vector<T> temp;
            size_t num = m_vec.size();
            for (size_t i = 0; i < num; i++) {
                auto elem = m_vec[i];
                if (i % 2 == 0) {
                    temp.push_back(elem); 
                }
                else{
                    m_drop_fn(elem);
                }
            }
            m_vec.swap(temp);
        }
    }

    void drop(int num=0){
        int queue_size;
        pthread_mutex_lock(&m_qmtx);
        if (num == 0) {
            num = this->size_impl();
        }
        if (this->size_impl() < num)
            return;
        if(m_type == 0){
            queue_size = m_queue.size();
            if(num>queue_size)
                num=queue_size;
            for(int i = 0; i<num;i++){
                m_queue.pop();
            }
        }
        else{
            queue_size = m_vec.size();
            if(num>queue_size)
                num=queue_size;
            m_vec.erase(m_vec.begin(),m_vec.begin()+num);
        }
        pthread_cond_broadcast(&m_pop_cond);
        pthread_mutex_unlock(&m_qmtx);
    }
    const std::string &name() { return m_name; }

private:
    bool m_stop;
    // Timer m_timer;
    std::string m_name;
    std::vector<T> m_vec;
    std::queue<T> m_queue;
    pthread_mutex_t m_qmtx;
    pthread_cond_t m_condv, m_pop_cond;
    int m_type, m_limit; //0:queue,1:vector
    std::function<void(T& obj)> m_drop_fn;
};

template <typename T>
class WorkerPool {
    BlockingQueue<T> *m_work_que;
    int m_thread_num;
    using OnWorkItemsCallback = std::function<void(std::vector<T> &item)>;
    OnWorkItemsCallback m_work_item_func;
    std::vector<std::thread *> m_threads;
    int m_max_pop_num;
    int m_min_pop_num;
public:
    WorkerPool():m_work_que(nullptr),m_thread_num(0),m_work_item_func(nullptr),m_max_pop_num(1),
    m_min_pop_num(1) {}

    virtual ~WorkerPool() {}

    int init(BlockingQueue<T> *que, int thread_num, int min_pop_num, int max_pop_num) {
        m_work_que = que;
        m_thread_num = thread_num;
        m_min_pop_num = min_pop_num;
        m_max_pop_num = max_pop_num;
        return 0;
    }


    int startWork(OnWorkItemsCallback fn) {
        m_work_item_func = fn;

        for (int i = 0; i < m_thread_num; ++i) {
            auto pth = new std::thread([this] {
                while (true) {
                    std::vector<T> items;
                    //if (m_work_que->size() < 4) { bm::usleep(10); continue; }
                    if (m_work_que->pop_front(items, m_min_pop_num, m_max_pop_num) != 0) {
                        break;
                    }
                    if (items.empty())
                        break;
                    m_work_item_func(items);
                }
            });

            m_threads.push_back(pth);
        }
        return 0;
    }

    int stopWork() {
        m_work_que->stop();
        for (int i = 0; i < m_thread_num; i++) {
            m_threads[i]->join();
            delete m_threads[i];
            m_threads[i] = nullptr;
        }
        return 0;
    }

    int flush() {
        m_work_que->stop();
        return 0;
    }
};



#endif //INFERENCE_FRAMEWORK_THREAD_QUEUE_H
