//
// Created by hsyuan on 2021-02-22.
//

#ifndef INFERENCE_FRAMEWORK_INFERENCE_H
#define INFERENCE_FRAMEWORK_INFERENCE_H

#include "bmutility.h"
#include "thread_queue.h"
namespace bm {

    // declare before
    template<typename T> class BMInferencePipe;

    template<typename T>
    class DetectorDelegate {
    protected:
        using DetectedFinishFunc = std::function<void(T &of)>;
        DetectedFinishFunc m_pfnDetectFinish = nullptr;
        BMInferencePipe<T>* m_nextInferPipe = nullptr;
    public:
        virtual ~DetectorDelegate() {}
        void set_next_inference_pipe(BMInferencePipe<T> *nextPipe) { m_nextInferPipe = nextPipe; }
        int set_detected_callback(DetectedFinishFunc func) { m_pfnDetectFinish = func; return 0; }

        virtual int preprocess(std::vector<T> &frames) = 0;

        virtual int forward(std::vector<T> &frames) = 0;

        virtual int postprocess(std::vector<T> &frames) = 0;

    };

    struct DetectorParam {
        DetectorParam() {
            preprocess_queue_size = 5;
            preprocess_thread_num = 4;

            inference_queue_size = 5;
            inference_thread_num = 1;

            postprocess_queue_size = 5;
            postprocess_thread_num = 2;
            batch_size = 4;
        }

        int preprocess_queue_size;
        int preprocess_thread_num;

        int inference_queue_size;
        int inference_thread_num;

        int postprocess_queue_size;
        int postprocess_thread_num;

        int batch_size;
    };

    template<typename T>
    class BMInferencePipe {
        DetectorParam m_param;
        std::shared_ptr<DetectorDelegate<T>> m_detect_delegate;

        BlockingQueue<T> m_preprocessQue;
        BlockingQueue<T> m_postprocessQue;
        BlockingQueue<T> m_forwardQue;

        WorkerPool<T> m_preprocessWorkerPool;
        WorkerPool<T> m_forwardWorkerPool;
        WorkerPool<T> m_postprocessWorkerPool;


    public:
        BMInferencePipe() {

        }

        virtual ~BMInferencePipe() {

        }

        int init(const DetectorParam &param, std::shared_ptr<DetectorDelegate<T>> delegate) {
            m_param = param;
            m_detect_delegate = delegate;

            m_preprocessWorkerPool.init(&m_preprocessQue, param.preprocess_thread_num, param.batch_size, param.batch_size);
            m_preprocessWorkerPool.startWork([this](std::vector<T> &items) {
                if (m_preprocessQue.size() > m_param.preprocess_queue_size) {
                    std::cout << "WARNING:preprocess queue_size(" << m_preprocessQue.size() << ") > "
                              << m_param.preprocess_queue_size << std::endl;
                }

                m_detect_delegate->preprocess(items);
                this->m_forwardQue.push(items);
            });

            m_forwardWorkerPool.init(&m_forwardQue, param.inference_thread_num, 1, 8);
            m_forwardWorkerPool.startWork([this](std::vector<T> &items) {
                if (m_forwardQue.size() > m_param.inference_queue_size) {
                    std::cout << "WARNING:forward queue_size(" << m_forwardQue.size() << ") > "
                              << m_param.inference_queue_size << std::endl;
                }

                m_detect_delegate->forward(items);
                this->m_postprocessQue.push(items);
            });

            m_postprocessWorkerPool.init(&m_postprocessQue, param.postprocess_thread_num, 1, 8);
            m_postprocessWorkerPool.startWork([this](std::vector<T> &items) {
                if (m_postprocessQue.size() > m_param.postprocess_queue_size) {
                    std::cout << "WARNING:postprocess queue_size(" << m_postprocessQue.size() << ") > "
                              << m_param.postprocess_queue_size << std::endl;
                }

                m_detect_delegate->postprocess(items);
            });
            return 0;
        }

        int flush_frame() {
            return m_preprocessWorkerPool.flush();
        }

        int push_frame(T *frame) {
            return m_preprocessQue.push(*frame);
        }
    };
} // end namespace bm


#endif //INFERENCE_FRAMEWORK_INFERENCE_H
