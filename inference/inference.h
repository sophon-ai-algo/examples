//
// Created by hsyuan on 2021-02-22.
//

#ifndef INFERENCE_FRAMEWORK_INFERENCE_H
#define INFERENCE_FRAMEWORK_INFERENCE_H

#include "bmutility.h"
#include "thread_queue.h"

namespace bm {
    template<typename T1, typename T2>
    class DetectorDelegate {
    protected:
        using DetectedFinishFunc = std::function<void(T2 &of)>;
        DetectedFinishFunc m_pfnDetectFinish;
    public:
        virtual ~DetectorDelegate() {}

        virtual void decode_process(T1 &) {
            // do nothing by default
        }

        virtual int preprocess(std::vector<T1> &frames, std::vector<T2> &of) = 0;

        virtual int forward(std::vector<T2> &frames) = 0;

        virtual int postprocess(std::vector<T2> &frames) = 0;

        virtual int set_detected_callback(DetectedFinishFunc func) { m_pfnDetectFinish = func; return 0;};
    };

    struct DetectorParam {
        DetectorParam() {
            preprocess_queue_size = 5;
            preprocess_thread_num = 4;

            inference_queue_size = 5;
            inference_thread_num = 1;

            postprocess_queue_size = 5;
            postprocess_thread_num = 2;
            batch_num=4;
        }

        int preprocess_queue_size;
        int preprocess_thread_num;

        int inference_queue_size;
        int inference_thread_num;

        int postprocess_queue_size;
        int postprocess_thread_num;
        int batch_num;

    };

    template<typename T1, typename T2>
    class BMInferencePipe {
        DetectorParam m_param;
        std::shared_ptr<DetectorDelegate<T1, T2>> m_detect_delegate;

        std::shared_ptr<BlockingQueue<T1>> m_preprocessQue;
        std::shared_ptr<BlockingQueue<T2>> m_postprocessQue;
        std::shared_ptr<BlockingQueue<T2>> m_forwardQue;

        WorkerPool<T1> m_preprocessWorkerPool;
        WorkerPool<T2> m_forwardWorkerPool;
        WorkerPool<T2> m_postprocessWorkerPool;


    public:
        BMInferencePipe() {

        }

        virtual ~BMInferencePipe() {

        }

        int init(const DetectorParam &param, std::shared_ptr<DetectorDelegate<T1, T2>> delegate) {
            m_param = param;
            m_detect_delegate = delegate;

            const int underlying_type_std_queue = 0;
            m_preprocessQue = std::make_shared<BlockingQueue<T1>>(
                "preprocess", underlying_type_std_queue,
                param.preprocess_queue_size);
            m_postprocessQue = std::make_shared<BlockingQueue<T2>>(
                "postprocess", underlying_type_std_queue,
                param.postprocess_queue_size);
            m_forwardQue = std::make_shared<BlockingQueue<T2>>(
                "inference", underlying_type_std_queue,
                param.inference_queue_size);

            m_preprocessWorkerPool.init(m_preprocessQue.get(), param.preprocess_thread_num, param.batch_num, param.batch_num);
            m_preprocessWorkerPool.startWork([this, &param](std::vector<T1> &items) {
                std::vector<T2> frames;
                m_detect_delegate->preprocess(items, frames);
                this->m_forwardQue->push(frames);
            });

            m_forwardWorkerPool.init(m_forwardQue.get(), param.inference_thread_num, 1, 8);
            m_forwardWorkerPool.startWork([this, &param](std::vector<T2> &items) {
                m_detect_delegate->forward(items);
                this->m_postprocessQue->push(items);
            });

            m_postprocessWorkerPool.init(m_postprocessQue.get(), param.postprocess_thread_num, 1, 8);
            m_postprocessWorkerPool.startWork([this, &param](std::vector<T2> &items) {
                m_detect_delegate->postprocess(items);
            });
            return 0;
        }

        int flush_frame() {
            m_preprocessWorkerPool.flush();
            return 0;
        }

        int push_frame(T1 *frame) {
            m_preprocessQue->push(*frame);
            return 0;
        }
    };

    // for one thread mode


    template<typename T1, typename T2>
    class BMInferenceSimple {
        std::shared_ptr<DetectorDelegate<T1, T2>> m_detect_delegate;

        std::shared_ptr<BlockingQueue<T1>> m_preprocessQue;
        WorkerPool<T1> m_preprocessWorkerPool;

        std::shared_ptr<BlockingQueue<T2>> m_postprocessQue;
        WorkerPool<T2> m_postprocessWorkerPool;

    public:
        BMInferenceSimple() {

        }

        virtual ~BMInferenceSimple() {

        }

        int init(int blocking, int preprocess_queue_size, int postprocess_queue_size,
                int batch_size, std::shared_ptr<DetectorDelegate<T1, T2>> delegate) {
            m_detect_delegate = delegate;

            const int underlying_type_std_queue = 0;
            m_preprocessQue = std::make_shared<BlockingQueue<T1>>(
                    "preprocess", underlying_type_std_queue,
                    blocking ? preprocess_queue_size : 0);

            m_preprocessWorkerPool.init(m_preprocessQue.get(), 1, batch_size,
                                        batch_size);
            m_preprocessWorkerPool.startWork([this, blocking, preprocess_queue_size](std::vector<T1> &items) {
                if (!blocking &&
                    m_preprocessQue->size() > preprocess_queue_size) {
                    std::cout << "WARNING:preprocess queue_size(" << m_preprocessQue->size() << ") > "
                              << preprocess_queue_size << std::endl;
                }

                std::vector<T2> frames;
                m_detect_delegate->preprocess(items, frames);
                m_detect_delegate->forward(frames);
                this->m_postprocessQue->push(frames);
            });

            //post process
            m_postprocessQue = std::make_shared<BlockingQueue<T2>>(
                    "postprocess", underlying_type_std_queue,
                    blocking ? postprocess_queue_size : 0);

            m_postprocessWorkerPool.init(m_postprocessQue.get(), 1, 1, 8);
            m_postprocessWorkerPool.startWork([this, blocking, postprocess_queue_size](std::vector<T2> &items) {
                if (!blocking &&
                    m_postprocessQue->size() > postprocess_queue_size) {
                    std::cout << "WARNING:preprocess queue_size(" << m_postprocessQue->size() << ") > "
                              << postprocess_queue_size << std::endl;
                }

                m_detect_delegate->postprocess(items);
            });

            return 0;
        }

        int flush_frame() {
            m_preprocessWorkerPool.flush();
            return 0;
        }

        int push_frame(T1 *frame) {
            m_preprocessQue->push(*frame);
            return 0;
        }
    };
} // end namespace bm


#endif //INFERENCE_FRAMEWORK_INFERENCE_H
