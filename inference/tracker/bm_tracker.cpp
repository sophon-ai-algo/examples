//
// Created by yuan on 11/24/21.
//

#include "bm_tracker.h"
#include "KalmanFilter/tracker.h"
#include <mutex>

namespace bm {
    class BMTrackerImpl : public BMTracker {
        tracker *m_tracker;
        std::mutex m_tracker_sync;
    public:
        BMTrackerImpl(float max_cosine_distance=0.2, int nn_budget=100):m_tracker(nullptr) {
            m_tracker = new tracker(max_cosine_distance, nn_budget);
        }

        virtual ~BMTrackerImpl() {
            delete m_tracker;
        }


        virtual void update(const bm::NetOutputObjects &rects, bm::NetOutputObjects &results) {
            std::lock_guard<std::mutex> lck(m_tracker_sync);
            DETECTIONS detections;
            for(auto rc : rects) {
                DETECTION_ROW row;
                row.tlwh = DETECTBOX(rc.x1, rc.y1, rc.width(), rc.height());
                row.feature.setZero();
                detections.push_back(row);
            }

            m_tracker->predict();
            m_tracker->update(detections);
            for(Track &track : m_tracker->tracks) {
                if(!track.is_confirmed() || track.time_since_update > 1) continue;

                auto tmpbox = track.to_tlwh();
                bm::NetOutputObject dst_rc(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
                dst_rc.track_id = track.track_id;
                results.push_back(dst_rc);
            }
        }

    };

    std::shared_ptr<BMTracker> BMTracker::create(float max_cosine_distance, int nn_budget)
    {
        return std::make_shared<BMTrackerImpl>(max_cosine_distance, nn_budget);
    }
}