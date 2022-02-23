//
// Created by yuan on 12/29/21.
//

#ifndef DDR_REDUCTION_DEMO_DDR_REDUCTION_H
#define DDR_REDUCTION_DEMO_DDR_REDUCTION_H

#ifdef __cplusplus
extern "C" {
#include "libavcodec/avcodec.h"
#include "libavutil/avutil.h"
#include "libavformat/avformat.h"
#include "libavutil/time.h"
}
#endif //!cplusplus

#include <iostream>
#include <memory>
#include <functional>

struct DDRReductionStat {
    int queue_packet_num;
    int total_recv_packet_num;   // average in 5 seconds
    int total_decode_frame_num;  // average fps in 5 seconds
    int total_buffer_packet_bytes;
};

class DDRReduction {
public:

    static std::shared_ptr<DDRReduction> create(int dev_id, AVCodecID codecId);

    virtual ~DDRReduction() {}

    virtual int put_packet(AVPacket *pkt, int64_t *p_id) = 0;

    virtual int put_packet(AVPacket *pkt, std::function<void(int64_t, AVFrame*)> cb) = 0;

    virtual int seek_frame(int64_t reference_id, AVFrame *frame, int *got_frame, int64_t *p_id) = 0;

    virtual int free_packet(int64_t id) = 0;

    virtual int flush() = 0;

    virtual int get_stat(DDRReductionStat *stat) = 0;
};




#endif //DDR_REDUCTION_DEMO_DDR_REDUCTION_H
