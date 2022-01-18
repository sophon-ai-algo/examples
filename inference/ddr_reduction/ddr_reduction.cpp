//
// Created by yuan on 12/29/21.
//

#include "ddr_reduction.h"
#include <iostream>
#include <deque>
#include <map>
#include <algorithm>
#include <assert.h>
#include <bmutility/bmutility_timer.h>
#include "bmutility_list.h"

struct NoCopyable {
    NoCopyable() = default;
    ~NoCopyable() = default;
    NoCopyable(const NoCopyable& o) = delete;
    NoCopyable& operator=(const NoCopyable& o) = delete;
};

struct PacketItem: public NoCopyable {
    PacketItem(int dev_id): id(dev_id){
        av_init_packet(&pkt);
        INIT_LIST_HEAD(&entry);
    }

    ~PacketItem() {
        av_packet_unref(&pkt);
    }

    ListHead entry;
    AVPacket pkt;
    int64_t id;
};


class DDRReductionImpl : public DDRReduction {
    ListHead m_list_packets;
    std::map<int64_t, std::shared_ptr<PacketItem>> m_map_packets;
    int64_t m_duration_msec;
    int m_dev_id;
    AVCodecContext* m_decoder;
    int64_t m_uid;
    int64_t m_last_seek_ref_id;

    std::mutex m_list_lock;

    //statistic
    int64_t m_total_recv_packet_num{0};
    int64_t m_total_decode_frame_num{0};
    int64_t m_total_packet_bytes{0};

private:
    AVCodecContext* create_decoder(int dev_id, AVCodecID codec_id) {
        AVCodec *pCodec = avcodec_find_decoder(codec_id);
        if (NULL == pCodec) {
            printf("can't find code_id %d\n", codec_id);
            return nullptr;
        }

        AVCodecContext* dec_ctx = avcodec_alloc_context3(pCodec);
        if (dec_ctx == NULL) {
            printf("avcodec_alloc_context3 err");
            return nullptr;
        }

        if (pCodec->capabilities & AV_CODEC_CAP_TRUNCATED) {
            dec_ctx->flags |= AV_CODEC_FLAG_TRUNCATED;
        }

        dec_ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;

        dec_ctx->workaround_bugs = FF_BUG_AUTODETECT;
        dec_ctx->err_recognition = AV_EF_CAREFUL;
        dec_ctx->error_concealment = FF_EC_GUESS_MVS | FF_EC_DEBLOCK;

        AVDictionary *opts = NULL;
        av_dict_set_int(&opts, "sophon_idx", dev_id, 0x0);

        // compressed format to reduce memory use.
        av_dict_set_int(&opts, "output_format", 101, 0);

        if (avcodec_open2(dec_ctx, pCodec, &opts) < 0) {
            std::cout << "Unable to open codec";
            avcodec_free_context(&dec_ctx);
            return nullptr;
        }

        return dec_ctx;
    }

public:
    DDRReductionImpl(int dev_id, AVCodecID codecId, int duration = 2000):m_duration_msec(duration),
    m_dev_id(dev_id),m_uid(1),m_last_seek_ref_id(-1) {
          m_decoder = create_decoder(dev_id, codecId);
          INIT_LIST_HEAD(&m_list_packets);

    }

    ~DDRReductionImpl() {
        // free all packet in the list.
        flush();

        // free decoder
        avcodec_close(m_decoder);
        avcodec_free_context(&m_decoder);


    }

    int decode_video2(AVCodecContext* dec_ctx, AVFrame *frame, int *got_picture, AVPacket* pkt)
    {
        int ret;
        *got_picture = 0;
        ret = avcodec_send_packet(dec_ctx, pkt);
        if (ret == AVERROR_EOF) {
            ret = 0;
        }
        else if (ret < 0) {
            fprintf(stderr, "Error sending a packet for decoding, %s\n", av_err2str(ret));
            return -1;
        }

        while (ret >= 0) {
            ret = avcodec_receive_frame(dec_ctx, frame);
            if (ret == AVERROR(EAGAIN)) {
                printf("need more data!\n");
                ret = 0;
                break;
            }else if (ret == AVERROR_EOF) {
                printf("File end!\n");
                avcodec_flush_buffers(dec_ctx);
                ret = 0;
                break;
            }
            else if (ret < 0) {
                fprintf(stderr, "Error during decoding\n");
                break;
            }
            //printf("saving frame %3d\n", dec_ctx->frame_number);
            *got_picture += 1;
            break;
        }

        if (*got_picture > 1) {
            printf("got picture %d\n", *got_picture);
        }

        return ret;
    }

    int put_packet(AVPacket* pkt, int64_t *p_id) override
    {
        std::shared_ptr<PacketItem> pkt_item = std::make_shared<PacketItem>(m_dev_id);

        av_packet_ref(&pkt_item->pkt, pkt);
        pkt_item->id = m_uid++;

        // add to list
        m_list_lock.lock();
        while (m_map_packets.size() > 8) {
            ListHead *pos;
            pos = list_back(&m_list_packets);
            PacketItem *item = LIST_HOST_ENTRY(pos, PacketItem, entry);
            m_total_packet_bytes-=item->pkt.size;
            list_del(&item->entry);
            m_map_packets.erase(item->id);
        }

        list_push_front(&pkt_item->entry, &m_list_packets);
        m_map_packets[pkt_item->id] = pkt_item;
        m_list_lock.unlock();

        if (p_id) *p_id = pkt_item->id;
        
        // update statistic
        m_total_recv_packet_num++;
        m_total_packet_bytes += pkt->size;

        return 0;
    }

    int put_packet(AVPacket *pkt, std::function<void(int64_t, AVFrame*)> cb) override
    {
        std::shared_ptr<PacketItem> pkt_item = std::make_shared<PacketItem>(m_dev_id);

        av_packet_ref(&pkt_item->pkt, pkt);
        pkt_item->id = m_uid++;

        // add to list
        m_list_lock.lock();
        while (m_map_packets.size() > 8) {
            ListHead *pos;
            pos = list_back(&m_list_packets);
            PacketItem *item = LIST_HOST_ENTRY(pos, PacketItem, entry);
            m_total_packet_bytes-=item->pkt.size;
            list_del(&item->entry);
            m_map_packets.erase(item->id);
        }

        list_push_front(&pkt_item->entry, &m_list_packets);
        m_map_packets[pkt_item->id] = pkt_item;
        m_list_lock.unlock();

        if (cb != nullptr) {
            int got_frame = 0;
            AVFrame* frame = av_frame_alloc();
            decode_video2(m_decoder, frame, &got_frame, pkt);
            if (got_frame) {
                cb(pkt_item->id, frame);
            }

            av_frame_free(&frame);
        }

        // update statistic
        m_total_recv_packet_num++;
        m_total_packet_bytes += pkt->size;
    }

    int seek_frame(int64_t ref_id, AVFrame *frame, int *got_frame, int64_t *p_id) override {
        int ret = 0;
        int64_t next_id = 0;
        std::lock_guard<std::mutex> lg(m_list_lock);
        // Random seek frame
        std::shared_ptr<PacketItem> item;
        auto find_item = m_map_packets.find(ref_id);
        if (find_item != m_map_packets.end()) {
            ListHead *pos = list_next(&find_item->second->entry);
            if (NULL == pos) return -1;
            PacketItem *pkt_item = LIST_HOST_ENTRY(pos, PacketItem, entry);
            next_id = pkt_item->id;
        }else{
            PacketItem *pkt_item;
            bool ok = false;
            int loop_times=0;
            list_for_each_entry_next(pkt_item, &m_list_packets, PacketItem, entry) {
                loop_times++;
                if (pkt_item->id > ref_id) {
                    next_id = pkt_item->id;
                    ok = true;
                    break;
                }
            }

            if (!ok) {
                std::cout << "can't find next packet id = " << ref_id << std::endl;
                return -1;
            }
        }

        if (p_id) *p_id = next_id;
        item = m_map_packets[next_id];

        // Search nearest previous key frame
        // Debug information
        int debug_prefore_search_times = 0;
        int debug_after_decode_times = 0;
        auto item_iterate = item.get();
        while(item_iterate!= NULL) {
            if (item_iterate->pkt.flags & AV_PKT_FLAG_KEY) {
                break;
            }
            debug_prefore_search_times ++;
            if (item_iterate->entry.prev != &m_list_packets) {
                item_iterate = LIST_HOST_ENTRY(item_iterate->entry.prev, PacketItem, entry);
                if (item_iterate->id == m_last_seek_ref_id) {
                    item_iterate = LIST_HOST_ENTRY(item_iterate->entry.next, PacketItem, entry);
                    break;
                }
            }else{
                item_iterate = NULL;
            }
        }

        while(item_iterate != NULL) {
            //
            // decode from key pkt to pkt with ref_id
            //
            ret = decode_video2(m_decoder, frame, got_frame, &item_iterate->pkt);
            if (ret < 0) {
                std::cout << "avcodec_decode_video2() err=" << ret << std::endl;
                assert(0);
            }

            debug_after_decode_times++;
            if (*got_frame > 0) {
                m_total_decode_frame_num++;
            }

            // end to ref_id
            if (item_iterate->id >= next_id) {
                m_last_seek_ref_id = next_id;
                break;
            }

            ListHead *pos = list_next(&item_iterate->entry);
            if (pos != &m_list_packets) {
                item_iterate = LIST_HOST_ENTRY(pos, PacketItem, entry);
            }else{
                item_iterate = NULL;
            }

            av_frame_unref(frame);
        }

        //std::printf("before search [%d], after decode[%d]\n", debug_prefore_search_times, debug_after_decode_times);

        return ret >=0 ? 0: -1;
    }

    int free_packet(int64_t id) override{
        std::lock_guard<std::mutex> lg(m_list_lock);
        auto find_item = m_map_packets.find(id);
        if (find_item == m_map_packets.end()) {
             return 0;
        }

        std::shared_ptr<PacketItem> pkt_item = find_item->second;
        m_total_packet_bytes-=pkt_item->pkt.size;
        list_del(&pkt_item->entry);
        m_map_packets.erase(find_item);
        return 0;
    }

    virtual int flush() override{
        std::lock_guard<std::mutex> lg(m_list_lock);
        ListHead *pos, *n;
        PacketItem* item;
        list_for_each_safe(pos, n, &m_list_packets) {
            item = LIST_HOST_ENTRY(pos, PacketItem, entry);
            list_del(&item->entry);
        }
        m_map_packets.clear();

        return 0;
    }

    int get_stat(DDRReductionStat *stat) override {
        assert(NULL != stat);
        stat->queue_packet_num = m_map_packets.size();
        stat->total_decode_frame_num = m_total_decode_frame_num;
        stat->total_recv_packet_num = m_total_recv_packet_num;
        stat->total_buffer_packet_bytes = m_total_packet_bytes;
        return 0;
    }

};

std::shared_ptr<DDRReduction> DDRReduction::create(int dev_id, AVCodecID codecId)
{
    return std::make_shared<DDRReductionImpl>(dev_id, codecId);
}