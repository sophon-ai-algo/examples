//
// Created by yuan on 3/4/21.
//

#ifndef INFERENCE_FRAMEWORK_WORKER_H
#define INFERENCE_FRAMEWORK_WORKER_H
#include "bmutility.h"
#include "bmgui.h"
#include "inference.h"
#include "stream_pusher.h"
#include "configuration.h"
#include "face_detector.h"
#include "bm_tracker.h"
#include "common_types.h"
#include "ddr_reduction.h"

struct TChannel: public bm::NoCopyable {
    int channel_id;
    uint64_t seq;
    bm::StreamDemuxer *demuxer;
    std::shared_ptr<bm::BMTracker> tracker;
    uint64_t m_last_feature_time=0; // last do feature time
    std::shared_ptr<DDRReduction> m_ddrr;
    int64_t ref_pkt_id = -1;
    AVCodecContext* m_decoder=nullptr;

    TChannel():channel_id(0), seq(0), demuxer(nullptr) {

         tracker = bm::BMTracker::create();
         m_last_feature_time = 0;
    }

    ~TChannel() {
        if (demuxer) delete demuxer;
        if (m_decoder) {
            avcodec_close(m_decoder);
            avcodec_free_context(&m_decoder);
        }
        std::cout << "TChannel(chan_id=" << channel_id << ") dtor" <<std::endl;
    }

    int create_video_decoder(int dev_id, AVFormatContext *ifmt_ctx) {
        int video_index = 0;

#if LIBAVCODEC_VERSION_MAJOR > 56
        auto codec_id = ifmt_ctx->streams[video_index]->codecpar->codec_id;
#else
        auto codec_id = ifmt_ctx->streams[video_index]->codec->codec_id;
#endif

        AVCodec *pCodec = avcodec_find_decoder(codec_id);
        if (NULL == pCodec) {
            printf("can't find code_id %d\n", codec_id);
            return -1;
        }

        m_decoder = avcodec_alloc_context3(pCodec);
        if (m_decoder == NULL) {
            printf("avcodec_alloc_context3 err");
            return -1;
        }

        int ret = 0;

#if LIBAVCODEC_VERSION_MAJOR > 56
        if ((ret = avcodec_parameters_to_context(m_decoder, ifmt_ctx->streams[video_index]->codecpar)) < 0) {
            fprintf(stderr, "Failed to copy %s codec parameters to decoder context\n",
                    av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
            return ret;
        }
#else
        /* Copy codec parameters from input stream to output codec context */
        if ((ret = avcodec_copy_context(m_dec_ctx, ifmt_ctx->streams[video_index]->codec)) < 0) {
            fprintf(stderr, "Failed to copy %s codec parameters to decoder context\n",
                    av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
            return ret;
        }
#endif

        if (pCodec->capabilities & AV_CODEC_CAP_TRUNCATED) {
            m_decoder->flags |= AV_CODEC_FLAG_TRUNCATED; /* we do not send complete frames */
        }

        //for PCIE
        AVDictionary* opts = NULL;
        av_dict_set_int(&opts, "sophon_idx", dev_id, 0x0);
        av_dict_set(&opts, "output_format", "101", 18);
        if (avcodec_open2(m_decoder, pCodec, &opts) < 0) {
            std::cout << "Unable to open codec";
            return -1;
        }

        return 0;
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
            *got_picture += 1;
            break;
        }

        if (*got_picture > 1) {
            printf("got picture %d\n", *got_picture);
        }

        return ret;
    }
};

using TChannelPtr = std::shared_ptr<TChannel>;

class OneCardInferApp {
    bm::VideoUIAppPtr m_guiReceiver;
    AppStatis &m_appStatis;
    std::shared_ptr<bm::DetectorDelegate<bm::cvs10FrameBaseInfo, bm::cvs10FrameInfo>> m_detectorDelegate;
    std::shared_ptr<bm::DetectorDelegate<bm::FeatureFrame, bm::FeatureFrameInfo>> m_featureDelegate;
    bm::BMNNContextPtr m_bmctx;
    bm::TimerQueuePtr m_timeQueue;
    int m_channel_start;
    int m_channel_num;
    int m_dev_id;
    int m_skipN;
    std::string m_output_url;
    int m_feature_delay;
    int m_feature_num;
    int m_use_l2_ddrr= 0;

    bm::BMInferencePipe<bm::cvs10FrameBaseInfo, bm::cvs10FrameInfo> m_inferPipe;
    bm::BMInferencePipe<bm::FeatureFrame, bm::FeatureFrameInfo> m_featurePipe;

    std::map<int, TChannelPtr> m_chans;
    std::vector<std::string> m_urls;
public:
    OneCardInferApp(AppStatis& statis,bm::VideoUIAppPtr gui, bm::TimerQueuePtr tq, bm::BMNNContextPtr ctx,
            int start_index, int num, int skip=0, int feat_delay=1000, int feat_num=8,
            int use_l2_ddrr=0): m_detectorDelegate(nullptr), m_channel_num(num),
            m_bmctx(ctx), m_appStatis(statis),m_use_l2_ddrr(use_l2_ddrr)
    {
        m_guiReceiver = gui;
        m_dev_id = m_bmctx->dev_id();
        m_timeQueue = tq;
        m_channel_start = start_index;
        m_skipN = skip;
        m_feature_delay = feat_delay;
        m_feature_num = feat_num;

    }

    ~OneCardInferApp()
    {
        std::cout << cv::format("OneCardInfoApp (devid=%d) dtor", m_dev_id) <<std::endl;
    }

    void setDetectorDelegate(std::shared_ptr<bm::DetectorDelegate<bm::cvs10FrameBaseInfo, bm::cvs10FrameInfo>> delegate){
        m_detectorDelegate = delegate;
    }

    void setFeatureDelegate(std::shared_ptr<bm::DetectorDelegate<bm::FeatureFrame, bm::FeatureFrameInfo>> delegate){
        m_featureDelegate = delegate;
    }

    void start(const std::vector<std::string>& vct_urls, Config& config);

    inline void loadConfig(bm::DetectorParam& param, Config& config) {
        SConcurrencyConfig cfg;
        if (config.get_phrase_config("preprocess", cfg)){
            param.preprocess_thread_num    = cfg.thread_num;
            param.preprocess_queue_size    = cfg.queue_size;
        }
        if (config.get_phrase_config("inference", cfg)){
            param.inference_thread_num    = cfg.thread_num;
            param.inference_queue_size    = cfg.queue_size;
        }
        if (config.get_phrase_config("postprocess", cfg)){
            param.postprocess_thread_num    = cfg.thread_num;
            param.postprocess_queue_size    = cfg.queue_size;
        }
    }
};

using OneCardInferAppPtr = std::shared_ptr<OneCardInferApp>;


#endif //INFERENCE_FRAMEWORK_MAIN_H
