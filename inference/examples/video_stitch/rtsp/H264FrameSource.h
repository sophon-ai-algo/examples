#ifndef _BT_H264FrameSource_H
#define _BT_H264FrameSource_H

#include <stdio.h>
#include <map>
#include <string>
#include <pthread.h>

#include "liveMedia.hh"
#include "BasicUsageEnvironment.hh"
#include "GroupsockHelper.hh"
#include "FramedSource.hh"
#include "thread_queue.h"



struct _tagPktData {
    char *buf;
    int  len;

    _tagPktData()
     : buf{nullptr}, len{0} {}
     
    ~_tagPktData() {
        if (buf != nullptr && len != 0) {
            delete[] buf;
            buf = nullptr;
            len = 0;
        }
    }
};
using PktData = std::shared_ptr<_tagPktData>;

#define SIZE 1080*1920

class H264FrameSourceListener {
public:
    bool m_source_state{false};
public:
    virtual void setSourceState(bool state) {
        m_source_state = state;
    }
};

class H264FrameSource : public H264VideoStreamFramer
{
public:
    static H264FrameSource* createNew(UsageEnvironment& env, FramedSource* inputSource,const char* fileName, H264FrameSourceListener* listener);

private:
    H264FrameSource(UsageEnvironment& env, FramedSource* inputSource, H264FrameSourceListener* listener);
    ~H264FrameSource(void);

private:
    virtual void doGetNextFrame();
    virtual unsigned int maxFrameSize() const;


public:
    char fileBuf[SIZE];
    //int fsize;
    long long m_framenusm;
    std::shared_ptr<BlockingQueue<PktData>> m_packetQueue;
public:
    void dropAllFrames();
    void pushEncodedPacket(PktData &pkt);

private:
    H264FrameSourceListener* m_listener;
};

#endif
