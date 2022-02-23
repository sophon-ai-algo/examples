#include <stdio.h>
#include "H264FrameSource.h"

unsigned int ff_gb28181_gettickcurmill(void) {
    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC,&tp);
    return tp.tv_sec*1000 + tp.tv_nsec/1000/1000;
}


H264FrameSource* H264FrameSource::createNew(UsageEnvironment& env, FramedSource* inputSource, const char* fileName, H264FrameSourceListener* listener)
{
  return new H264FrameSource(env, inputSource, listener);
}



H264FrameSource::H264FrameSource(UsageEnvironment& env, FramedSource* inputSource, H264FrameSourceListener* listener)
  : H264VideoStreamFramer(env, inputSource, false, false)
  , m_listener(listener)
{
    m_packetQueue = std::make_shared<BlockingQueue<PktData>>("live555rtsp", 0, 0);
}


H264FrameSource::~H264FrameSource()
{
    if (m_listener != nullptr) {
        m_listener->setSourceState(false);
    }
    std::cerr << "H264FrameSource deconstruct" << std::endl;
}


unsigned int H264FrameSource::maxFrameSize() const  
{
  return 1500000;
}

void H264FrameSource::doGetNextFrame()
{
    std::vector<PktData> items;
    if (m_packetQueue->pop_front(items, 1, 1) == 0) {
        fFrameSize = 0;
        assert(items.size() == 1);
        PktData pktData = items[0];
        fFrameSize = pktData->len;
        memcpy(fTo, pktData->buf, pktData->len);
        gettimeofday(&fPresentationTime, NULL);
        fDurationInMicroseconds = 1000 / 8 * 1000;
    } else {
        std::cerr << "Got 00000 from encoded queue" << std::endl;
    }
    nextTask() = envir().taskScheduler().scheduleDelayedTask(0,
          (TaskFunc*)FramedSource::afterGetting, this); 
    return;
}

void H264FrameSource::pushEncodedPacket(PktData &pkt) {
    m_packetQueue->push(pkt);
}

void H264FrameSource::dropAllFrames() {
    // drop all
    m_packetQueue->drop();
}
