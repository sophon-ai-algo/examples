//by yingyemin
#include "H264QueMediaSubsession.h"
#include "H264FrameSource.h"
#include "H264VideoStreamFramer.hh"
#include "H264VideoRTPSink.hh"


H264QueMediaSubsession* H264QueMediaSubsession::createNew(UsageEnvironment& env, const char* fileName, bool reuseFirstSource)
{
  H264QueMediaSubsession* sms = new H264QueMediaSubsession(env, fileName, reuseFirstSource);
  return sms;
}


H264QueMediaSubsession::H264QueMediaSubsession(UsageEnvironment& env, const char*fileName, bool reuseFirstSource)
  : OnDemandServerMediaSubsession(env, reuseFirstSource)
{

}

H264QueMediaSubsession::~H264QueMediaSubsession()
{
    std::cerr << "H264QueMediaSubsession deconstruct" << std::endl;
}


FramedSource* H264QueMediaSubsession::createNewStreamSource(unsigned clientsessionId, unsigned& estBitrate)
{
  estBitrate = 1000;//估计比特率
  
  _source = H264FrameSource::createNew(envir(), NULL, NULL, this);
  m_source_state = true;
  return _source;
}

void H264QueMediaSubsession::inputH264Packet(PktData& pkt) {
    if (m_source_state) {
        _source->pushEncodedPacket(pkt);
    }
}

void H264QueMediaSubsession::flushH264Packet() {
    if (m_source_state) {
        _source->dropAllFrames();
    }
}

RTPSink* H264QueMediaSubsession::createNewRTPSink(Groupsock* rtpGroupsock, unsigned char rtpPayloadTypeIfDynamic, FramedSource* inputSource)
{
  return H264VideoRTPSink::createNew(envir(), rtpGroupsock, 96);;
}

//char const* H264MemoryLiveVideoServerMediaSubsession::sdpLines()
//{
//    return fSDPLines = 
//            "v=0\r\n"
//            "s=H3C IPC Realtime stream\r\n"
//            "m=video 0 RTP/AVP 96\r\n"
//            "c=IN IP4 0.0.0.0\r\n"
//            "a=rtpmap:96 H264/90000\r\n"
//            "a=fmtp:96 DecoderTag=h3c-v3 RTCP=0\r\n"
//            "a=control:track1\r\n";
//}
//
