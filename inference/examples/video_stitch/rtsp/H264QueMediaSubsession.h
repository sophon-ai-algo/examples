//by yingyemin
#ifndef _BT_H264QueMediaSubsession_H
#define _BT_H264QueMediaSubsession_H

#include "liveMedia.hh"
#include "BasicUsageEnvironment.hh"
#include "GroupsockHelper.hh"

#include "OnDemandServerMediaSubsession.hh"
#include "H264FrameSource.h"

class H264QueMediaSubsession : public OnDemandServerMediaSubsession,
                               public H264FrameSourceListener
{
public:
  static H264QueMediaSubsession * createNew(UsageEnvironment& env, const char* fileName, bool reuseFirstSource = false);

private:
  H264QueMediaSubsession(UsageEnvironment& env, const char* fileName, bool reuseFirstSource = false);
  ~H264QueMediaSubsession(void);

  virtual FramedSource * createNewStreamSource(unsigned clientSessionId, unsigned & estBitrate); // "estBitrate" is the stream's estimated bitrate, in kbps
  virtual RTPSink * createNewRTPSink(Groupsock * rtpGroupsock, unsigned char rtpPayloadTypeIfDynamic, FramedSource * inputSource);

public:
  void inputH264Packet(PktData& pkt);
  void flushH264Packet();

private:
  char fFileName[100];
  H264FrameSource* _source;
};

#endif//_BT_H264QueMediaSubsession_H
