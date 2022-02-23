// Copyright (c) 1996-2018, Live Networks, Inc.  All rights reserved
// LIVE555 Media Server
// main program

#include <BasicUsageEnvironment.hh>
#include "version.hh"
#include <pthread.h>
#include <assert.h>
#include "Live555RtspServer.h"

static BTRTSPServer* rtspServer = nullptr;

void *threadCreateRtspServer(void *arg)
{    
  // Begin by setting up our usage environment:
  TaskScheduler* scheduler = BasicTaskScheduler::createNew();
  UsageEnvironment* env = BasicUsageEnvironment::createNew(*scheduler);

  UserAuthenticationDatabase* authDB = NULL;

  // Create the RTSP server.  Try first with the default port number (554),
  // and then with the alternative port number (8554):
  if (rtspServer != nullptr) {
    *env << "RTSP server already exit\n";
    exit(1);
  }
  portNumBits rtspServerPortNum = 554;
  rtspServer = BTRTSPServer::createNew(*env, rtspServerPortNum, authDB);
  int i = 0;
  while ((rtspServer == NULL) && (i < 100)) {
    rtspServerPortNum = 1554 + i++*10;
    rtspServer = BTRTSPServer::createNew(*env, rtspServerPortNum, authDB);
  }
  if (rtspServer == NULL) {
    *env << "Failed to create RTSP server: " << env->getResultMsg() << "\n";
    exit(1);
  }

  *env << "LIVE555 Media Server\n";
  *env << "\tversion " << MEDIA_SERVER_VERSION_STRING
       << " (LIVE555 Streaming Media library version "
       << LIVEMEDIA_LIBRARY_VERSION_STRING << ").\n";

  char* urlPrefix = rtspServer->rtspURLPrefix();
  *env << "Play streams from this server using the URL\n\t"
       << urlPrefix << "bitmain\n";


  env->taskScheduler().doEventLoop(); // does not return
  return NULL;
}


int CreateRtspServer() {

    pthread_t threadid;
    if(pthread_create(&threadid,NULL,threadCreateRtspServer,NULL) != 0)
    {
      //printf("%s错误出现在第%s行",__FUNCTION__,__LINE__);
      return -1;;
    }
    else
    {
      sleep(1);    //挂起1秒等待线程运行
    }
    return 0;
}

BTRTSPServer* GetRTSPInstance() {
  assert(rtspServer != nullptr);
  return rtspServer;
}