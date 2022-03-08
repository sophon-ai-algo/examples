/**********
This library is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version. (See <http://www.gnu.org/copyleft/lesser.html>.)

This library is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
more details.

You should have received a copy of the GNU Lesser General Public License
along with this library; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
**********/
// Copyright (c) 1996-2018, Live Networks, Inc.  All rights reserved
// A subclass of "RTSPServer" that creates "ServerMediaSession"s on demand,
// based on whether or not the specified stream name exists as a file
// Implementation

#include <string.h>
#include <liveMedia.hh>
#include "BTRTSPServer.h"
#include "H264QueMediaSubsession.h"

BTRTSPServer*
BTRTSPServer::createNew(UsageEnvironment& env, Port ourPort,
           UserAuthenticationDatabase* authDatabase,
           unsigned reclamationTestSeconds) {
  int ourSocket = setUpOurSocket(env, ourPort);
  if (ourSocket == -1) return NULL;

  return new BTRTSPServer(env, ourSocket, ourPort, authDatabase, reclamationTestSeconds);
}

BTRTSPServer::BTRTSPServer(UsageEnvironment& env, int ourSocket,
             Port ourPort,
             UserAuthenticationDatabase* authDatabase, unsigned reclamationTestSeconds)
  : RTSPServer(env, ourSocket, ourPort, authDatabase, reclamationTestSeconds) {
}


BTRTSPServer::~BTRTSPServer() {
}


ServerMediaSession* BTRTSPServer
::lookupServerMediaSession(char const* streamName, Boolean isFirstLookupInSession) {
  // First, check whether the specified "streamName" exists as a local file:
//  FILE* fid = fopen(streamName, "rb");
//  Boolean fileExists = fid != NULL;
  //printf("lookupServerMediaSession ****** streamName=%s\n",streamName);
  // Next, check whether we already have a "ServerMediaSession" for this file:
  if (strcmp(streamName, "bitmain") != 0) {
    return nullptr;
  }
  ServerMediaSession* sms = RTSPServer::lookupServerMediaSession(streamName);
  Boolean smsExists = sms != NULL;


  if (sms == NULL) {
    sms = ServerMediaSession::createNew(envir(), streamName, streamName, "desc");
    OutPacketBuffer::maxSize = 15*1000*1000;
    sms->addSubsession(H264QueMediaSubsession::createNew(envir(), streamName, true));
    addServerMediaSession(sms);
  }

    //fclose(fid);
    return sms;
}

void BTRTSPServer::inputData(PktData& pkt) {
    H264QueMediaSubsession* subSession = GetH264SubSession();
    if (subSession != nullptr) {
        subSession->inputH264Packet(pkt);
    }
}

H264QueMediaSubsession* BTRTSPServer::GetH264SubSession() {
    ServerMediaSession* sms = RTSPServer::lookupServerMediaSession("bitmain");
    if (sms != nullptr) {
        ServerMediaSubsessionIterator sessionIter = ServerMediaSubsessionIterator(*sms);
        ServerMediaSubsession*        subSession  = sessionIter.next();
        if (subSession != nullptr) {
            return reinterpret_cast<H264QueMediaSubsession*>(subSession);
        }
    }
    return nullptr;
}

void BTRTSPServer::flushData() {
    H264QueMediaSubsession* subSession = GetH264SubSession();
    if (subSession != nullptr) {
        subSession->flushH264Packet();
    }
}
