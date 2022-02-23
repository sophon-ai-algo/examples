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
// Header file

#ifndef _BT_RTSP_SERVER_HH
#define _BT_RTSP_SERVER_HH
#include "liveMedia.hh"
#include "H264QueMediaSubsession.h"


class BTRTSPServer: public RTSPServer {
public:
  static BTRTSPServer* createNew(UsageEnvironment& env, Port ourPort,
              UserAuthenticationDatabase* authDatabase,
              unsigned reclamationTestSeconds = 65);

protected:
  BTRTSPServer(UsageEnvironment& env, int ourSocket, Port ourPort,
        UserAuthenticationDatabase* authDatabase, unsigned reclamationTestSeconds);
  // called only by createNew();
  virtual ~BTRTSPServer();

protected: // redefined virtual functions
  virtual ServerMediaSession*
  lookupServerMediaSession(char const* streamName, Boolean isFirstLookupInSession);
public:
  void inputData(PktData& pkt);
  void flushData();
private:
    H264QueMediaSubsession* GetH264SubSession();
};

#endif
