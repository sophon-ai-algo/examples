#ifndef _LIVE555RTSPSERVER_H_
#define _LIVE555RTSPSERVER_H_

#include <pthread.h>
#include <queue>
#include "BTRTSPServer.h"




int CreateRtspServer();

BTRTSPServer* GetRTSPInstance();

#endif//_LIVE555RTSPSERVER_H_
