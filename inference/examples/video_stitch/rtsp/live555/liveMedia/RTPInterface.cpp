/**********
This library is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation; either version 2.1 of the License, or (at your
option) any later version. (See <http://www.gnu.org/copyleft/lesser.html>.)

This library is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
more details.

You should have received a copy of the GNU Lesser General Public License
along with this library; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
**********/
// "liveMedia"
// Copyright (c) 1996-2016 Live Networks, Inc.  All rights reserved.
// An abstraction of a network interface used for RTP (or RTCP).
// (This allows the RTP-over-TCP hack (RFC 2326, section 10.12) to
// be implemented transparently.)
// Implementation

#include "RTPInterface.hh"
#include <GroupsockHelper.hh>
#include <stdio.h>
#include <time.h>

////////// Helper Functions - Definition //////////

// Helper routines and data structures, used to implement
// sending/receiving RTP/RTCP over a TCP socket:

// Reading RTP-over-TCP is implemented using two levels of hash tables.
// The top-level hash table maps TCP socket numbers to a
// "SocketDescriptor" that contains a hash table for each of the
// sub-channels that are reading from this socket.
#include <map>

using namespace std;

FILE* g_mapFile[1024] = {NULL};
//FILE* g_senddataFile[1024] = {NULL};
unsigned int g_lastsendtick[1024] = {0};

static HashTable* socketHashTable(UsageEnvironment& env, Boolean createIfNotPresent = True) {
  _Tables* ourTables = _Tables::getOurTables(env, createIfNotPresent);
  if (ourTables == NULL) return NULL;

  if (ourTables->socketTable == NULL) {
    // Create a new socket number -> SocketDescriptor mapping table:
    ourTables->socketTable = HashTable::create(ONE_WORD_HASH_KEYS);
  }
  return (HashTable*)(ourTables->socketTable);
}

unsigned int gettickcurmill() {
  struct timespec tp;
  clock_gettime(CLOCK_MONOTONIC,&tp);
  return tp.tv_sec*1000 + tp.tv_nsec/1000/1000;
}


class SocketDescriptor {
public:
  SocketDescriptor(UsageEnvironment& env, int socketNum);
  virtual ~SocketDescriptor();

  void registerRTPInterface(unsigned char streamChannelId,
                RTPInterface* rtpInterface);
  RTPInterface* lookupRTPInterface(unsigned char streamChannelId);
  void deregisterRTPInterface(unsigned char streamChannelId);

  void setServerRequestAlternativeByteHandler(ServerRequestAlternativeByteHandler* handler, void* clientData) {
    fServerRequestAlternativeByteHandler = handler;
    fServerRequestAlternativeByteHandlerClientData = clientData;
  }

private:
  static void tcpReadHandler(SocketDescriptor*, int mask);
  Boolean tcpReadHandler1(int mask);

private:
  UsageEnvironment& fEnv;
  int fOurSocketNum;
  HashTable* fSubChannelHashTable;
  ServerRequestAlternativeByteHandler* fServerRequestAlternativeByteHandler;
  void* fServerRequestAlternativeByteHandlerClientData;
  u_int8_t fStreamChannelId, fSizeByte1;
  Boolean fReadErrorOccurred, fDeleteMyselfNext, fAreInReadHandlerLoop;
  enum { AWAITING_DOLLAR, AWAITING_STREAM_CHANNEL_ID, AWAITING_SIZE1, AWAITING_SIZE2, AWAITING_PACKET_DATA } fTCPReadingState;
};

static SocketDescriptor* lookupSocketDescriptor(UsageEnvironment& env, int sockNum, Boolean createIfNotFound = True) {
  HashTable* table = socketHashTable(env, createIfNotFound);
  if (table == NULL) return NULL;

  char const* key = (char const*)(long)sockNum;
  SocketDescriptor* socketDescriptor = (SocketDescriptor*)(table->Lookup(key));
  if (socketDescriptor == NULL) {
    if (createIfNotFound) {
      socketDescriptor = new SocketDescriptor(env, sockNum);
      table->Add((char const*)(long)(sockNum), socketDescriptor);
    } else if (table->IsEmpty()) {
      // We can also delete the table (to reclaim space):
      _Tables* ourTables = _Tables::getOurTables(env);
      delete table;
      ourTables->socketTable = NULL;
      ourTables->reclaimIfPossible();
    }
  }

  return socketDescriptor;
}

static void removeSocketDescription(UsageEnvironment& env, int sockNum) {
  char const* key = (char const*)(long)sockNum;
  HashTable* table = socketHashTable(env);
  table->Remove(key);

  if (table->IsEmpty()) {
    // We can also delete the table (to reclaim space):
    _Tables* ourTables = _Tables::getOurTables(env);
    delete table;
    ourTables->socketTable = NULL;
    ourTables->reclaimIfPossible();
  }
}


////////// RTPInterface - Implementation //////////

RTPInterface::RTPInterface(Medium* owner, Groupsock* gs)
  : fOwner(owner), fGS(gs),
    fTCPStreams(NULL),
    fNextTCPReadSize(0), fNextTCPReadStreamSocketNum(-1),
    fNextTCPReadStreamChannelId(0xFF), fReadHandlerProc(NULL),
    fAuxReadHandlerFunc(NULL), fAuxReadHandlerClientData(NULL) {
  // Make the socket non-blocking, even though it will be read from only asynchronously, when packets arrive.
  // The reason for this is that, in some OSs, reads on a blocking socket can (allegedly) sometimes block,
  // even if the socket was previously reported (e.g., by "select()") as having data available.
  // (This can supposedly happen if the UDP checksum fails, for example.)
  makeSocketNonBlocking(fGS->socketNum());
//      unsigned requestedSize = 1024000;
//      SOCKLEN_T sizeSize = sizeof requestedSize;
//      setsockopt(fInputSocketNum, SOL_SOCKET, SO_SNDBUF, (char*)&requestedSize, sizeSize);
//      setsockopt(fInputSocketNum, SOL_SOCKET, SO_RCVBUF, (char*)&requestedSize, sizeSize);



  increaseSendBufferTo(envir(), fGS->socketNum(), 1024*1024);
  for(int i =0;i<1024;i++) {
    g_mapFile[i] = NULL;
  }
}

RTPInterface::~RTPInterface() {
  stopNetworkReading();
  //printf("====%s,%d. fTCPStreams.\n",__FUNCTION__,__LINE__);
  if (fTCPStreams) {
    //printf("====%s,%d. delete fTCPStreams=[%p].\n",__FUNCTION__,__LINE__,fTCPStreams);
    delete fTCPStreams;
  }
}

void RTPInterface::setStreamSocket(int sockNum,
                   unsigned char streamChannelId) {
  //printf("====setStreamSocket   sockNum=[%d],streamChannelId[%c].",sockNum,streamChannelId);
  fGS->removeAllDestinations();
  envir().taskScheduler().disableBackgroundHandling(fGS->socketNum()); // turn off any reading on our datagram socket
  fGS->reset(); // and close our datagram socket, because we won't be using it anymore

  addStreamSocket(sockNum, streamChannelId);
}

void RTPInterface::addStreamSocket(int sockNum,
                   unsigned char streamChannelId) {
  if (sockNum < 0) return;
  //printf("====addStreamSocket   sockNum=[%d],streamChannelId[%c].",sockNum,streamChannelId);

  for (tcpStreamRecord* streams = fTCPStreams; streams != NULL;
       streams = streams->fNext) {
    if (streams->fStreamSocketNum == sockNum
    && streams->fStreamChannelId == streamChannelId) {
      return; // we already have it
    }
  }

  //printf("======last fTCPStreams=[%p].\n",fTCPStreams);
  fTCPStreams = new tcpStreamRecord(sockNum, streamChannelId, fTCPStreams);
  //printf("======new fTCPStreams=[%p].\n",fTCPStreams);

  // Also, make sure this new socket is set up for receiving RTP/RTCP-over-TCP:
  SocketDescriptor* socketDescriptor = lookupSocketDescriptor(envir(), sockNum);
  socketDescriptor->registerRTPInterface(streamChannelId, this);
}

static void deregisterSocket(UsageEnvironment& env, int sockNum, unsigned char streamChannelId) {
  //printf("=====%s,%d.socknum=%d,streamChannelId=%d.\n",__FUNCTION__,__LINE__,sockNum,streamChannelId);
  SocketDescriptor* socketDescriptor = lookupSocketDescriptor(env, sockNum, False);
  if (socketDescriptor != NULL) {
    socketDescriptor->deregisterRTPInterface(streamChannelId);
        // Note: This may delete "socketDescriptor",
        // if no more interfaces are using this socket
  }
}

void RTPInterface::removeStreamSocket(int sockNum,
                      unsigned char streamChannelId) {
  //printf("====%s,%d.socknum=%d,streamChannelId=%d.\n",__FUNCTION__,__LINE__,sockNum,streamChannelId);
  while (1) {
    tcpStreamRecord** streamsPtr = &fTCPStreams;

    while (*streamsPtr != NULL) {
      if ((*streamsPtr)->fStreamSocketNum == sockNum
      && (streamChannelId == 0xFF || streamChannelId == (*streamsPtr)->fStreamChannelId)) {
    // Delete the record pointed to by *streamsPtr :
    tcpStreamRecord* next = (*streamsPtr)->fNext;
//printf("===next stream is %p\n", next);
//printf("====cur stream is %p,delete streamsPtr\n", *streamsPtr);
    (*streamsPtr)->fNext = NULL;
    delete (*streamsPtr);
    *streamsPtr = next;
//printf("====f tcp stream is %p\n", fTCPStreams);

    // And 'deregister' this socket,channelId pair:
    deregisterSocket(envir(), sockNum, streamChannelId);

    if (streamChannelId != 0xFF) return; // we're done
    break; // start again from the beginning of the list, in case the list has changed
      } else {
    streamsPtr = &((*streamsPtr)->fNext);
      }
    }
    if (*streamsPtr == NULL) break;
  }
}

void RTPInterface::setServerRequestAlternativeByteHandler(UsageEnvironment& env, int socketNum,
                              ServerRequestAlternativeByteHandler* handler, void* clientData) {
  SocketDescriptor* socketDescriptor = lookupSocketDescriptor(env, socketNum, False);

  if (socketDescriptor != NULL) socketDescriptor->setServerRequestAlternativeByteHandler(handler, clientData);
}

void RTPInterface::clearServerRequestAlternativeByteHandler(UsageEnvironment& env, int socketNum) {
  setServerRequestAlternativeByteHandler(env, socketNum, NULL, NULL);
}

//#define TEST

Boolean RTPInterface::sendPacket(unsigned char* packet, unsigned packetSize) {
        //printf("start send packet\n");
  Boolean success = True; // we'll return False instead if any of the sends fail
  unsigned int beg = gettickcurmill();
  // Normal case: Send as a UDP packet:
  if (!fGS->output(envir(), packet, packetSize)) success = False;

//  static unsigned int cccc111 = gettickcurmill();
//  unsigned int curtick111 = gettickcurmill();
//  if(curtick111 > cccc111+41) {
//    printf("send packet interval ====:::::=== 1000.%d.\n",curtick111-cccc111);
//  }
//  cccc111 = curtick111;

  // Also, send over each of our TCP sockets:
  tcpStreamRecord* nextStream;
  for (tcpStreamRecord* stream = fTCPStreams; stream != NULL; stream = nextStream) {
    nextStream = stream->fNext; // Set this now, in case the following deletes "stream":

//        printf("start for send packet\n");
#if 1
//        static unsigned int cccc = gettickcurmill();
    unsigned int curtick = gettickcurmill();
//        if(curtick > cccc+41) {
//      printf("send packet interval ====:::::=== 1000.%d.sock=%d.\n",curtick-cccc,stream->fStreamSocketNum);
//    }
//    cccc = curtick;
    //当前是一个新包，比较上次发送的时间，超过40ms  80ms 160ms 400ms的分别打印
//    if(curtick > (stream->lastSendNalTick +1000)) {
//      printf("send packet interval ============ 1000.%d.sock=%d.\n",curtick-stream->lastSendNalTick,stream->fStreamSocketNum);
//    }
//    else if(curtick > (stream->lastSendNalTick +400)) {
//      printf("send packet interval ============ 400.%d.sock=%d.\n",curtick-stream->lastSendNalTick,stream->fStreamSocketNum);
//    }
//    else if(curtick > (stream->lastSendNalTick +160)) {
//      printf("send packet interval ============ 160.%d.sock=%d.\n",curtick-stream->lastSendNalTick,stream->fStreamSocketNum);
//    }
//    else if(curtick > (stream->lastSendNalTick +80)) {
//      printf("send packet interval ============ 80.%d.sock=%d.\n",curtick-stream->lastSendNalTick,stream->fStreamSocketNum);
//    }
//    else if(curtick > (stream->lastSendNalTick +50)) {
//      printf("send packet interval ============ 50.%d.sock=%d.\n",curtick-stream->lastSendNalTick,stream->fStreamSocketNum);
//    }
    stream->lastSendNalTick = curtick;
        //printf("start test for send packet\n");
    u_int8_t framingHeader[4];
    framingHeader[0] = '$';
    framingHeader[1] = stream->fStreamChannelId;
    framingHeader[2] = (u_int8_t) ((packetSize&0xFF00)>>8);
    framingHeader[3] = (u_int8_t) (packetSize&0xFF);

    Boolean copy = False;
    int length = sizeof(framingHeader) + packetSize;
    if (length <= stream->Space())
    {
        copy = True;
        stream->PushDataBuf(framingHeader, sizeof(framingHeader));
        stream->PushDataBuf(packet, packetSize);
    }

    /*if (!sendRTPorRTCPPacketOverTCP(buf, size, stream->fStreamSocketNum, sendNum)) {
        success = False;
                printf("send packet failed===============\n");
    }
    else {

        stream->OffsetReadSize(sendNum);
        if (stream->IsEmpty()) {
            stream->Reset();
        }

        if (!copy && length < stream->Space()) {
            stream->PushDataBuf(framingHeader, sizeof(framingHeader));
            stream->PushDataBuf(packet, packetSize);
        }
    }*/
        //printf("start while  send packet\n");
    while(true) {
        unsigned char* buf = stream->DataBuf();
        int size = stream->DataLength();
        int sendNum = 0;

        //if(g_mapFile[stream->fStreamSocketNum] == NULL) {
        //    char file_name_[32] = {0};
        //    sprintf(file_name_,"%d.nal",stream->fStreamSocketNum);
        //    FILE *fp = fopen(file_name_,"wb+");
        //    g_mapFile[stream->fStreamSocketNum] = fp;
        //    printf("open file\n");
        //}
        //if(g_mapFile[stream->fStreamSocketNum] != NULL) {
        //    fwrite(buf,1,size,g_mapFile[stream->fStreamSocketNum]);
        //    printf("before send packet  sock=%d,sendNum=%d.\n",stream->fStreamSocketNum, sendNum);
        //}

        //static int dddd = 0;
        //char file__[32] = {0};
        //sprintf(file__,"rtsp_%d.nal",dddd++);
        //FILE *fpdd = fopen(file__,"wb+");
        //fwrite(buf,1,size,fpdd);
        //fclose(fpdd);
        //printf("pktsize[%d] %02x_%02x_%02x_%02x_%02x_%02x_%02x_%02x___%02x_%02x_%02x_%02x_%02x_%02x_%02x_%02x___%02x_%02x_%02x_%02x_%02x_%02x_%02x_%02x___%02x_%02x_%02x_%02x_%02x_%02x_%02x_%02x___%02x_%02x_%02x_%02x_%02x_%02x_%02x_%02x.\n"
                //      ,size
                //      ,buf[0],buf[1],buf[2],buf[3],buf[4],buf[5],buf[6],buf[7]
                //      ,buf[8],buf[9],buf[10],buf[11],buf[12],buf[13],buf[14],buf[15]
                //      ,buf[16],buf[17],buf[18],buf[19],buf[20],buf[21],buf[22],buf[23]
                //      ,buf[24],buf[25],buf[26],buf[27],buf[28],buf[29],buf[30],buf[31]
                //      ,buf[32],buf[33],buf[34],buf[35],buf[36],buf[37],buf[38],buf[39]);
        if (!sendRTPorRTCPPacketOverTCP(buf, size, stream->fStreamSocketNum, sendNum)) {
            success = False;
                    printf("send packet failed===============\n");
            break;
        } else {
            if (sendNum == 0) {
                printf("sendNum is 0\n");
                break;
            }
        //    printf("copy is ---> %d space is ---> %d\n", copy, stream->Space());
            stream->OffsetReadSize(sendNum);
            if (stream->IsEmpty()) {
                stream->Reset();
                break;
            }

        }
        //printf("after send packet\n");

    }
    if (!copy && length < stream->Space()) {
                stream->PushDataBuf(framingHeader, sizeof(framingHeader));
                stream->PushDataBuf(packet, packetSize);
        }

#else
//        printf("start else send packet\n");
    if (!sendRTPorRTCPPacketOverTCP(packet, packetSize,
                    stream->fStreamSocketNum, stream->fStreamChannelId)) {
      success = False;
    }
#endif
  }
  if(gettickcurmill() > (beg +20)) {
    //printf("consum is bigger than 20.%d.\n",gettickcurmill() - beg);
  }
  return success;
}

void RTPInterface
::startNetworkReading(TaskScheduler::BackgroundHandlerProc* handlerProc) {
  // Normal case: Arrange to read UDP packets:
  envir().taskScheduler().
    turnOnBackgroundReadHandling(fGS->socketNum(), handlerProc, fOwner);

  // Also, receive RTP over TCP, on each of our TCP connections:
  fReadHandlerProc = handlerProc;
  for (tcpStreamRecord* streams = fTCPStreams; streams != NULL;
       streams = streams->fNext) {
    // Get a socket descriptor for "streams->fStreamSocketNum":
    SocketDescriptor* socketDescriptor = lookupSocketDescriptor(envir(), streams->fStreamSocketNum);

    // Tell it about our subChannel:
    socketDescriptor->registerRTPInterface(streams->fStreamChannelId, this);
  }
}

Boolean RTPInterface::handleRead(unsigned char* buffer, unsigned bufferMaxSize,
                 unsigned& bytesRead, struct sockaddr_in& fromAddress,
                 int& tcpSocketNum, unsigned char& tcpStreamChannelId,
                 Boolean& packetReadWasIncomplete) {
  packetReadWasIncomplete = False; // by default
  Boolean readSuccess;
  if (fNextTCPReadStreamSocketNum < 0) {
    // Normal case: read from the (datagram) 'groupsock':
    tcpSocketNum = -1;
    readSuccess = fGS->handleRead(buffer, bufferMaxSize, bytesRead, fromAddress);
  } else {
    // Read from the TCP connection:
    tcpSocketNum = fNextTCPReadStreamSocketNum;
    tcpStreamChannelId = fNextTCPReadStreamChannelId;

    bytesRead = 0;
    unsigned totBytesToRead = fNextTCPReadSize;
    if (totBytesToRead > bufferMaxSize) totBytesToRead = bufferMaxSize;
    unsigned curBytesToRead = totBytesToRead;
    int curBytesRead;
    while ((curBytesRead = readSocket(envir(), fNextTCPReadStreamSocketNum,
                      &buffer[bytesRead], curBytesToRead,
                      fromAddress)) > 0) {
      bytesRead += curBytesRead;
      if (bytesRead >= totBytesToRead) break;
      curBytesToRead -= curBytesRead;
    }
    fNextTCPReadSize -= bytesRead;
    if (fNextTCPReadSize == 0) {
      // We've read all of the data that we asked for
      readSuccess = True;
    } else if (curBytesRead < 0) {
      // There was an error reading the socket
      bytesRead = 0;
      readSuccess = False;
    } else {
      // We need to read more bytes, and there was not an error reading the socket
      packetReadWasIncomplete = True;
      return True;
    }
    fNextTCPReadStreamSocketNum = -1; // default, for next time
  }

  if (readSuccess && fAuxReadHandlerFunc != NULL) {
    // Also pass the newly-read packet data to our auxilliary handler:
    (*fAuxReadHandlerFunc)(fAuxReadHandlerClientData, buffer, bytesRead);
  }
  return readSuccess;
}

void RTPInterface::stopNetworkReading() {
  // Normal case
  if (fGS != NULL) envir().taskScheduler().turnOffBackgroundReadHandling(fGS->socketNum());

  // Also turn off read handling on each of our TCP connections:
  for (tcpStreamRecord* streams = fTCPStreams; streams != NULL; streams = streams->fNext) {
    deregisterSocket(envir(), streams->fStreamSocketNum, streams->fStreamChannelId);
  }
}


////////// Helper Functions - Implementation /////////

Boolean RTPInterface::sendRTPorRTCPPacketOverTCP(unsigned char* data, unsigned size,
    int socketNum, int& sendNum)
{
#ifdef DEBUG_SEND
    fprintf(stderr, "sendRTPorRTCPPacketOverTCP: %d bytes over channel %d (socket %d)\n",
        packetSize, socketNum); fflush(stderr);
#endif
    if (!sendDataOverTCP(socketNum, data, size, False, sendNum)) {
        return False;
    }

    return True;
}

Boolean RTPInterface::sendRTPorRTCPPacketOverTCP(u_int8_t* packet, unsigned packetSize,
                         int socketNum, unsigned char streamChannelId) {
#ifdef DEBUG_SEND
  fprintf(stderr, "sendRTPorRTCPPacketOverTCP: %d bytes over channel %d (socket %d)\n",
      packetSize, streamChannelId, socketNum); fflush(stderr);
#endif
  // Send a RTP/RTCP packet over TCP, using the encoding defined in RFC 2326, section 10.12:
  //     $<streamChannelId><packetSize><packet>
  // (If the initial "send()" of '$<streamChannelId><packetSize>' succeeds, then we force
  // the subsequent "send()" for the <packet> data to succeed, even if we have to do so with
  // a blocking "send()".)
  do {
    u_int8_t framingHeader[4];
    framingHeader[0] = '$';
    framingHeader[1] = streamChannelId;
    framingHeader[2] = (u_int8_t) ((packetSize&0xFF00)>>8);
    framingHeader[3] = (u_int8_t) (packetSize&0xFF);
    if (!sendDataOverTCP(socketNum, framingHeader, 4, False)) break;

    if (!sendDataOverTCP(socketNum, packet, packetSize, True)) break;
#ifdef DEBUG_SEND
    fprintf(stderr, "sendRTPorRTCPPacketOverTCP: completed\n"); fflush(stderr);
#endif

    return True;
  } while (0);

#ifdef DEBUG_SEND
  fprintf(stderr, "sendRTPorRTCPPacketOverTCP: failed! (errno %d)\n", envir().getErrno()); fflush(stderr);
#endif
  return False;
}

#ifndef RTPINTERFACE_BLOCKING_WRITE_TIMEOUT_MS
#define RTPINTERFACE_BLOCKING_WRITE_TIMEOUT_MS 500
#endif

Boolean RTPInterface::sendDataOverTCP(int socketNum, u_int8_t const* data, unsigned dataSize, Boolean forceSendToSucceed) {
  sendDataOverTCPWriteFile(socketNum,data,dataSize);
  //if(gettickcurmill() > (g_lastsendtick[socketNum] + 100)) {
  //  printf("******************* bigger than 100.sock[%d],tick[%d].\n",socketNum,gettickcurmill()-g_lastsendtick[socketNum]);
  //}
  //g_lastsendtick[socketNum] = gettickcurmill();

  //unsigned int curtick = gettickcurmill();
  //unsigned int lasttick= mapSockSendTick[socketNum];
  //if(curtick > (lasttick +1000)) {
  //  printf("send packet526------ 1000.%d.sock=%d.\n",curtick-lasttick,socketNum);
  //}
  //else if(curtick > (lasttick +400)) {
  //  printf("send packet526------ 400.%d.sock=%d.\n",curtick-lasttick,socketNum);
  //}
  //else if(curtick > (lasttick +160)) {
  //  printf("send packet526------ 160.%d.sock=%d.\n",curtick-lasttick,socketNum);
  //}
  //else if(curtick > (lasttick +80)) {
  //  printf("send packet526------ 80.%d.sock=%d.\n",curtick-lasttick,socketNum);
  //}
  //else if(curtick > (lasttick +50)) {
  //  printf("send packet526------ 50.%d.sock=%d.\n",curtick-lasttick,socketNum);
  //}
  //mapSockSendTick[socketNum] = curtick;

  int sendResult = send(socketNum, (char const*)data, dataSize, 0/*flags*/);
  if(sendResult == -1) {
     printf("=================send failed.\n");
  }
  if (sendResult < (int)dataSize) {
     printf("=================send failed   sendResult < (int)dataSize.\n");
    // The TCP send() failed - at least partially.

    unsigned numBytesSentSoFar = sendResult < 0 ? 0 : (unsigned)sendResult;
    if (numBytesSentSoFar > 0 || (forceSendToSucceed && envir().getErrno() == EAGAIN)) {
      // The OS's TCP send buffer has filled up (because the stream's bitrate has exceeded
      // the capacity of the TCP connection!).
      // Force this data write to succeed, by blocking if necessary until it does:
      unsigned numBytesRemainingToSend = dataSize - numBytesSentSoFar;
#ifdef DEBUG_SEND
      fprintf(stderr, "sendDataOverTCP: resending %d-byte send (blocking)\n", numBytesRemainingToSend); fflush(stderr);
#endif
      makeSocketBlocking(socketNum, RTPINTERFACE_BLOCKING_WRITE_TIMEOUT_MS);
      sendResult = send(socketNum, (char const*)(&data[numBytesSentSoFar]), numBytesRemainingToSend, 0/*flags*/);
  if(sendResult == -1) {
     printf("=================send failed.\n");
  }
      if ((unsigned)sendResult != numBytesRemainingToSend) {
    // The blocking "send()" failed, or timed out.  In either case, we assume that the
    // TCP connection has failed (or is 'hanging' indefinitely), and we stop using it
    // (for both RTP and RTP).
    // (If we kept using the socket here, the RTP or RTCP packet write would be in an
    //  incomplete, inconsistent state.)
#ifdef DEBUG_SEND
    fprintf(stderr, "sendDataOverTCP: blocking send() failed (delivering %d bytes out of %d); closing socket %d\n", sendResult, numBytesRemainingToSend, socketNum); fflush(stderr);
#endif
  //printf("======%s,%d. call removeStreamSocket. socketNum=%d.\n",__FUNCTION__,__LINE__,socketNum);
    removeStreamSocket(socketNum, 0xFF);
    return False;
      }
      makeSocketNonBlocking(socketNum);

      return True;
    } else if (sendResult < 0 && envir().getErrno() != EAGAIN) {
      // Because the "send()" call failed, assume that the socket is now unusable, so stop
      // using it (for both RTP and RTCP):
  //printf("======%s,%d.call removeStreamSocket.socketNum=%d.\n",__FUNCTION__,__LINE__,socketNum);
      removeStreamSocket(socketNum, 0xFF);
    }

    return False;
  }

  return True;
}

Boolean RTPInterface::sendDataOverTCP(int socketNum, u_int8_t const* data, unsigned dataSize, Boolean forceSendToSucceed, int& sendSize)
{
  sendDataOverTCPWriteFile(socketNum,data,dataSize);
  if(gettickcurmill() > (g_lastsendtick[socketNum] + 100)) {
    //printf("******************* bigger than5 100.sock[%d],tick[%d].\n",socketNum,gettickcurmill()-g_lastsendtick[socketNum]);
  }
  g_lastsendtick[socketNum] = gettickcurmill();
  //printf("[%d]send packet5beg------sock=%d.\n",gettickcurmill(),socketNum);
  unsigned int curtick = gettickcurmill();
  static unsigned int lasttick= gettickcurmill();
  //if(curtick > (lasttick +1000)) {
  //  printf("[%d]send packet5------ 1000.%d.sock=%d.cur=%d,last=%d\n",gettickcurmill(),curtick-lasttick,socketNum,curtick,lasttick);
  //}
  //else if(curtick > (lasttick +400)) {
  //  printf("[%d]send packet5------ 400.%d.sock=%d.\n",gettickcurmill(),curtick-lasttick,socketNum);
  //}
  //else if(curtick > (lasttick +160)) {
  //  printf("[%d]send packet5------ 160.%d.sock=%d.\n",gettickcurmill(),curtick-lasttick,socketNum);
  //}
  //else if(curtick > (lasttick +80)) {
  //  printf("[%d]send packet5------ 80.%d.sock=%d.\n",gettickcurmill(),curtick-lasttick,socketNum);
  //}
  //else if(curtick > (lasttick +50)) {
  //  printf("[%d]send packet5------ 50.%d.sock=%d.\n",gettickcurmill(),curtick-lasttick,socketNum);
  //}
  //mapSockSendTick[socketNum] = curtick;
  lasttick = curtick;
  int sendResult = send(socketNum, (char const*)data, dataSize, 0/*flags*/);

  if (sendResult < 0 && envir().getErrno() == EAGAIN) {
    printf("sedn sendResult=%d\n",sendResult);
    makeSocketBlocking(socketNum, RTPINTERFACE_BLOCKING_WRITE_TIMEOUT_MS);
    sendResult = send(socketNum, (char const*)data, dataSize, 0/*flags*/);
    makeSocketNonBlocking(socketNum);
    unsigned curSize;
    SOCKLEN_T sizeSize = sizeof curSize;
    if (getsockopt(socketNum, SOL_SOCKET, SO_SNDBUF,(char*)&curSize, &sizeSize) < 0) {
      return 0;
    }
    curSize = curSize*2;
    setsockopt(socketNum, SOL_SOCKET, SO_SNDBUF, (char*)&curSize, sizeSize);
    printf("send buf is fulli curSize=%d.\n",curSize);
  }

  if (sendResult < (int)dataSize) {
    // The TCP send() failed - at least partially.
    //printf("*************socketNum[%d]*sendResult[%d] < (int)dataSize[%d]\n",socketNum,sendResult,(int)dataSize);
    unsigned numBytesSentSoFar = sendResult < 0 ? 0 : (unsigned)sendResult;
    if (numBytesSentSoFar > 0 || (forceSendToSucceed && envir().getErrno() == EAGAIN)) {
      // The OS's TCP send buffer has filled up (because the stream's bitrate has exceeded
      // the capacity of the TCP connection!).
      // Force this data write to succeed, by blocking if necessary until it does:
      unsigned numBytesRemainingToSend = dataSize - numBytesSentSoFar;
#ifdef DEBUG_SEND
      fprintf(stderr, "sendDataOverTCP: resending %d-byte send (blocking)\n", numBytesRemainingToSend); 
      fflush(stderr);
#endif
      makeSocketBlocking(socketNum, RTPINTERFACE_BLOCKING_WRITE_TIMEOUT_MS);
      sendResult = send(socketNum, (char const*)(&data[numBytesSentSoFar]), numBytesRemainingToSend, 0/*flags*/);
      //printf("**************sendResult[%d] numBytesRemainingToSend[%d]\n",sendResult,numBytesRemainingToSend);
      if ((unsigned)sendResult != numBytesRemainingToSend) {
        // The blocking "send()" failed, or timed out.  In either case, we assume that the
        // TCP connection has failed (or is 'hanging' indefinitely), and we stop using it
        // (for both RTP and RTP).
        // (If we kept using the socket here, the RTP or RTCP packet write would be in an
        //  incomplete, inconsistent state.)
#ifdef DEBUG_SEND
        fprintf(stderr, "sendDataOverTCP: blocking send() failed (delivering %d bytes out of %d); closing socket %d\n", sendResult, numBytesRemainingToSend, socketNum); 
        fflush(stderr);
#endif
      }
      makeSocketNonBlocking(socketNum);

      sendSize = numBytesSentSoFar;
      if (sendResult > 0) {
        sendSize += sendResult;
      }

      return True;
    } else if (sendResult < 0 && envir().getErrno() != EAGAIN) {
      // Because the "send()" call failed, assume that the socket is now unusable, so stop
      // using it (for both RTP and RTCP):
      printf("====%s,%d.call removeStreamSocket.socketNum=%d.\n",__FUNCTION__,__LINE__,socketNum);
      removeStreamSocket(socketNum, 0xFF);
      return False;
    }
    else {
      printf("send failed.\n");
    }
  }

  sendSize = dataSize;
  return True;
}

void RTPInterface::sendDataOverTCPWriteFile(int socketNum, u_int8_t const* data, unsigned dataSize) {

//  if(g_senddataFile[socketNum] == NULL) {
//    char fff[32] = {0};
//    sprintf(fff,"sendrtpovertcp_nal_%d.h264",socketNum);
//    g_senddataFile[socketNum] = fopen(fff,"wb+");
//  }
//  static FILE *fp_sendrtpovertcp = fopen("sendrtpovertcp_nal.h264","wb+");
  static int frame_index = 0;
  char tempbuf[32] = {0};
  int  templen = 0;
  if(dataSize < 18) {
    return ;
  }
  u_int8_t pt = data[5] & 0x7f;
  if((pt>=72) && (pt <= 78)) {
    //printf("this is rtcp packet.\n");
    return;
  }
  u_int8_t naltype = data[16] & 0x1f;
  if(naltype == 6) {
    //sei
    //printf("naltype: sei\n");
    tempbuf[0] = 0;
    tempbuf[1] = 0;
    tempbuf[2] = 0;
    tempbuf[3] = 1;
    tempbuf[4] = 0x06;
    templen = 5;

//    char filename[32] = {0};
//    sprintf(filename,"%d_nal.h264",frame_index);
//    FILE *fp_sei = fopen(filename,"wb+");
//    fwrite(tempbuf,1,templen,fp_sei);
//    fwrite(&data[17],1,dataSize-17,fp_sei);
//    fclose(fp_sei);
    frame_index++;
  //fwrite(tempbuf,1,templen,g_senddataFile[socketNum]);
  //fwrite(&data[17],1,dataSize-17,g_senddataFile[socketNum]);
  }
  else if (naltype == 7) {
    //sps
    //printf("naltype: sps\n");
    tempbuf[0] = 0;
    tempbuf[1] = 0;
    tempbuf[2] = 0;
    tempbuf[3] = 1;
    tempbuf[4] = 0x67;
    templen = 5;

//    char filename[32] = {0};
//    sprintf(filename,"%d_nal.h264",frame_index);
//    FILE *fp_sps = fopen(filename,"wb+");
//    fwrite(tempbuf,1,templen,fp_sps);
//    fwrite(&data[17],1,dataSize-17,fp_sps);
//    fclose(fp_sps);
    frame_index++;
  //fwrite(tempbuf,1,templen,g_senddataFile[socketNum]);
  //fwrite(&data[17],1,dataSize-17,g_senddataFile[socketNum]);
  }
  else if (naltype == 8) {
    //sps
    //printf("naltype: pps\n");
    tempbuf[0] = 0;
    tempbuf[1] = 0;
    tempbuf[2] = 0;
    tempbuf[3] = 1;
    tempbuf[4] = 0x68;
    templen = 5;
//    char filename[32] = {0};
//    sprintf(filename,"%d_nal.h264",frame_index);
//    FILE *fp_pps = fopen(filename,"wb+");
//    fwrite(tempbuf,1,templen,fp_pps);
//    fwrite(&data[17],1,dataSize-17,fp_pps);
//    fclose(fp_pps);
    frame_index++;
  //fwrite(tempbuf,1,templen,g_senddataFile[socketNum]);
  //fwrite(&data[17],1,dataSize-17,g_senddataFile[socketNum]);
  }
  else if (naltype == 28) {
    //i b p
    //printf("naltype: data\n");
    u_int8_t start = data[17]>>7;
    u_int8_t end   = data[17]>>6 & 0x01;
    u_int8_t type  = data[17] & 0x1f;
    if(start == 1) {
      //printf("***********: start\n");
      //插入 000001 65/61
      tempbuf[0] = 0;
      tempbuf[1] = 0;
      tempbuf[2] = 0;
      tempbuf[3] = 1;
      if(type == 5) {
        //printf("***********====: or i frame\n");
        tempbuf[4] = 0x65;
      }
      else {
        //printf("***********====: or p frame\n");
        tempbuf[4] = 0x61;
      }
      templen = 5;
    }
    else if (end == 1) {
      //printf("***********: end\n");
      templen = 0;
    }
    else {
      //printf("***********: middle\n");
      templen = 0;
    }
  //fwrite(tempbuf,1,templen,g_senddataFile[socketNum]);
  //fwrite(&data[18],1,dataSize-18,g_senddataFile[socketNum]);
  }
  else {
    //other printf
    //printf("naltype:unknow.\n");

  }

  return;
}

SocketDescriptor::SocketDescriptor(UsageEnvironment& env, int socketNum)
  :fEnv(env), fOurSocketNum(socketNum),
    fSubChannelHashTable(HashTable::create(ONE_WORD_HASH_KEYS)),
   fServerRequestAlternativeByteHandler(NULL), fServerRequestAlternativeByteHandlerClientData(NULL),
   fReadErrorOccurred(False), fDeleteMyselfNext(False), fAreInReadHandlerLoop(False), fTCPReadingState(AWAITING_DOLLAR) {
}

SocketDescriptor::~SocketDescriptor() {
  fEnv.taskScheduler().turnOffBackgroundReadHandling(fOurSocketNum);
  removeSocketDescription(fEnv, fOurSocketNum);

  if (fSubChannelHashTable != NULL) {
    // Remove knowledge of this socket from any "RTPInterface"s that are using it:
    HashTable::Iterator* iter = HashTable::Iterator::create(*fSubChannelHashTable);
    RTPInterface* rtpInterface;
    char const* key;

    while ((rtpInterface = (RTPInterface*)(iter->next(key))) != NULL) {
      u_int64_t streamChannelIdLong = (u_int64_t)key;
      unsigned char streamChannelId = (unsigned char)streamChannelIdLong;

      //printf("socket descibe xigou===\n");
      rtpInterface->removeStreamSocket(fOurSocketNum, streamChannelId);
    }
    delete iter;

    // Then remove the hash table entries themselves, and then remove the hash table:
    while (fSubChannelHashTable->RemoveNext() != NULL) {}
    delete fSubChannelHashTable;
  }

  // Finally:
  if (fServerRequestAlternativeByteHandler != NULL) {
    // Hack: Pass a special character to our alternative byte handler, to tell it that either
    // - an error occurred when reading the TCP socket, or
    // - no error occurred, but it needs to take over control of the TCP socket once again.
    u_int8_t specialChar = fReadErrorOccurred ? 0xFF : 0xFE;
    (*fServerRequestAlternativeByteHandler)(fServerRequestAlternativeByteHandlerClientData, specialChar);
  }
}

void SocketDescriptor::registerRTPInterface(unsigned char streamChannelId,
                        RTPInterface* rtpInterface) {
  Boolean isFirstRegistration = fSubChannelHashTable->IsEmpty();
#if defined(DEBUG_SEND)||defined(DEBUG_RECEIVE)
  fprintf(stderr, "SocketDescriptor(socket %d)::registerRTPInterface(channel %d): isFirstRegistration %d\n", fOurSocketNum, streamChannelId, isFirstRegistration);
#endif
  fSubChannelHashTable->Add((char const*)(long)streamChannelId,
                rtpInterface);

  if (isFirstRegistration) {
    // Arrange to handle reads on this TCP socket:
    TaskScheduler::BackgroundHandlerProc* handler
      = (TaskScheduler::BackgroundHandlerProc*)&tcpReadHandler;
    fEnv.taskScheduler().
      setBackgroundHandling(fOurSocketNum, SOCKET_READABLE|SOCKET_EXCEPTION, handler, this);
  }
}

RTPInterface* SocketDescriptor
::lookupRTPInterface(unsigned char streamChannelId) {
  char const* lookupArg = (char const*)(long)streamChannelId;
  return (RTPInterface*)(fSubChannelHashTable->Lookup(lookupArg));
}

void SocketDescriptor
::deregisterRTPInterface(unsigned char streamChannelId) {
#if defined(DEBUG_SEND)||defined(DEBUG_RECEIVE)
  fprintf(stderr, "SocketDescriptor(socket %d)::deregisterRTPInterface(channel %d)\n", fOurSocketNum, streamChannelId);
#endif
  fSubChannelHashTable->Remove((char const*)(long)streamChannelId);

  if (fSubChannelHashTable->IsEmpty() || streamChannelId == 0xFF) {
    // No more interfaces are using us, so it's curtains for us now:
    if (fAreInReadHandlerLoop) {
      fDeleteMyselfNext = True; // we can't delete ourself yet, but we'll do so from "tcpReadHandler()" below
    } else {
      delete this;
    }
  }
}

void SocketDescriptor::tcpReadHandler(SocketDescriptor* socketDescriptor, int mask) {
  // Call the read handler until it returns false, with a limit to avoid starving other sockets
  unsigned count = 2000;
  socketDescriptor->fAreInReadHandlerLoop = True;
  while (!socketDescriptor->fDeleteMyselfNext && socketDescriptor->tcpReadHandler1(mask) && --count > 0) {}
  socketDescriptor->fAreInReadHandlerLoop = False;
  if (socketDescriptor->fDeleteMyselfNext) delete socketDescriptor;
}

Boolean SocketDescriptor::tcpReadHandler1(int mask) {
  // We expect the following data over the TCP channel:
  //   optional RTSP command or response bytes (before the first '$' character)
  //   a '$' character
  //   a 1-byte channel id
  //   a 2-byte packet size (in network byte order)
  //   the packet data.
  // However, because the socket is being read asynchronously, this data might arrive in pieces.
  
  u_int8_t c;
  struct sockaddr_in fromAddress;
  if (fTCPReadingState != AWAITING_PACKET_DATA) {
    int result = readSocket(fEnv, fOurSocketNum, &c, 1, fromAddress);
    if (result == 0) { // There was no more data to read
      return False;
    } else if (result != 1) { // error reading TCP socket, so we will no longer handle it
#ifdef DEBUG_RECEIVE
      fprintf(stderr, "SocketDescriptor(socket %d)::tcpReadHandler(): readSocket(1 byte) returned %d (error)\n", fOurSocketNum, result);
#endif
      fReadErrorOccurred = True;
      fDeleteMyselfNext = True;
      return False;
    }
  }

  Boolean callAgain = True;
  switch (fTCPReadingState) {
    case AWAITING_DOLLAR: {
      if (c == '$') {
#ifdef DEBUG_RECEIVE
    fprintf(stderr, "SocketDescriptor(socket %d)::tcpReadHandler(): Saw '$'\n", fOurSocketNum);
#endif
    fTCPReadingState = AWAITING_STREAM_CHANNEL_ID;
      } else {
    // This character is part of a RTSP request or command, which is handled separately:
    if (fServerRequestAlternativeByteHandler != NULL && c != 0xFF && c != 0xFE) {
      // Hack: 0xFF and 0xFE are used as special signaling characters, so don't send them
      (*fServerRequestAlternativeByteHandler)(fServerRequestAlternativeByteHandlerClientData, c);
    }
      }
      break;
    }
    case AWAITING_STREAM_CHANNEL_ID: {
      // The byte that we read is the stream channel id.
      if (lookupRTPInterface(c) != NULL) { // sanity check
    fStreamChannelId = c;
    fTCPReadingState = AWAITING_SIZE1;
      } else {
    // This wasn't a stream channel id that we expected.  We're (somehow) in a strange state.  Try to recover:
#ifdef DEBUG_RECEIVE
    fprintf(stderr, "SocketDescriptor(socket %d)::tcpReadHandler(): Saw nonexistent stream channel id: 0x%02x\n", fOurSocketNum, c);
#endif
    fTCPReadingState = AWAITING_DOLLAR;
      }
      break;
    }
    case AWAITING_SIZE1: {
      // The byte that we read is the first (high) byte of the 16-bit RTP or RTCP packet 'size'.
      fSizeByte1 = c;
      fTCPReadingState = AWAITING_SIZE2;
      break;
    }
    case AWAITING_SIZE2: {
      // The byte that we read is the second (low) byte of the 16-bit RTP or RTCP packet 'size'.
      unsigned short size = (fSizeByte1<<8)|c;
      
      // Record the information about the packet data that will be read next:
      RTPInterface* rtpInterface = lookupRTPInterface(fStreamChannelId);
      if (rtpInterface != NULL) {
    rtpInterface->fNextTCPReadSize = size;
    rtpInterface->fNextTCPReadStreamSocketNum = fOurSocketNum;
    rtpInterface->fNextTCPReadStreamChannelId = fStreamChannelId;
      }
      fTCPReadingState = AWAITING_PACKET_DATA;
      break;
    }
    case AWAITING_PACKET_DATA: {
      callAgain = False;
      fTCPReadingState = AWAITING_DOLLAR; // the next state, unless we end up having to read more data in the current state
      // Call the appropriate read handler to get the packet data from the TCP stream:
      RTPInterface* rtpInterface = lookupRTPInterface(fStreamChannelId);
      if (rtpInterface != NULL) {
    if (rtpInterface->fNextTCPReadSize == 0) {
      // We've already read all the data for this packet.
      break;
    }
    if (rtpInterface->fReadHandlerProc != NULL) {
#ifdef DEBUG_RECEIVE
      fprintf(stderr, "SocketDescriptor(socket %d)::tcpReadHandler(): reading %d bytes on channel %d\n", fOurSocketNum, rtpInterface->fNextTCPReadSize, rtpInterface->fNextTCPReadStreamChannelId);
#endif
      fTCPReadingState = AWAITING_PACKET_DATA;
      rtpInterface->fReadHandlerProc(rtpInterface->fOwner, mask);
    } else {
#ifdef DEBUG_RECEIVE
      fprintf(stderr, "SocketDescriptor(socket %d)::tcpReadHandler(): No handler proc for \"rtpInterface\" for channel %d; need to skip %d remaining bytes\n", fOurSocketNum, fStreamChannelId, rtpInterface->fNextTCPReadSize);
#endif
      int result = readSocket(fEnv, fOurSocketNum, &c, 1, fromAddress);
      if (result < 0) { // error reading TCP socket, so we will no longer handle it
#ifdef DEBUG_RECEIVE
        fprintf(stderr, "SocketDescriptor(socket %d)::tcpReadHandler(): readSocket(1 byte) returned %d (error)\n", fOurSocketNum, result);
#endif
        fReadErrorOccurred = True;
        fDeleteMyselfNext = True;
        return False;
      } else {
        fTCPReadingState = AWAITING_PACKET_DATA;
        if (result == 1) {
          --rtpInterface->fNextTCPReadSize;
          callAgain = True;
        }
      }
    }
      }
#ifdef DEBUG_RECEIVE
      else fprintf(stderr, "SocketDescriptor(socket %d)::tcpReadHandler(): No \"rtpInterface\" for channel %d\n", fOurSocketNum, fStreamChannelId);
#endif
    }
  }

  return callAgain;
}


////////// tcpStreamRecord implementation //////////

tcpStreamRecord
::tcpStreamRecord(int streamSocketNum, unsigned char streamChannelId,
          tcpStreamRecord* next)
  : fNext(next),
    fStreamSocketNum(streamSocketNum), fStreamChannelId(streamChannelId), fReadCurPos(0), fWriteCurPos(0), fMaxSize(4000000) {
    //printf("%s,%d. delete next[%p].\n",__FUNCTION__,__LINE__,next);

        // fDataBuf = new unsigned char[fMaxSize];
}

tcpStreamRecord::~tcpStreamRecord() {
  //printf("%s,%d. fNext.\n",__FUNCTION__,__LINE__);
  if (fNext) {
    //printf("%s,%d. delete fNext[%p].\n",__FUNCTION__,__LINE__,fNext);
    delete fNext;
  }
  // if (fDataBuf){
  //   delete fDataBuf;
  //   fDataBuf = NULL;
  // }
}

Boolean tcpStreamRecord::PushDataBuf(unsigned char* dataBuf, int length)
{
    int spaceSize = fMaxSize - fWriteCurPos + fReadCurPos; 
    if (length  > spaceSize)
    {
        //printf("**********************************************************************\n");
        //printf("**********************************************************************\n");
        //printf("**********************************************************************\n");
        //printf("space is smal.\n");
        //printf("**********************************************************************\n");
        //printf("**********************************************************************\n");
        //printf("**********************************************************************\n");
        return False;
    }

    if (length > fMaxSize - fWriteCurPos)
    {
        //move
        memcpy(fDataBuf, fDataBuf + fReadCurPos, fWriteCurPos - fReadCurPos);
        fReadCurPos = 0;
        fWriteCurPos = fWriteCurPos - fReadCurPos;
    }

    memcpy(fDataBuf + fWriteCurPos, dataBuf, length);
    fWriteCurPos += length;
    return True;
}

unsigned char* tcpStreamRecord::DataBuf()
{
    return fDataBuf + fReadCurPos;
}

int tcpStreamRecord::DataLength()
{
    return fWriteCurPos - fReadCurPos;
}

Boolean tcpStreamRecord::IsEmpty() const
{
    if (fReadCurPos == fWriteCurPos)
    {
        return True;
    }
    else
    {
        return False;
    }
}

void tcpStreamRecord::Reset()
{
    fReadCurPos = 0;
    fWriteCurPos = 0;
}

int tcpStreamRecord::Space() const
{
    return fMaxSize - fWriteCurPos + fReadCurPos;
}

 void tcpStreamRecord::OffsetReadSize(int offset)
 {
     fReadCurPos += offset;
 }
