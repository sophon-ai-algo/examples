/*==========================================================================
 * Copyright 2016-2022 by Sophgo Technologies Inc. All rights reserved.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
============================================================================*/
//
// Created by hsyuan on 2019-02-28.
//

#ifndef BMUTILITY_STREAM_SEI_H
#define BMUTILITY_STREAM_SEI_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

uint32_t reversebytes(uint32_t value);

uint32_t h264sei_calc_packet_size(uint32_t size);
int h264sei_packet_write(uint8_t *oPacketBuf, bool isAnnexb, const uint8_t *content, uint32_t size);
int h264sei_packet_read(uint8_t *inPacket, uint32_t size, uint8_t *buffer, int buff_size);

// H265
int h265sei_packet_write(unsigned char * packet, bool isAnnexb, const uint8_t * content, uint32_t size);
int h265sei_packet_read(unsigned char * packet, uint32_t size, uint8_t * buffer, int buf_size);

#endif //PROJECT_BM_LOG_H
