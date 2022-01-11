//
// Created by hsyuan on 2019-02-28.
//

#include "stream_sei.h"
#include <memory.h>

#define UUID_SIZE 16

//FFMPEG uuid
//static unsigned char uuid[] = { 0xdc, 0x45, 0xe9, 0xbd, 0xe6, 0xd9, 0x48, 0xb7, 0x96, 0x2c, 0xd8, 0x20, 0xd9, 0x23, 0xee, 0xef };

//self UUID
static unsigned char uuid[] = { 0x54, 0x80, 0x83, 0x97, 0xf0, 0x23, 0x47, 0x4b, 0xb7, 0xf7, 0x4f, 0x32, 0xb5, 0x4e, 0x06, 0xac };

//��ʼ��
static unsigned char start_code[] = { 0x00,0x00,0x00,0x01 };


uint32_t h264sei_calc_nalu_size(uint32_t content)
{
    //SEI payload size
    uint32_t sei_payload_size = content + UUID_SIZE;
    //NALU + payload���� + ���ݳ��� + ����
    uint32_t sei_size = 1 + 1 + (sei_payload_size / 0xFF + (sei_payload_size % 0xFF != 0 ? 1 : 0)) + sei_payload_size;
    //��ֹ��
    uint32_t tail_size = 2;
    if (sei_size % 2 == 1)
    {
        tail_size -= 1;
    }
    sei_size += tail_size;

    return sei_size;
}

uint32_t h264sei_calc_packet_size(uint32_t sei_size)
{
    // Packet size = NALUSize + StartCodeSize
    return h264sei_calc_nalu_size(sei_size) + 4;
}

int h264sei_packet_write(unsigned char * packet, bool isAnnexb, const uint8_t * content, uint32_t size)
{
    unsigned char * data = (unsigned char*)packet;
    //unsigned int nalu_size = (unsigned int)h264sei_calc_nalu_size(size);
    //uint32_t sei_size = nalu_size;


    memcpy(data, start_code, sizeof(start_code));

    data += sizeof(unsigned int);

    //unsigned char * sei = data;
    //NAL header
    *data++ = 6; //SEI
    //sei payload type
    *data++ = 5; //unregister
    size_t sei_payload_size = size + UUID_SIZE;

    while (true)
    {
        *data++ = (sei_payload_size >= 0xFF ? 0xFF : (char)sei_payload_size);
        if (sei_payload_size < 0xFF) break;
        sei_payload_size -= 0xFF;
    }

    //UUID
    memcpy(data, uuid, UUID_SIZE);
    data += UUID_SIZE;

    memcpy(data, content, size);
    data += size;

    *data = 0x80;
    data++;


    return data - packet;
}

static int get_sei_buffer(unsigned char * data, uint32_t size, uint8_t * buff, int buf_size)
{
    unsigned char * sei = data;
    int sei_type = 0;
    unsigned sei_size = 0;
    //payload type
    do {
        sei_type += *sei;
    } while (*sei++ == 255);

    do {
        sei_size += *sei;
    } while (*sei++ == 255);

    if (sei_size >= UUID_SIZE && sei_size <= (data + size - sei) &&
        sei_type == 5 && memcmp(sei, uuid, UUID_SIZE) == 0)
    {
        sei += UUID_SIZE;
        sei_size -= UUID_SIZE;

        if (buff != NULL && buf_size != 0)
        {
            if (buf_size > (int)sei_size)
            {
                memcpy(buff, sei, sei_size);
            }else{
                printf("ERROR:input buffer(%d) < SEI size(%d)\n", buf_size, sei_size);
                return -1;
            }
        }

        return sei_size;
    }
    return -1;
}

int h264sei_packet_read(unsigned char * packet, uint32_t size, uint8_t * buffer, int buf_size)
{
    unsigned char ANNEXB_CODE_LOW[] = { 0x00,0x00,0x01 };
    unsigned char ANNEXB_CODE[] = { 0x00,0x00,0x00,0x01 };

    unsigned char *data = packet;
    bool isAnnexb = false;
    if ((size > 3 && memcmp(data, ANNEXB_CODE_LOW, 3) == 0) ||
        (size > 4 && memcmp(data, ANNEXB_CODE, 4) == 0)
            )
    {
        isAnnexb = true;
    }

    if (isAnnexb)
    {
        while (data < packet + size) {
            if ((packet + size - data) > 4 && data[0] == 0x00 && data[1] == 0x00)
            {
                int startCodeSize = 2;
                if (data[2] == 0x01)
                {
                    startCodeSize = 3;
                }
                else if (data[2] == 0x00 && data[3] == 0x01)
                {
                    startCodeSize = 4;
                }

                if (startCodeSize == 3 || startCodeSize == 4)
                {
                    if ((packet + size - data) > (startCodeSize + 1))
                    {
                        //SEI
                        unsigned char * sei = data + startCodeSize + 1;

                        int ret = get_sei_buffer(sei, (packet + size - sei), buffer, buf_size);
                        if (ret != -1)
                        {
                            return ret;
                        }
                    }
                    data += startCodeSize + 1;
                }
                else
                {
                    data += startCodeSize + 1;
                }
            }
            else
            {
                data++;
            }
        }
    }
    else
    {
        printf("can't find NALU startCode\n");
    }
    return -1;
}


int h265sei_packet_write(unsigned char * packet, bool isAnnexb, const uint8_t * content, uint32_t size)
{
    unsigned char * data = (unsigned char*)packet;
    //unsigned int nalu_size = (unsigned int)h264sei_calc_nalu_size(size);
    //uint32_t sei_size = nalu_size;

    memcpy(data, start_code, sizeof(start_code));

    data += sizeof(unsigned int);

    uint8_t nalUnitType = 39;
    //unsigned char * sei = data;
    //NAL header
    *data++ = (uint8_t)nalUnitType << 1;
    *data++ = 1 + (nalUnitType == 2);
    //sei payload type
    *data++ = 5; //unregister
    size_t sei_payload_size = size + UUID_SIZE;

    while (true)
    {
        *data++ = (sei_payload_size >= 0xFF ? 0xFF : (char)sei_payload_size);
        if (sei_payload_size < 0xFF) break;
        sei_payload_size -= 0xFF;
    }

    //UUID
    memcpy(data, uuid, UUID_SIZE);
    data += UUID_SIZE;

    memcpy(data, content, size);
    data += size;

    *data = 0x80;
    data++;


    return data - packet;
}

int h265sei_packet_read(unsigned char * packet, uint32_t size, uint8_t * buffer, int buf_size)
{
    unsigned char ANNEXB_CODE_LOW[] = { 0x00,0x00,0x01 };
    unsigned char ANNEXB_CODE[] = { 0x00,0x00,0x00,0x01 };

    unsigned char *data = packet;
    bool isAnnexb = false;
    if ((size > 3 && memcmp(data, ANNEXB_CODE_LOW, 3) == 0) ||
        (size > 4 && memcmp(data, ANNEXB_CODE, 4) == 0)
            )
    {
        isAnnexb = true;
    }

    if (isAnnexb)
    {
        while (data < packet + size) {
            if ((packet + size - data) > 4 && data[0] == 0x00 && data[1] == 0x00)
            {
                int startCodeSize = 2;
                if (data[2] == 0x01)
                {
                    startCodeSize = 3;
                }
                else if (data[2] == 0x00 && data[3] == 0x01)
                {
                    startCodeSize = 4;
                }

                if (startCodeSize == 3 || startCodeSize == 4)
                {
                    if ((packet + size - data) > (startCodeSize + 2))
                    {
                        //SEI
                        unsigned char * sei = data + startCodeSize + 2;

                        int ret = get_sei_buffer(sei, (packet + size - sei), buffer, buf_size);
                        if (ret != -1)
                        {
                            return ret;
                        }
                    }
                    data += startCodeSize + 2;
                }
                else
                {
                    data += startCodeSize + 2;
                }
            }
            else
            {
                data++;
            }
        }
    }
    else
    {
        printf("can't find NALU startCode\n");
    }
    return -1;
}
