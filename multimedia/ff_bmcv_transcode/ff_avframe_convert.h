#ifndef __AVFRAME_CONVERT_H_
#define __AVFRAME_CONVERT_H_

extern "C" {
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"
#include "libavformat/avformat.h"
#include "libavfilter/buffersink.h"
#include "libavfilter/buffersrc.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include <stdio.h>
#include <unistd.h>
#include "bmcv_api.h"
#include "bmcv_api_ext.h"
}


#define USEING_MEM_HEAP2 4

typedef struct{
        bm_image *bmImg;
        uint8_t* buf0;
        uint8_t* buf1;
        uint8_t* buf2;
}transcode_t;

int avframeToAvframeConvertPixSize(bm_handle_t &bmHandle,AVFrame *inPic,AVFrame *outPic,int enc_frame_height,int enc_frame_width,int enc_pix_format);

#endif