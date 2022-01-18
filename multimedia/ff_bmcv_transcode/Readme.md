## How to build
 * modify the PRODUCTFORM and top_dir at the Mafefile accoring to the actual situation
 * PRODUCTFORM: can be x86,soc, arm_pcie, loongarch64
 * top_dir: the root directory of sdks

## how to run
in pcie crad
    ./test_ff_bmcv_transcode input.mp4 test.mp4 I420 h265_bm 1280 720 25 3000 1 1 0
    I420: NV12 transcode to yuv420
    h265_bm: Hardware speedup encode
    1280*720: resize para
    25:fps
    3000: bitrate
    1:thread num
    1:YUV data does not copy to host memory
    0:card 0
    ./test_ff_video_transcode input.mp4 test.mp4 NV12 h265_bm 1280 720 25 3000 1 1 0
    NV12: no pixel format transcode
    h265_bm: Hardware speedup encode
    1280*720: If it is nv12, this parameter is invalid.The image size is generated according to the original image size
    25:fps
    3000: bitrate
    1:thread num
    1:YUV data does not copy to host memory
    0:card 0


in soc
    ./test_ff_bmcv_transcode input.mp4 test.mp4 I420 h265_bm 1280 720 25 3000 1
    I420: NV12 transcode to yuv420
    h265_bm: Hardware speedup encode
    1280*720: resize para
    25:fps
    3000: bitrate
    1:thread num

    ./test_ff_video_transcode input.mp4 test.mp4 NV12 h265_bm 1280 720 25 3000 1
    NV12: no pixel format transcode
    h265_bm: Hardware speedup encode
    1280*720: If it is nv12, this parameter is invalid.The image size is generated according to the original image size
    25:fps
    3000: bitrate
    1:thread num


If the program is multithreaded the Target file will be test0.mp4 test1.mp4 test2.mp4 ...

