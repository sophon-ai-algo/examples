## 功能介绍
用于测试opencv视频编码功能

## 如何编译

可选编译参数:
* DEBUG: 是否输出调试信息，默认不输出(DEBUG=0)
* PRODUCTFORM: 芯片工作模式，支持soc(独立主机)、pcie(x86平台)、arm_pcie(arm平台)选项，请根据使用环境指定
* top_dir: SDK根目录，安装SDK后会自动配置

如, 在x86平台、pcie工作模式下进行编译:
``` bash
make PRODUCTFORM=pcie
```
编译后生成可执行文件<strong> test_ocv_video_xcode </strong>

## 如何运行
``` bash
test_ocv_video_xcode input code_type frame_num outputname yuv_enable roi_enable [device_id] [encodeparams]
```
参数说明:
* input: 输入视频文件或码流, 支持rtsp、rtmp协议
* code_type: 指定编码器, 可选的编码器有<strong>H264enc</strong>、<strong>H265enc</strong>、<strong>MPEG2enc</strong>
* frame_num: 编码帧数
* outputname: 输出文件名
* yuv_enable: 输出文件格式, 0: BGR格式, 1: YUV格式
* roi_enable: 是否启用roi编码, 0: 不启用, 1: 启用<strong><em> *** 如果启用roi编码, 请将outputname设置为null或Null, 并在encodeparams中设置roi_eanble=1 *** </em></strong>
* device_id: 使用的芯片序号(对应/dev下的设备文件序号)<strong><em> *** 只在pcie模式下支持 *** </em></strong>
* encodeparams: 编码参数, 可配置的参数如下: gop=30:bitrate=800:gop_preset=2:mb_rc=1:delta_qp=3:min_qp=20:max_qp=40:roi_enable=1:push_stream=rtmp/rtsp

命令示例:
``` bash 
> ./test_ocv_video_xcode ~/multimedia_files/h265_1920x1080_30fps_1024k_main.mp4 H265enc 30 encoder_test265.ts 1 0 0 bitrate=1000
[/home/jenkins/workspace/all_in_one_sa5_daily/daily_build/bmetc/sa5/middleware-soc/bm_opencv/modules/core/src/cv_bmcpu.cpp:49->InternalBMCpuRegister]total 4 devices need to enable on-chip CPU. It may need serveral minutes                     for loading, please be patient....
[/home/jenkins/workspace/all_in_one_sa5_daily/daily_build/bmetc/sa5/middleware-soc/bm_opencv/modules/core/src/cv_bmcpu.cpp:226->getLocationPath]bmcpu full path: /home/bitmain/sophonsdk_vMaster/scripts/../lib/opencv/pcie/bmcpu/
[/home/jenkins/workspace/all_in_one_sa5_daily/daily_build/bmetc/sa5/middleware-soc/bm_opencv/modules/core/src/cv_bmcpu.cpp:95->InternalBMCpuRegister]0/4 devices finished
[/home/jenkins/workspace/all_in_one_sa5_daily/daily_build/bmetc/sa5/middleware-soc/bm_opencv/modules/core/src/cv_bmcpu.cpp:95->InternalBMCpuRegister]1/4 devices finished
[/home/jenkins/workspace/all_in_one_sa5_daily/daily_build/bmetc/sa5/middleware-soc/bm_opencv/modules/core/src/cv_bmcpu.cpp:95->InternalBMCpuRegister]2/4 devices finished
[/home/jenkins/workspace/all_in_one_sa5_daily/daily_build/bmetc/sa5/middleware-soc/bm_opencv/modules/core/src/cv_bmcpu.cpp:95->InternalBMCpuRegister]3/4 devices finished
[/home/jenkins/workspace/all_in_one_sa5_daily/daily_build/bmetc/sa5/middleware-soc/bm_opencv/modules/core/src/cv_bmcpu.cpp:104->InternalBMCpuRegister]device 0 failed to enable on-chip CPU! If not using bmcpu_opencv function, please ignore it.
[/home/jenkins/workspace/all_in_one_sa5_daily/daily_build/bmetc/sa5/middleware-soc/bm_opencv/modules/core/src/cv_bmcpu.cpp:104->InternalBMCpuRegister]device 1 failed to enable on-chip CPU! If not using bmcpu_opencv function, please ignore it.
[/home/jenkins/workspace/all_in_one_sa5_daily/daily_build/bmetc/sa5/middleware-soc/bm_opencv/modules/core/src/cv_bmcpu.cpp:104->InternalBMCpuRegister]device 2 failed to enable on-chip CPU! If not using bmcpu_opencv function, please ignore it.
[/home/jenkins/workspace/all_in_one_sa5_daily/daily_build/bmetc/sa5/middleware-soc/bm_opencv/modules/core/src/cv_bmcpu.cpp:104->InternalBMCpuRegister]device 3 failed to enable on-chip CPU! If not using bmcpu_opencv function, please ignore it.
BMvidDecCreateW5 board id 0 coreid 0
[VDI] Open board 0, core 0, fd 5, dev /dev/bm-sophon0
libbmvideo.so addr : /home/bitmain/sophonsdk_vMaster/scripts/../lib/decode/pcie/libbmvideo.so, name_len: 59
vpu firmware addr: /home/bitmain/sophonsdk_vMaster/scripts/../lib/decode/pcie/vpu_firmware/chagall_dec.bin
[VDI] Open board 0, core 0, fd 5, dev /dev/bm-sophon0
VERSION=0, REVISION=213135
CAP_PROP_OUTPUT_SAR: 1
orig CAP_PROP_FRAME_HEIGHT: 1080
orig CAP_PROP_FRAME_WIDTH: 1920
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
OpenCV: FFMPEG: tag 0x31637668/'hvc1' is not supported with codec id 173 and format 'mpegts / MPEG-TS (MPEG-2 Transport Stream)'
[7f20d3560700] src/enc.c:270 (vpu_EncInit)   sophon_idx 0, VPU core index 4
[7f20d3560700] src/vdi.c:137 (bm_vdi_init)   [VDI] Open device /dev/bm-sophon0, fd=7
[7f20d3560700] src/vdi.c:137 (bm_vdi_init)   [VDI] Open device /dev/bm-sophon0, fd=7
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
OpenCV: FFMPEG: tag 0x31637668/'hvc1' is not supported with codec id 173 and format 'mpegts / MPEG-TS (MPEG-2 Transport Stream)'
[7f20d3560700] src/enc.c:270 (vpu_EncInit)   sophon_idx 0, VPU core index 4
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
[7f20d3560700] src/vdi.c:137 (bm_vdi_init)   [VDI] Open device /dev/bm-sophon0, fd=7
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
exit: stream EOF
[/home/jenkins/workspace/all_in_one_sa5_daily/daily_build/bmetc/sa5/middleware-soc/bm_opencv/modules/core/src/cv_bmcpu.cpp:113->~InternalBMCpuRegister]deconstructor function is called
```