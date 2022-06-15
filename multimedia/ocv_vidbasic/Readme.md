## 功能介绍
用于测试opencv视频解码功能

## 如何编译

可选编译参数:
* DEBUG: 是否输出调试信息，默认不输出(DEBUG=0)
* PRODUCTFORM: 芯片工作模式，支持soc(独立主机)、pcie(x86平台)、arm_pcie(arm平台)选项，请根据使用环境指定
* top_dir: SDK根目录，安装SDK后会自动配置

如, 在x86平台、pcie工作模式下进行编译:
``` bash
make PRODUCTFORM=pcie
```
编译后生成可执行文件<strong> test_ocv_vidbasic </strong>

## 如何运行
``` bash
test_ocv_vidbasic <input_video> <output_name> <frame_num> <yuv_enable> [card] [WxH] [dump.BGR or dump.YUV]
```
参数说明:
* input_video: 输入视频文件
* output_name: 输出文件名
* frame_num: 视频解码帧数
* yuv_enable: 输出文件格式, 0: png文件(BGR格式), 1: jpg文件(YUV格式)
* card: 使用的芯片序号(对应/dev下的设备文件序号)<strong><em> *** 只在pcie模式下支持 *** </em></strong>
* WxH: 非必填, 指定输出文件的宽度和高度, 最大支持4096x4096, 默认不改变宽高
* dump.BGR or dump.YUV: 非必填，指定输出dump文件名, 请根据输出文件格式指定后缀为.BGR或.YUV, 默认不生成dump文件

命令示例:
``` bash 
> ./test_ocv_vidbasic ~/multimedia_files/h265_1920x1080_30fps_1024k_main.mp4 test 10 1 0 128x128 dump.YUV
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
CAP_PROP_FRAME_HEIGHT: 128
CAP_PROP_FRAME_WIDTH: 128
Warning: compression frame use coded width/height (1920, 1080), it should be 16-aligned(avc) 32-aligned(hevc)!!
Open /dev/bm-sophon0 successfully, device index = 0, jpu fd = 9, vpp fd = 9
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