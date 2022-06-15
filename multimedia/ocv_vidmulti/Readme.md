## 功能介绍
用于测试opencv视频解码或转码性能

## 如何编译

可选编译参数:
* DEBUG: 是否输出调试信息，默认不输出(DEBUG=0)
* PRODUCTFORM: 芯片工作模式，支持soc(独立主机)、pcie(x86平台)、arm_pcie(arm平台)选项，请根据使用环境指定
* top_dir: SDK根目录，安装SDK后会自动配置

如, 在x86平台、pcie工作模式下进行编译:
``` bash
make PRODUCTFORM=pcie
```
编译后生成可执行文件<strong> test_ocv_vidmulti </strong>

## 如何运行
``` bash
test_ocv_vidmulti thread_num input_video [card] [enc_enable] input_video [card] [enc_enable] ...
```
参数说明:
* thread_num: 解码路数, 最大支持512路
* input_video: 输入视频文件
* card: 使用的芯片序号(对应/dev下的设备文件序号)<strong><em> *** 只在pcie模式下支持 *** </em></strong>
* enc_enable: 是否进行编码, 0: 不编码, 1: 编码

---

解码多路视频时需要对应输入多组 input_video [card] [enc_enable] 参数</br>
配置环境变量 export VIDMULTI_DSIPLAY_FRAMERATE=1 可查看帧率、分辨率

---

命令示例:
``` bash 
> ./test_ocv_vidmulti 2 ~/multimedia_files/1920_1088.264 0 1 ~/multimedia_files/station.avi 1 0
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
[VDI] Open board 0, core 0, fd 7, dev /dev/bm-sophon0
libbmvideo.so addr : /home/bitmain/sophonsdk_vMaster/scripts/../lib/decode/pcie/libbmvideo.so, name_len: 59
vpu firmware addr: /home/bitmain/sophonsdk_vMaster/scripts/../lib/decode/pcie/vpu_firmware/chagall_dec.bin
[VDI] Open board 0, core 0, fd 7, dev /dev/bm-sophon0
VERSION=0, REVISION=213135
[7fbfb7bd3700] src/enc.c:270 (vpu_EncInit)   sophon_idx 0, VPU core index 4
[7fbfb7bd3700] src/vdi.c:137 (bm_vdi_init)   [VDI] Open device /dev/bm-sophon0, fd=10
[7fbfb7bd3700] src/vdi.c:137 (bm_vdi_init)   [VDI] Open device /dev/bm-sophon0, fd=10
[7fbfb7bd3700] src/enc.c:270 (vpu_EncInit)   sophon_idx 0, VPU core index 4
[7fbfb7bd3700] src/vdi.c:137 (bm_vdi_init)   [VDI] Open device /dev/bm-sophon0, fd=10
BMvidDecCreateW5 board id 1 coreid 5
[VDI] Open board 1, core 0, fd 14, dev /dev/bm-sophon1
libbmvideo.so addr : /home/bitmain/sophonsdk_vMaster/scripts/../lib/decode/pcie/libbmvideo.so, name_len: 59
vpu firmware addr: /home/bitmain/sophonsdk_vMaster/scripts/../lib/decode/pcie/vpu_firmware/chagall_dec.bin
[VDI] Open board 1, core 0, fd 14, dev /dev/bm-sophon1
VERSION=0, REVISION=213135
maybe grab ends normally, retry count = 513
file ends!
BMvidDecCreateW5 board id 1 coreid 5
libbmvideo.so addr : /home/bitmain/sophonsdk_vMaster/scripts/../lib/decode/pcie/libbmvideo.so, name_len: 59
vpu firmware addr: /home/bitmain/sophonsdk_vMaster/scripts/../lib/decode/pcie/vpu_firmware/chagall_dec.bin
[VDI] Open board 1, core 0, fd 15, dev /dev/bm-sophon1
VERSION=0, REVISION=213135
loop again id:1rtsp: /home/bitmain/multimedia_files/station.avi
may be endof.. please check it.............
may be endof.. please check it.............
may be endof.. please check it.............
may be endof.. please check it.............
maybe grab ends normally, retry count = 513
file ends!
BMvidDecCreateW5 board id 0 coreid 0
libbmvideo.so addr : /home/bitmain/sophonsdk_vMaster/scripts/../lib/decode/pcie/libbmvideo.so, name_len: 59
vpu firmware addr: /home/bitmain/sophonsdk_vMaster/scripts/../lib/decode/pcie/vpu_firmware/chagall_dec.bin
[VDI] Open board 0, core 0, fd 13, dev /dev/bm-sophon0
VERSION=0, REVISION=213135
loop again id:0rtsp: /home/bitmain/multimedia_files/1920_1088.264
maybe grab ends normally, retry count = 513
file ends!
...
```