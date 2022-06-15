## 功能介绍
用于测试opencv图像解码功能

## 如何编译

可选编译参数:
* DEBUG: 是否输出调试信息，默认不输出(DEBUG=0)
* PRODUCTFORM: 芯片工作模式，支持soc(独立主机)、pcie(x86平台)、arm_pcie(arm平台)选项，请根据使用环境指定
* top_dir: SDK根目录，安装SDK后会自动配置

如, 在x86平台、pcie工作模式下进行编译:
``` bash
make PRODUCTFORM=pcie
```
编译后生成可执行文件<strong> test_ocv_jpubasic </strong>

## 如何运行
``` bash
test_ocv_jpubasic <file> <loop> <yuv_enable> <dump_enable> [card]
```
参数说明:
* file: 输入图像文件
* loop: 解码循环次数, 输出N个文件
* yuv_enable: 是否输出yuv格式, 0: 输出BGR格式, 1: 输出yuv格式
* dump_enable: 是否生成dump文件, 0: 不生成dump文件, 1: 生成dump文件
* card: 使用的芯片序号(对应/dev下的设备文件序号)

命令示例:
``` bash 
> ./test_ocv_jpubasic ~/multimedia_files/1088test1_420.jpg 3 1 1 1
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
Test case 1
Open /dev/bm-sophon1 successfully, device index = 1, jpu fd = 4, vpp fd = 4
decoder time(ms): 25.7897

Test case 2

Test case 3
encoder time(ms): 3.93799
[/home/jenkins/workspace/all_in_one_sa5_daily/daily_build/bmetc/sa5/middleware-soc/bm_opencv/modules/core/src/cv_bmcpu.cpp:113->~InternalBMCpuRegister]deconstructor function is called
```