## 功能介绍
用于测试opencv图像编解码功能

## 如何编译

可选编译参数:
* DEBUG: 是否输出调试信息，默认不输出(DEBUG=0)
* PRODUCTFORM: 芯片工作模式，支持soc(独立主机)、pcie(x86平台)、arm_pcie(arm平台)选项，请根据使用环境指定
* top_dir: SDK根目录，安装SDK后会自动配置

如, 在x86平台、pcie工作模式下进行编译:
``` bash
make PRODUCTFORM=pcie
```
编译后生成可执行文件<strong> test_ocv_jpumulti </strong>

## 如何运行
``` bash
test_ocv_jpumulti <test type> <inputfile> <loop> <num_threads> <outjpg> [card]
```
参数说明:
* test type: 选择测试功能, 1: 只解码, 2: 只编码, 3: 解码和编码(偶数路解码, 奇数路编码)
* inputfile: 输入图像文件
* loop: 测试循环次数
* num_threads: 输出路数, 最多12路
* outjpg: 是否输出到文件, 0: 不生成输出文件, 1: 生成输出文件
* card: 使用的芯片序号(对应/dev下的设备文件序号)

命令示例:
``` bash 
> ./test_ocv_jpumulti 3 ~/multimedia_files/1088test1_420.jpg 5 4 1 0
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
test type: 3   1-only dec  2-only enc  3-mix
input bitstream file: /home/bitmain/multimedia_files/1088test1_420.jpg
 thread input : /home/bitmain/multimedia_files/1088test1_420.jpg, test times = 5
 thread input : /home/bitmain/multimedia_files/1088test1_420.jpg, test times = 5
 thread input : /home/bitmain/multimedia_files/1088test1_420.jpg, test times = 5
 thread input : /home/bitmain/multimedia_files/1088test1_420.jpg, test times = 5
Open /dev/bm-sophon0 successfully, device index = 0, jpu fd = 6, vpp fd = 6
Decoder0 time(second): 0.033328
Decoder2 time(second): 0.033635
Encoder1 time(second): 0.028474
Encoder3 time(second): 0.028719
[/home/jenkins/workspace/all_in_one_sa5_daily/daily_build/bmetc/sa5/middleware-soc/bm_opencv/modules/core/src/cv_bmcpu.cpp:113->~InternalBMCpuRegister]deconstructor function is called
```