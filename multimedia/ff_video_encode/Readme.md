## 功能介绍
用于测试ffmpeg视频编码功能

## 如何编译

可选编译参数:
* DEBUG: 是否输出调试信息，默认不输出(DEBUG=0)
* PRODUCTFORM: 芯片工作模式，支持soc(独立主机)、pcie(x86平台)、arm_pcie(arm平台)选项，请根据使用环境指定
* top_dir: SDK根目录，安装SDK后会自动配置

如, 在x86平台、pcie工作模式下进行编译:
``` bash
make PRODUCTFORM=pcie
```
编译后生成可执行文件<strong> test_ff_video_encode </strong>

## 如何运行
``` bash
test_ff_video_encode <input file> <output file> <encoder> <width> <height> <roi_enable> <input pixel format> <bitrate(kbps)> <frame rate> <sophon device index>
```
参数说明:
* input file: 输入视频文件
* output file: 输出视频文件
* encoder: 指定编码器, 使用硬件加速, 可选的编码器有<strong>H264(AVC编码)</strong>、<strong>H265(HEVC编码)</strong>
* width: 输出视频宽度, 256 <= width <= 8192
* height: 输出视频高度, 128 <= height <= 8192
* roi_enable: 是否启用roi编码, 0: 不启用, 1: 启用, 默认为0
* input pixel format: 输出视频格式, 支持<strong>I420(YUV格式)</strong>、<strong>NV12(NV格式)</strong>选项, 默认为I420
* bitrate: 输出视频比特率, 可指定 10 < bitrate <= 100000, 默认为30*width*height/8(使用H265编码器则为30*width*height/16)
* framerate: 输出视频帧率(FPS), 可指定 10 < framerate <= 60, 默认为30
* sophon device index: 使用的芯片序号(对应/dev下的设备文件序号)<strong><em> *** 只在pcie模式下支持 *** </em></strong>

命令示例:
``` bash 
> ./test_ff_video_encode ~/multimedia_files/h265_1920x1080_30fps_1024k_main.mp4 test.mp4 H265 1920 1080 0 NV12 1000 30 0
src/encoder.c:247 (bmvpu_enc_load)   INFO: libbmvpuapi version 1.0.0
[7f8bbd6d3380] src/enc.c:270 (vpu_EncInit)   sophon_idx 0, VPU core index 4
[7f8bbd6d3380] src/vdi.c:137 (bm_vdi_init)   [VDI] Open device /dev/bm-sophon0, fd=5
[7f8bbd6d3380] src/vdi.c:229 (bm_vdi_init)   [VDI] success to init driver 
[bd6d3380] src/common.c:108 (find_firmware_path)   vpu firmware path: /home/bitmain/sophonsdk_vMaster/scripts/../lib/decode/pcie/vpu_firmware/chagall.bin
[7f8bbd6d3380] src/vdi.c:137 (bm_vdi_init)   [VDI] Open device /dev/bm-sophon0, fd=5
[7f8bbd6d3380] src/vdi.c:229 (bm_vdi_init)   [VDI] success to init driver 
[7f8bbd6d3380] src/enc.c:1336 (vpu_InitWithBitcode)   reload firmware...
[7f8bbd6d3380] src/enc.c:2471 (Wave5VpuInit)   
VPU INIT Start!!!
[7f8bbd6d3380] src/enc.c:314 (vpu_EncInit)   VPU Firmware is successfully loaded!
[7f8bbd6d3380] src/enc.c:318 (vpu_EncInit)   VPU FW VERSION=0x0, REVISION=250327
[h265_bm @ 0x55cb5560cac0] width        : 1920
[h265_bm @ 0x55cb5560cac0] height       : 1080
[h265_bm @ 0x55cb5560cac0] pix_fmt      : nv12
[h265_bm @ 0x55cb5560cac0] sophon device: 0
No output from encoder
No output from encoder
No output from encoder
No output from encoder
No output from encoder
No output from encoder
No output from encoder
The end of file!
Flushing video encoder
Flushing video encoder
Flushing video encoder
Flushing video encoder
Flushing video encoder
Flushing video encoder
Flushing video encoder
Flushing video encoder
encode finish! 
#######VideoEnc_FFMPEG exit 
```