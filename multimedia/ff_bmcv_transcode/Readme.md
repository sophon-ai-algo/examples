## 功能介绍
用于测试ffmpeg视频转码功能及ffmpeg与bmcv之间的相互转换

## 如何编译

可选编译参数:
 * DEBUG: 是否输出调试信息，默认不输出(DEBUG=0)
 * PRODUCTFORM: 芯片工作模式，支持soc(独立主机)、pcie(x86平台)、arm_pcie(arm平台)选项，请根据使用环境指定
 * top_dir: SDK根目录，安装SDK后会自动配置

如, 在x86平台、pcie工作模式下进行编译:
``` bash
make PRODUCTFORM=pcie
```
编译后生成可执行文件<strong> test_ff_bmcv_transcode</strong>

## 如何运行
``` bash
test_ff_bmcv_transcode [src_filename] [output_filename] [encode_pixel_format] [codecer_name] [width] [height] [frame_rate] [bitrate] [thread_num] [zero_copy] [sophon_idx]
```
参数说明:
 * src_filename: 输入视频文件路径
 * output_filename: 输出视频文件路径
 * encode_pixel_format: 转码格式, 支持<strong>I420</strong>(yuv420p格式)、<strong>NV12</strong>(nv12格式)选项
 * encoder_name: 编码器选择, 使用硬件加速, 支持<strong>h264_bm</strong>、<strong>h265_bm</strong>选项
 * width: 输出视频宽度, 256 <= width <= 4096<strong><em> *** NV12格式下继承原视频的宽度 *** </em></strong>
 * height: 输出视频高度, 128 <= hieght <= 4096<strong><em> *** NV12格式下继承原视频的高度 *** </em></strong>
 * frame_rate: 输出视频帧率(FPS)
 * bitrate: 输出视频比特率, 500 < bitrate < 10000, 单位kbps
 * thread_num: 输出视频路数, 最大256路
 * zero_copy: 是否拷贝到host memory, 0: 拷贝, 1: 不拷贝<strong><em> *** 只在pcie模式下支持 *** </em></strong>
 * sophon_idx: 使用的芯片序号(对应/dev下的设备文件序号)<strong><em> *** 只在pcie模式下支持 *** </em></strong>

命令示例:
``` bash 
> ./test_ff_bmcv_transcode ~/multimedia_files/h265_1920x1080_30fps_1024k_main.mp4 test.ts I420 h265_bm 1920 1080 30 1024 1 0 0
...
> ffmpeg -i test0.ts -hide_banner
Input #0, mpegts, from 'test0.ts':
  Duration: 00:05:00.00, start: 0.200000, bitrate: 1136 kb/s
  Program 1 
    Metadata:
      service_name    : Service01
      service_provider: FFmpeg
    Stream #0:0[0x100]: Video: hevc (Main) ([36][0][0][0] / 0x0024), nv12(tv), 1920x1080, 60 fps, 30 tbr, 90k tbn, 60 tbc

```

``` bash
> ./test_ff_bmcv_transcode ~/multimedia_files/test-long.h265.mp4 test-nv.mp4 NV12 h265_bm 256 128 60 5000 3 1 1
...
> ffmpeg -i test-nv0.mp4 -hide_banner
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'test-nv0.mp4':
  Metadata:
    major_brand     : isom
    minor_version   : 512
    compatible_brands: isomiso2mp41
    encoder         : Lavf58.20.100
  Duration: 00:04:05.53, start: 0.000000, bitrate: 5007 kb/s
    Stream #0:0(und): Video: hevc (Main) (hev1 / 0x31766568), nv12(tv), 1920x1080, 5001 kb/s, 60 fps, 60 tbr, 15360 tbn, 120 tbc (default)
    Metadata:
      handler_name    : VideoHandler

```
