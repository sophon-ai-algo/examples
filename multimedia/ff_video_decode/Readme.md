## 功能介绍
用于测试ffmpeg视频解码功能

## 如何编译

可选编译参数:
* DEBUG: 是否输出调试信息，默认不输出(DEBUG=0)
* PRODUCTFORM: 芯片工作模式，支持soc(独立主机)、pcie(x86平台)、arm_pcie(arm平台)选项，请根据使用环境指定
* top_dir: SDK根目录，安装SDK后会自动配置

如, 在x86平台、pcie工作模式下进行编译:
``` bash
make PRODUCTFORM=pcie
```
编译后生成可执行文件<strong> test_ff_video_decode </strong>

## 如何运行
``` bash
test_ff_video_decode [yuv_format] [pre_allocation_frame] [codec_name] [sophon_idx] [zero_copy] [input_file/url] [input_file/url] ...
```
参数说明:
* yuv_format: 是否输出压缩数据, 0: 不压缩, 1: 压缩
* pre_allocation_frame: 缓存帧数, 最多64帧
* codec_name: 指定解码器, 使用硬件加速, 可选的解码器有<strong>h264_bm</strong>、<strong>hevc_bm</strong>, 使用<strong>no</strong>则不指定
* sophon_idx: 使用的芯片序号(对应/dev下的设备文件序号)<strong><em> *** 只在pcie模式下支持 *** </em></strong>
* zero_copy: 是否拷贝到host memory, 0: 拷贝, 1: 不拷贝<strong><em> *** 只在pcie模式下支持 *** </em></strong>
* input_file/url: 输入视频文件或码流, 可同时解码多路视频

命令示例:
``` bash 
> ./test_ff_video_decode 1 5 hevc_bm 0 0 ~/multimedia_files/h265_1920x1080_30fps_1024k_main.mp4
This is pcie module
reconnect stream[/home/bitmain/multimedia_files/h265_1920x1080_30fps_1024k_main.mp4] times:0.
[hevc_bm @ 0x7fce0401d880] bm decoder id: 12
[hevc_bm @ 0x7fce0401d880] sophon device: 0
[hevc_bm @ 0x7fce0401d880] don't copy back image? : 0
[hevc_bm @ 0x7fce0401d880] bm output format: 101
[hevc_bm @ 0x7fce0401d880] mode bitstream: 2, frame delay: -1
BMvidDecCreateW5 board id 0 coreid 0
[VDI] Open board 0, core 0, fd 5, dev /dev/bm-sophon0
libbmvideo.so addr : /home/bitmain/sophonsdk_vMaster/scripts/../lib/decode/pcie/libbmvideo.so, name_len: 59
vpu firmware addr: /home/bitmain/sophonsdk_vMaster/scripts/../lib/decode/pcie/vpu_firmware/chagall_dec.bin
[VDI] Open board 0, core 0, fd 5, dev /dev/bm-sophon0
VERSION=0, REVISION=213135
[hevc_bm @ 0x7fce0401d880] openDec video_stream_idx = 0, pix_fmt = 23
  0th thread process is 286.3985 fps!
  0th thread process is 291.0593 fps!
  0th thread process is 293.0248 fps!
  0th thread process is 293.0108 fps!
  0th thread process is 293.5187 fps!
  0th thread process is 293.1400 fps!
  0th thread process is 293.7719 fps!
  0th thread process is 294.2475 fps!
  0th thread process is 294.4899 fps!
  0th thread process is 294.4815 fps!
  0th thread process is 294.6588 fps!
  0th thread process is 294.6619 fps!
  0th thread process is 294.6422 fps!
  0th thread process is 294.6460 fps!
  0th thread process is 294.4564 fps!
  0th thread process is 294.5256 fps!
  0th thread process is 294.6207 fps!
  0th thread process is 294.6088 fps!
  0th thread process is 294.7200 fps!
  0th thread process is 294.7043 fps!
  0th thread process is 294.6211 fps!
  0th thread process is 294.8220 fps!
  0th thread process is 294.8669 fps!
  0th thread process is 294.7269 fps!
  0th thread process is 294.6678 fps!
  0th thread process is 294.6354 fps!
  0th thread process is 294.7342 fps!
  0th thread process is 294.8362 fps!
  0th thread process is 294.9313 fps!
  0th thread process is 294.8655 fps!
[hevc_bm @ 0x7fce0401d880] av_read_frame ret(-541478725) maybe eof...
no frame ! 
  0th thread Decode 9000 frame in total, avg: 294.8983, time: 30519ms!
#VideoDec_FFMPEG exit 
...
```