## BMCV multi thread demo

* SOC

> make on x86 docker, but run on SOC.

```shell
make -f Makefile.arm

./ssd300_multi_bmcv_bmrt.arm video <video url>  <bmodel path> <video thread> <detect htread> <batch> <test loops> <yuv:1,bgr:0> <decode opencv:0,ffmpeg:1> <debug:0/1>



# video url：视频路径
# bmodel path： bmodel路径
# video thread: 解码线程的个数
# detect htread: 检测线程的个数
# batch: batch size，目前只支持1
# test loops: 循环次数
# yuv:1,bgr:0: 解码结果使用yuv（1）或者bgr（0）
# decode opencv:0,ffmpeg:1：解码器选择，使用opencv（0）或者ffmpeg(1)
# debug: 是否保存处理结果
# 结果保存名称为：detectThreadIDd_videoThreadIDv_image_id.bmp

```
