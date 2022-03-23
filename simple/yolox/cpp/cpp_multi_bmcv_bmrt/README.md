# demo for bmopencv decode + bmcv preprocess
## usage:

*SOC

# make in host
make -f Makefile.arm

# run in soc

```shell

  ./yolox_cv_bmcv_bmrt.arm video <video url> <bmodel path> <name file> <detect threshold> <nms threshold> <device id> <video thread> <detect thread> <save:0/1>


# video url：视频路径
# bmodel path： bmodel路径
# name file: namefile路径
# detect threshold: detect threshold
# nms threshold: nms threshold
# device id: device id
# video thread: 解码线程的个数
# detect htread: 检测线程的个数
# save: save flage, if 1,save result in path "./results"


```