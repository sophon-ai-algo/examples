# demo for bmopencv decode + bmcv preprocess
## usage:

* x86 pcie 

```shell
make -f Makefile.pcie

./yolox_bmcv_sail.pcie video your_path/*.mp4  your_path/*.bmodel labelfile loops detect_threshold nms_threshold devid 

#./yolox_bmcv_sail.pcie video /data/video/zhuheqiao.mp4 ../data/models/yolox_s_4_int8.bmodel ../data/coco.names 16 0.25 0.45 0
# the output with filename such as loop-loopidx-batch-batchidx-dev-devid.jpg, you can use any picture tool to open it


* soc (buidl in pcie, run in soc)

```shell
make -f Makefile.arm

./yolox_bmcv_sail.arm video your_path/*.mp4  your_path/*.bmodel labelfile loops detect_threshold nms_threshold devid 

#./yolox_bmcv_sail.arm video /data/video/zhuheqiao.mp4 ../data/models/yolox_s_4_int8.bmodel ../data/coco.names 16 0.25 0.45 0
# the output with filename such as loop-loopidx-batch-batchidx-dev-devid.jpg, you can use any picture tool to open it
```