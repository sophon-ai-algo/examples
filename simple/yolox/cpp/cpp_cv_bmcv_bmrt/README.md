# demo for bmopencv decode + bmcv preprocess
## usage:
* x86 pcie

```shell
make -f Makefile.pcie
./yolox_cv_bmcv_bmrt.pcie image your_path/*.jpg  your_path/*.bmodel labelfile loops detect_threshold nms_threshold devid 

#./yolox_cv_bmcv_bmrt.pcie image ../data/images/dog.jpg ../data/models/yolox_s_4_int8.bmodel ../data/coco.names  16 0.25 0.45 0
# the output is in results with filename such as loop-loopidx-batch-batchidx-int8|fp32-dev-devid-dog.jpg, you can use any picture tool to open it


./yolox_cv_bmcv_bmrt.pcie video your_path/*.mp4  your_path/*.bmodel labelfile loops detect_threshold nms_threshold devid 

#./yolox_cv_bmcv_bmrt.pcie video ../data/videos/zhuheqiao_crop.mp4 ../data/models/yolox_s_4_int8.bmodel ../data/coco.names  16 0.25 0.45 0
# the output is in results with filename such as loop-loopidx-batch-batchidx-int8|fp32-dev-devid-video.jpg, you can use any picture tool to open it

* soc

```shell
make -f Makefile.arm
./yolox_cv_bmcv_bmrt.arm image your_path/*.jpg  your_path/*.bmodel labelfile loops detect_threshold nms_threshold devid 

#./yolox_cv_bmcv_bmrt.pcie image ../data/images/dog.jpg ../data/models/yolox_s_4_int8.bmodel ../data/coco.names  16 0.25 0.45 0
# the output is in results with filename such as loop-loopidx-batch-batchidx-int8|fp32-dev-devid-dog.jpg, you can use any picture tool to open it

./yolox_cv_bmcv_bmrt.arm video your_path/*.mp4  your_path/*.bmodel labelfile loops detect_threshold nms_threshold devid 

#./yolox_cv_bmcv_bmrt.arm video ../data/videos/zhuheqiao_crop.mp4 ../data/models/yolox_s_4_int8.bmodel ../data/coco.names  16 0.25 0.45 0
# the output is in results with filename such as loop-loopidx-batch-batchidx-int8|fp32-dev-devid-video.jpg, you can use any picture tool to open it
