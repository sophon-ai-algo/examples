# YOLOv3/v4 demo for bmopencv decode + bmcv preprocess

## compile:
* x86 pcie

```shell
make -f Makefile.pcie

# yolo_test.pcie will be generated
```
* arm pcie

```shell
make -f Makefile.arm_pcie

# yolo_test.arm_pcie will be generated
```
* SOC

```shell
make -f Makefile.arm

# yolo_test.arm will be generated, copy the file to soc product and run
```
* x86 pcie

```shell
make -f Makefile.mips

# yolo_test.mips will be generated
```

## usage:

```shell
# image list or video url is a txt file with image path or video path in each line
# yolo_text.xxx differs on different platform
./yolo_test.xxx image <image list> <bmodel file> 
./yolo_test.xxx video <video url>  <bmodel file>

# result images with drawn prediction bboxes will be saved in result_imgs directory.
```