# 1. Introduction

This is a simple demo to run yolov3/yolov4 with BMNNSDK2

# 2. Usage

put this demo dir to the bmnnsdk dir and enter docker, and init bmnnsdk firstly

## 2.1 cpp demo usage:

### 2.1.1 for pcie platform

compile the application
```shell
$ cd cpp_cv_bmcv_bmrt_postprocess
$ make -f Makefile.pcie #will generate yolo_test.pcie

then put yolo_test.pcie and data dir on pcie host with bm1684

$ ./yolo_test.pcie image <imagelist.txt> <bmodelfile> 

> # For example: yolo_test.pcie image imagelist.txt ../data/models/yolov4_608_coco_fp32.bmodel

```

### 2.1.2 for SE5/arm
compile the application

```shell 
$ cd cpp_cv_bmcv_bmrt_postprocess
$ make -f Makefile.arm #will generate yolo_test.arm
```
then put yolo_test.arm and data dir on SE5
```shell
 $ ./yolo_test.arm image <imagelist.txt> <bmodelfile> 

> # For example: yolo_test.arm image imagelist.txt ../data/models/yolov4_608_coco_fp32.bmodel

```

## 3. python demo usage:

### 3.1 on the host with opencv and pytorch

``` shell
$ cd python
$ python3 main.py --cfgfile=configs/yolov4_608.yml --input <image file path>
$ python3 main.py --help #show help info

```
