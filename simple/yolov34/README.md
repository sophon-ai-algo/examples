# 1. Introduction

This is a simple demo to run yolov3/yolov4 with BMNNSDK2.

# 2. Usage

- init bmnnsdk2 first: please refer to [BMNNSDK2 Introduction Notes](https://sophgo-doc.gitbook.io/bmnnsdk2-bm1684/bmnnsdk2/setup)
- Remember to use your own anchors, mask and classes number config values in `cpp/yolov3.hpp` and `python/configs/*.yml`
- Deploy on SE5, remember to [set the environmental variables](https://sophgo-doc.gitbook.io/bmnnsdk2-bm1684/bmnnsdk2/setup/on-soc#1.5.3.3-yun-hang-huan-jing-pei-zhi)
- For INT8 BModel, do not forget the scale factors for input and output tensors

## 2.1 prepare test data

use `scripts/0_prepare_test_data.sh` to download test images data and videos data to `data/`

```bash
cd scripts
bash ./0_prepare_test_data.sh
```

## 2.2 prepare bmodel

### 2.2.1 fp32 bmodel

use `scripts/gen_fp32bmodel.sh` to generate fp32 bmodel from `.cfg` and `.weights`, and `.bmodel` will be saved in `../data/models/`:

```bash
cd scripts
bash ./gen_fp32bmodel.sh
```

use `bmnetd` in BMNNSDK dev docker to generate other fp32 bmodels.

### 2.2.2 int8 bmodel

Follow the instructions in [Quantization-Tools User Guide](https://doc.sophgo.com/docs/docs_latest_release/calibration-tools/html/index.html) to generate int8 bmodel, the typical steps are:

- use `ufwio.io` to generate LMDB from images
- use `bmnetd --mode=GenUmodel` to generate fp32 umodel from `.cfg` and `.weights`
- use `calibration_use_pb quantize` to generate int8 umodel from fp32 umodel
- use `bmnetu` to generate int8 bmodel from int8 umodel

### 2.2.3 prepared bmodels

Several bmodels converted from [darknet](https://github.com/AlexeyAB/darknet) yolov3/yolov4  trained on [MS COCO](http://cocodataset.org/#home) are provided.

> Download bmodels from [here](http://219.142.246.77:65000/sharing/SJ4gY5Nzz) and put them in `data/models/` directory
>

| 模型文件                       | 输入                                                | 输出                                                         | anchors and masks                                            |
| ------------------------------ | --------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| yolov3_416_coco_fp32_1b.bmodel | input: data, [1, 3, 416, 416], float32, scale: 1    | output: Yolo0, [1, 255, 13, 13], float32, scale: 1<br/>output: Yolo1, [1, 255, 26, 26], float32, scale: 1<br/>output: Yolo2, [1, 255, 52, 52], float32, scale: 1 | YOLO_MASKS: [6, 7, 8, 3, 4, 5, 0, 1, 2]<br />YOLO_ANCHORS: [10, 13, 16, 30, 33, 23, 30, 61, 62, 45,59, 119, 116, 90, 156, 198, 373, 326] |
| yolov3_416_coco_int8_1b.bmodel | input: data, [1, 3, 416, 416], int8, scale: 128     | output: Yolo0, [1, 255, 13, 13], float32, scale: 0.0078125<br/>output: Yolo1, [1, 255, 26, 26], float32, scale: 0.0078125<br/>output: Yolo2, [1, 255, 52, 52], float32, scale: 0.0078125 | YOLO_MASKS: [6, 7, 8, 3, 4, 5, 0, 1, 2]<br />YOLO_ANCHORS: [10, 13, 16, 30, 33, 23, 30, 61, 62, 45,59, 119, 116, 90, 156, 198, 373, 326] |
| yolov3_608_coco_fp32_1b.bmodel | input: data, [1, 3, 608, 608], float32, scale: 1    | output: Yolo0, [1, 255, 19, 19], float32, scale: 1<br/>output: Yolo1, [1, 255, 38, 38], float32, scale: 1<br/>output: Yolo2, [1, 255, 76, 76], float32, scale: 1 | YOLO_MASKS: [6, 7, 8, 3, 4, 5, 0, 1, 2]<br />YOLO_ANCHORS: [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326] |
| yolov4_416_coco_fp32_1b.bmodel | input: data, [1, 3, 416, 416], float32, scale: 1    | output: Yolo0, [1, 255, 52, 52], float32, scale: 1<br/>output: Yolo1, [1, 255, 26, 26], float32, scale: 1<br/>output: Yolo2, [1, 255, 13, 13], float32, scale: 1 | YOLO_MASKS: [0, 1, 2, 3, 4, 5, 6, 7, 8]<br />YOLO_ANCHORS: [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401] |
| yolov4_608_coco_fp32_1b.bmodel | input: data, [1, 3, 608, 608], float32, scale: 1    | output: Yolo0, [1, 255, 76, 76], float32, scale: 1<br/>output: Yolo1, [1, 255, 38, 38], float32, scale: 1<br/>output: Yolo2, [1, 255, 19, 19], float32, scale: 1 | YOLO_MASKS: [0, 1, 2, 3, 4, 5, 6, 7, 8]<br /> YOLO_ANCHORS: [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401] |
| yolov4_608_coco_int8_1b.bmodel | input: data, [1, 3, 608, 608], int8, scale: 127.986 | output: Yolo0, [1, 255, 76, 76], float32, scale: 0.0078125<br/>output: Yolo1, [1, 255, 38, 38], float32, scale: 0.0078125<br/>output: Yolo2, [1, 255, 19, 19], float32, scale: 0.0078125 | YOLO_MASKS: [0, 1, 2, 3, 4, 5, 6, 7, 8]<br /> YOLO_ANCHORS: [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401] |

## 2.3 cpp demo usage

> Notes：`cpp/yolov3.hpp` uses "{ 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401}" as the default anchors and 80 as the default classes number. Remember to modify the values for your own model. 

For more detailed instructions, refer to [yolov3.v4-HowTO.pdf](docs/yolov3.v4-HowTO.pdf) .

### 2.3.1 for x86 with SC5

- compile the application in bmnnsdk2 dev docker

```shell
$ cd cpp/cpp_cv_bmcv_bmrt_postprocess
$ make -f Makefile.pcie # will generate yolo_test.pcie
```

- then put yolo_test.pcie and data dir on pcie host with bm1684

```shell
$ realpath ../../data/images/* > imagelist.txt
$ ./yolo_test.pcie image imagelist.txt ../../data/models/yolov4_608_coco_fp32.bmodel 

# USAGE:
#  ./yolo_test.pcie image <image list> <bmodel file> 
#  ./yolo_test.pcie video <video list>  <bmodel file>
```

### 2.3.2 for arm SE5/SM5
- compile the application in bmnnsdk2 dev docker


```shell 
$ cd cpp/cpp_cv_bmcv_bmrt_postprocess
$ make -f Makefile.arm # will generate yolo_test.arm
```
- then put yolo_test.arm and data dir on SE5

```shell
$ realpath ../../data/images/* > imagelist.txt
$ ./yolo_test.arm image imagelist.txt ../../data/models/yolov4_608_coco_fp32.bmodel 

# USAGE:
#  ./yolo_test.arm image <image list> <bmodel file> 
#  ./yolo_test.arm video <video list>  <bmodel file>
```

## 2.4 python demo usage

> Notes：For Python codes,  create your own config file *.yml in `configs` based on the values of `ENGINE_FILE`, `LABEL_FILE `, `YOLO_MASKS`, `YOLO_ANCHORS`, `OUTPUT_TENSOR_CHANNELS` for your model.

### 2.4.1 for x86 with SC5 & arm SE5/SM5

#### Installations

- for x86 SC5, the environment variable is set when `source envsetup_pcie.sh`, you need to install SAIL

```bash
# for example, python3.7 in docker
cd /workspace/lib/sail/python3/pcie/py37
pip3 install sophon-2.7.0-py3-none-any.whl
```

- for arm SE5, you need to set the environment variable:

```bash
# set the environment variable
export PATH=$PATH:/system/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/system/lib/:/system/usr/lib/aarch64-linux-gnu
export PYTHONPATH=$PYTHONPATH:/system/lib
```

you probably need to install NumPy, then you could use OpenCV and SAIL:

```bash
# please specify numpy version 1.17.2
sudo apt update
sudo apt-get install python3-pip
sudo pip3 install numpy==1.17.2
```

#### Usages

``` shell
$ cd python
$ python3 main.py # default: --cfgfile=configs/yolov3_416.yml --input=../data/images/person.jpg
#$ python3 main.py --cfgfile=<config file> --input=<image file path>
#$ python3 main.py --help # show help info
```
