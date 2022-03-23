# 1. Introduction

This is a demo to run CenterNet(backbone dlav0) object detection with BMNNSDK.

# 2. Usage

Put this demo dir into BMNNSDK docker container and init the environment of sdk firstly.

> Download the prepared bmodels from BaiDu Netdisk:
> access url: https://pan.baidu.com/s/1d3f8CjzC3BF2-2I2OF0q1g access code: lt59 

## 2.1 Generate bmodels
Change directory to `data/scripts`.
There is a `torchscript` file in this directory. It is made by weights file `ctdet_coco_dlav0_1x.pth` which is downloaded from [CenterNet model zoo](https://drive.google.com/drive/folders/1px-Xg7jXSC79QqgsD1AAGJQkuf5m0zh_)
To be attention, we use dlav0 as the backbone of CenterNet and we concatenate the heatmap, wh, offset output to be one. This means the output shape of the pt is `1x84x128x128`

```shell
cd data/scripts
```
### 2.1.1 fp32 bmodel

```shell
./gen_fp32_bmodel.sh
```

After a few minutes, we can get `ctdet_coco_dlav0_1x_fp32.bmodel` in `./models` directory

### 2.1.2 int8 bmodel
```shell
# usage: ./gen_int8_bmodel.sh <batch_size> <img_size> <validation_image_dir>
./gen_int_bmodel.sh 1 512 ../val2017
```
We choose about 200 picutures from val2017 to quantization and calibrate int8 bmodel.
You can use any picture you like.

After a few minutes, we get `ctdet_coco_dlav0_1x_int8_b1.bmodel` or `ctdet_coco_dlav0_1x_int8_b4.bmodel` depends on the `batch_size` you use.



## 2.2 python demo
```shell
cd py_bmcv_sail
python3 det_centernet_bmcv_1b_4b.py \
    --input=../data/ctdet_test.jpg \
    --bmodel=../data/models/ctdet_coco_dlav0_1x_int8_b4.bmodel \
    --tpu_id=0
```
This demo use bmcv for preprocess, sail for inference and numpy for postproces.
If success, we get result image like `ctdet_result_2022_-x-x-x-x-x_b_x.jpg` in current directory.

## 2.3 cpp demo
```shell
cd cpp_bmcv_sail
# compile in PCIE mode
make -f Makefile.pcie
./centernet_bmcv_sail.pcie \
    --bmodel=/workspace/examples/centernet_test/CenterNet_object/data/models/ctdet_coco_dlav0_1x_fp32.bmodel \
    --image=/workspace/examples/centernet_test/CenterNet_object/data/ctdet_test.jpg \
    --conf=0.35 \
    --tpu_id=0
# compile in Soc mode
# TODO
```
This demo use bmcv for preprocess, sail for inference and numpy for postproces.
If success, we get result image like `ctdet_result_2022_-x-x-x-x-x_b_x.jpg` in current directory.

> **WARNING**: 
The python demo only supports `1 batch or 4 batch` bmodel
The cpp demo only supports `1 batch` bmodel