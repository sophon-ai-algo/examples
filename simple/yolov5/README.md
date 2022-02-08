# 1. Introduction

This is a simple demo to run YOLOv5 with BMNNSDK2

# 2. Usage

put this demo dir to the bmnnsdk dir and enter docker, and init bmnnsdk firstly

## 2.1 generate models

Checkout the original yolov5 from github and put it outside YOLOv5_object first.

export yolov5 offline model of torchscript format, the exported model will be in data/models
``` shell
$ script/gen_pytorch.sh [img_size] [batch_size] # by default, img_size=640 batch_size=1

```

generate fp32 bmodel for running on device, pytorch model must exist in data/models

``` shell 
$ script/gen_fp32_bmodel.sh [img_size] [batch_size] # by default, img_size=640 batch_size=1
```

generate fix8b bmodel for running on device, pytorch model must exist in data/models

``` shell
$ script/gen_fix8b_bmodel.sh dataset_dir [iteration] [img_size] [batch_size] # by default, iteration=1 img_size=640 batch_size=1
```


## 2.2 cpp demo usage:

### 2.2.1 for pcie platform

compile the application
```shell
$ cd cpp
$ make -f Makefile.pcie #will generate yolov5_demo.pcie
```

then put yolov5_demo.pcie and data dir on pcie host with bm1684

```shell 
 $ ./yolov5_demo.pcie
 $ ./yolov5_demo.pcie --input=path/to/image
 $ ./yolov5_demo.pcie --input=path/to/image --bmodel=xxx.bmodel # use your own yolov5 bmodel
 $ ./yolov5_demo.arm --input=../data/images/dance.mp4 --is_video=true # use video as input, and process all frames
 $ ./yolov5_demo.arm --input=../data/images/dance.mp4 --is_video=true --frame_num=4 # use video as input, and process the first 4 frames
 $ ./yolov5_demo.pcie --help # see detail help info
```

### 2.2.2 for SE5/arm
compile the application

```shell 
$ cd cpp
$ make -f Makefile.arm #will generate yolov5_demo.arm
```
then put yolov5_demo.arm and data dir on SE5
```shell
 $ ./yolov5_demo.arm
 $ ./yolov5_demo.arm --input=path/to/image
 $ ./yolov5_demo.arm --input=path/to/image --bmodel=xxx.bmodel # use your own yolov5 bmodel
 $ ./yolov5_demo.arm --input=../data/images/dance.mp4 --is_video=true # use video as input, and process all frames
 $ ./yolov5_demo.arm --input=../data/images/dance.mp4 --is_video=true --frame_num=4 # use video as input, and process the first 4 frames
 $ ./yolov5_demo.arm --help # see detail help info
```

## 3. python demo usage:



> For python, The model must satisfy the followingg conditions: 
>
> ​      input: [1, 3, image_size, image_size]
>
> ​      output: [1, box_num, 5+class_num] 



### 3.1 on the host with opencv and pytorch

``` shell
$ python3 python/pytorch.py
$ python3 python/pytorch.py -h #show help info
```

### 3.2 on the pcie host with bm1684
```shell
$ python3 python/sail.py 
$ python3 python/sail.py -h #show help info
```

### 3.3 on SE5

```shell
$ export PYTHONPATH=$PYTHONPATH:/system/lib
$ python3 python/sail.py 
$ python3 python/sail.py -h # show help info
```