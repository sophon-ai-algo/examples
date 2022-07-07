# RetinaFace

## 目录

* [RetinaFace](#RetinaFace)
  * [目录](#目录)
  * [1. 简介](#1-简介)
  * [2. 数据集](#2-数据集)
  * [3. 准备环境与数据](#3-准备环境与数据)
    * [3.1 准备环境](#31-准备开发环境)
    * [3.2 准备模型与数据](#32-准备模型与数据)
  * [4. 模型转换](#4-模型转换)
    * [4.1 生成FP32 BModel](#41-生成fp32-bmodel)
    * [4.2 生成INT8 BModel](#42-生成int8-bmodel)
  * [5. 推理测试](#5-推理测试)
    * [5.1 环境配置](#51-环境配置)
    * [5.2 C++例程推理](#52-C++例程推理)
    * [5.3 Python例程推理](#53-Python例程推理)

## 1. 简介
Retinaface Face Detect Demos

## 2. 数据集
待整理
## 3. 准备环境与数据
### 3.1 准备开发环境
转换模型前需要进入docker环境，切换到sdk根目录，启动docker容器：  
- 从宿主机SDK根目录下执行脚本进入docker环境  
```
./docker_run_<***>sdk.sh
```
- 在docker容器内安装SDK及设置环境变量
```bash
# 在docker容器内执行
cd $REL_TOP/scripts
# 安装库
./install_lib.sh nntc
# 设置环境变量，注意此命令只对当前终端有效，重新进入需要重新执行
source envsetup_pcie.sh    # for PCIE MODE
# source envsetup_cmodel.sh  # for CMODEL MODE
```
### 3.2 准备模型与数据
可以使用scripts下脚本文件prepare.sh从nas网盘下载模型和数据：

```bash
./scripts/prepare.sh
```
脚本将原始的retinaface_mobilenet0.25.onnx和编译好的retinaface_mobilenet0.25_384x640_fp32_b1.bmodel、retinaface_mobilenet0.25_384x640_fp32_b4.bmodel下载至data/models/文件夹。如果需要resnet50模型请通过nas网盘链接下载：http://219.142.246.77:65000/sharing/m6cELCAx7

同时会下载测试视频station.avi放至data/videos文件夹。

## 4. 模型转换
模型转换的过程需要在x86下的docker开发环境中完成。以下操作均在x86下的docker开发环境中完成。
### 4.1 生成fp32 bmodel
[3.2](#32-准备模型与数据)已将编译好的retinaface_mobilenet0.25_384x640_fp32_b1.bmodel、retinaface_mobilenet0.25_384x640_fp32_b4.bmodel下载放至data/models/文件夹。

转换脚本待整理

### 4.2 生成INT8 BModel
转换脚本待整理

## 5. 推理测试
### 5.1 环境配置
#### 5.1.1 x86 PCIe
对于x86平台PCIe，运行环境与开发环境是一致的，可以参考[3.1](#31-准备开发环境)在PCIE模式下配置运行环境。

由于Python例程用到sail库，需安装Sophon Inference：

```bash
# 确认平台及python版本，然后进入相应目录，比如x86平台，python3.7
pip3 install $REL_TOP/lib/sail/python3/pcie/py37/sophon-*-py3-none-any.whl

#### 5.1.2 arm SoC
对于arm SoC平台，内部已经集成了相应的SDK运行库包，位于/system目录下，只需设置环境变量即可。

```bash
# 设置环境变量
export PATH=$PATH:/system/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/system/lib/:/system/usr/lib/aarch64-linux-gnu
export PYTHONPATH=$PYTHONPATH:/system/lib
```

如果您使用的设备是Debian系统，您可能需要安装numpy包，以在Python中使用OpenCV和SAIL：

```bash
# 对于Debian9，请指定numpy版本为1.17.2
sudo apt update
sudo apt-get install python3-pip
sudo pip3 install numpy==1.17.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果您使用的设备是Ubuntu20.04系统，系统内已经集成了numpy环境，不需要进行额外的安装。

### 5.2 C++例程推理

#### 5.2.1 x86 PCIe
- 编译

```bash
$ cd cpp
$ make -f Makefile.pcie # 生成face_test
```

- 测试
```bash
# 图片模式，1batch，fp32
# imagelist.txt的每一行是图片的路径
# 如果模型是多batch的，会每攒够batch数的图片做一次推理
$ ./face_test 0 ../data/images/imagelist.txt ../data/models/retinaface_mobilenet0.25_384x640_fp32_b1.bmodel

# 视频模式，1batch，fp32
# videolist.txt的每一行是一个mp4视频路径或者一个rtsp url
# videolist.txt的视频数和模型的batch数相等
$ ./face_test 1 ../data/videos/videolist.txt  ../data/models/retinaface_mobilenet0.25_384x640_fp32_b1.bmodel
```
执行完毕后，会在当前目录生成一个名为result_imgs的文件夹，里面可以看到结果图片。

#### 5.2.2 arm SoC

对于arm平台SoC，需要在docker开发容器中使用交叉编译工具链编译生成可执行文件，而后拷贝到Soc目标平台运行。

- 在docker开发容器中交叉编译
```bash
$ cd cpp
$ make -f Makefile.arm # 生成face_test
```

- 将生成的可执行文件及所需的模型和测试图片或视频文件拷贝到盒子中测试，测试命令同上。

### 5.3 Python例程推理

Python代码无需编译，无论是x86 PCIe平台还是arm SoC平台配置好环境之后就可直接运行。但需要安装第三方库：
``` shell
$ cd python
$ pip3 install -r requirements.txt
```

```
#### 5.3.1  测试命令
- 查看测试命令参数
``` shell
# 使用opencv做前处理
$ python3 retinaface_sophon_opencv.py --help
# 使用bmcv做前处理
$ python3 retinaface_sophon_bmcv.py --help
```

- 测试图片

``` shell
# 使用opencv做前处理
$ python3 retinaface_sophon_opencv.py --bmodel ../data/models/retinaface_mobilenet0.25_384x640_fp32_b1.bmodel --network mobile0.25 --input ../data/images/face1.jpg --tpu_id 0 --conf 0.02 --nms 0.3 --use_np_file_as_input False
# 使用bmcv做前处理
$ python3 retinaface_sophon_bmcv.py --bmodel ../data/models/retinaface_mobilenet0.25_384x640_fp32_b1.bmodel --network mobile0.25 --input ../data/images/face1.jpg --tpu_id 0 --conf 0.02 --nms 0.3 --use_np_file_as_input False
```
测试结束后会将预测图片保存至result_imgs目录下，并打印相关测试时间如下：
``` shell
+--------------------------------------------------------------------------------+
|                           Running Time Cost Summary                            |
+------------------------+----------+--------------+--------------+--------------+
|        函数名称        | 运行次数 | 平均耗时(秒) | 最大耗时(秒) | 最小耗时(秒) |
+------------------------+----------+--------------+--------------+--------------+
|     predict_numpy      |    1     |    0.082     |    0.082     |    0.082     |
| preprocess_with_opencv |    1     |    0.007     |    0.007     |    0.007     |
|      infer_numpy       |    1     |    0.008     |    0.008     |    0.008     |
|      postprocess       |    1     |    0.029     |    0.029     |    0.029     |
+------------------------+----------+--------------+--------------+--------------+
```
-  测试视频
``` shell
# 使用opencv做前处理
$ python3 retinaface_sophon_opencv.py --bmodel ../data/models/retinaface_mobilenet0.25_384x640_fp32_b1.bmodel --network mobile0.25 --input ../data/videos/station.avi --tpu_id 0 --conf 0.02 --nms 0.3 --use_np_file_as_input False
# 使用bmcv做前处理
$ python3 retinaface_sophon_bmcv.py --bmodel ../data/models/retinaface_mobilenet0.25_384x640_fp32_b1.bmodel --network mobile0.25 --input ../data/videos/station.avi --tpu_id 0 --conf 0.02 --nms 0.3 --use_np_file_as_input False
```
测试结束后会将预测图片保存至result_imgs目录下，并打印相关测试时间。