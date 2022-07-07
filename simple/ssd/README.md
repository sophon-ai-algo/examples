
# SSD

## 目录
* [SSD](#SSD)
  * [目录](##目录)
  * [1. 简介](#1-简介)
  * [2. 数据集](#2-数据集)
  * [3. 准备环境与数据](#3-准备环境与数据)
    * [3.1 准备开发环境](#31-准备开发环境)
    * [3.2 准备模型与数据](#32-准备模型与数据)
  * [4. 模型转换](#4-模型转换)
    * [4.1 生成FP32 BModel](#41-生成fp32-bmodel)
    * [4.2 生成INT8 BModel](#42-生成int8-bmodel)
  * [5. 推理测试](#5-推理测试)
    * [5.1 环境配置](#51-环境配置)
    * [5.2 C++例程推理](#52-C++例程推理)
    * [5.3 Python例程推理](#53-Python例程推理)
    

## 1. 简介
SSD300 Object Detect Demos

## 2. 数据集
VOC0712

## 3. 准备环境与数据
### 3.1 准备开发环境
转换模型前需要进入docker环境，切换到sdk根目录，启动docker容器：  
- 从宿主机SDK根目录下执行脚本进入docker环境  
```
./docker_run_<***>sdk.sh
```
- 在docker容器内安装SDK及设置环境变量
```
# 在docker容器内执行
cd $REL_TOP/scripts
# 安装库
./install_lib.sh nntc
# 设置环境变量，注意此命令只对当前终端有效，重新进入需要重新执行
source envsetup_pcie.sh    # for PCIE MODE
# source envsetup_cmodel.sh  # for CMODEL MODE
```

### 3.2 准备模型与数据
可以使用scripts下脚本文件prepare.sh从nas网盘下载原始SSD模型和相关数据：

```bash
./scripts/prepare.sh
```

脚本将在下载完的ssd300.caffemodel和ssd300_deploy.prototxt放至data/models/下。这两个文件就是我们所需的原始Caffe模型文件。
同时会下载量化数据data.mdb至data/images/lmdb/文件夹。

## 4. 模型转换
模型转换的过程需要在x86下的docker开发环境中完成。以下操作均在x86下的docker开发环境中完成。
### 4.1 生成fp32 bmodel
```bash
./scripts/gen_fp32bmodel.sh
```
执行成功后，会在data/models/fp32bmodel目录下生成ssd300_fp32_1b.bmodel、ssd300_fp32_4b.bmodel文件。

### 4.2 生成INT8 BModel
```bash
./scripts/gen_int8bmodel.sh
```
执行成功后，会在data/models/int8bmodel目录下生成ssd300_int8_1b.bmodel、ssd300_int8_4b.bmodel文件。

## 5. 推理测试
### 5.1 环境配置
#### 5.1.1 x86 PCIe

对于x86 PCIe平台，程序执行所需的环境变量执行`source envsetup_pcie.sh`时已经配置完成。

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

具体查看cpp目录下各例程的README.md

### 5.3 Python例程推理
具体查看python目录下各例程的README.md