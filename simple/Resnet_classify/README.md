
# Resnet

## 目录
* [Resnet](#Resnet)
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

    

## 1. 简介
待整理

## 2. 数据集
待整理

## 3. 准备环境与数据
### 3.1 准备开发环境
转换模型前需要进入docker环境，切换到sdk根目录，启动docker容器：  
- 从宿主机SDK根目录下执行脚本进入docker环境  
```
./docker_run_bmnnsdk.sh
```
- 在docker容器内安装SDK及设置环境变量
```
# 在docker容器内执行
cd $REL_TOP/scripts
# 安装库
./install_lib.sh nntc
# 设置环境变量，注意此命令只对当前终端有效，重新进入需要重新执行
source envsetup_pcie.sh    # for PCIE MODE
source envsetup_cmodel.sh  # for CMODEL MODE
```

### 3.2 准备模型与数据
可以使用scripts下脚本文件prepare.sh从nas网盘下载编译好的模型：

```bash
./scripts/prepare.sh
```

脚本将下载resnet50.int8.bmodel并放至data/models/下。

## 4. 模型转换
模型转换的过程需要在x86下的docker开发环境中完成。以下操作均在x86下的docker开发环境中完成。
### 4.1 生成fp32 bmodel
转换脚本待整理

### 4.2 生成INT8 BModel
转换脚本待整理
本例程可直接使用[3.2](#32-准备模型与数据)下载好的resnet50.int8.bmodel。

## 5. 推理测试
### 5.1 环境配置
#### 5.1.1 x86 SC5

对于x86 SC5平台，程序执行所需的环境变量执行`source envsetup_pcie.sh`时已经配置完成。

#### 5.1.2 arm SE5
对于arm SE5平台，内部已经集成了相应的SDK运行库包，位于/system目录下，只需设置环境变量即可。

```bash
# 设置环境变量
export PATH=$PATH:/system/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/system/lib/:/system/usr/lib/aarch64-linux-gnu
export PYTHONPATH=$PYTHONPATH:/system/lib
```

您可能需要安装numpy包，以在Python中使用OpenCV和SAIL：

```bash
# 请指定numpy版本为1.17.2
sudo apt update
sudo apt-get install python3-pip
sudo pip3 install numpy==1.17.2
```
### 5.2 C++例程推理
具体查看cpp目录下例程的README.md
