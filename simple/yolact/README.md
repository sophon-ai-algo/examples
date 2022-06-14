# YOLACT

## 目录

* [YOLACT](#YOLACT)
  * [目录](#目录)
  * [1. 简介](#1-简介)
  * [2. 数据集](#2-数据集)
    * [2.1 测试数据](#21-测试数据)
    * [2.2 量化数据集](#22-量化数据集)
  * [3. 准备环境与数据](#3-准备环境与数据)
    * [3.1 准备开发环境](#31-准备开发环境)
    * [3.2 准备模型](#32-准备模型)
    * [3.3 准备量化集](#33-准备量化集)
  * [4. 模型转换](#4-模型转换)
    * [4.1 生成JIT模型](#41-生成JIT模型)
    * [4.2 生成FP32 BModel](#42-生成fp32-bmodel)
    * [4.3 生成INT8 BModel](#43-生成int8-bmodel)
  * [5. 部署测试](#5-部署测试)
    * [5.1 环境配置](#51-环境配置)
    * [5.2 C++例程部署测试](#52-C++例程部署测试)
    * [5.3 Python例程部署测试](#53-Python例程部署测试)

## 1. 简介

YOLACT是一种实时的实例分割的方法。

论文地址: [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)

官方源码: https://github.com/dbolya/yolact

## 2. 数据集

### 2.1 测试数据

使用`scripts/01_prepare_test_data.sh`下载测试数据，下载完成后测试数据(图片和视频)将保存在`data`目录下：

```bash
cd scripts
bash ./01_prepare_test_data.sh
```

### 2.2 量化数据集

[MS COCO](http://cocodataset.org/#home)，是微软构建的一个包含分类、检测、分割等任务的大型的数据集。使用[yolact](https://github.com/dbolya/yolact)基于COCO Detection 2017预训练好的80类通用目标检测模型。

> MS COCO提供了一些[API](https://github.com/cocodataset/cocoapi)，方便对数据集的使用和模型评估，您可以使用pip安装` pip3 install pycocotools`，并使用COCO提供的API进行下载。

## 3 准备环境与数据

### 3.1 准备开发环境

开发环境是指用于模型转换或验证以及程序编译等开发过程的环境，目前只支持x86，需要使用我们提供的基于Ubuntu16.04的docker镜像。

运行环境是具备Sophon设备的平台上实际使用设备进行算法应用部署的环境，有PCIe加速卡、SM5模组、SE5边缘计算盒子等，所有运行环境上的BModel都是一样的，SDK中各模块的接口也是一致的。

开发环境与运行环境可能是统一的（如插有SC5加速卡的x86主机，既是开发环境又是运行环境），也可能是分离的（如使用x86主机作为开发环境转换模型和编译程序，使用SE5盒子部署运行最终的算法应用）。

但是，无论使用的产品是SoC模式还是PCIe模式，都需要一台x86主机作为开发环境，模型的转换工作必须在开发环境中完成。

#### 3.1.1开发主机准备：

- 开发主机：一台安装了Ubuntu16.04/18.04/20.04的x86主机，运行内存建议12GB以上

- 安装docker：参考《[官方教程](https://docs.docker.com/engine/install/)》，若已经安装请跳过

  ```bash
  # 安装docker
  sudo apt-get install docker.io
  # docker命令免root权限执行
  # 创建docker用户组，若已有docker组会报错，没关系可忽略
  sudo groupadd docker
  # 将当前用户加入docker组
  sudo gpasswd -a ${USER} docker
  # 重启docker服务
  sudo service docker restart
  # 切换当前会话到新group或重新登录重启X会话
  newgrp docker
  ```

#### 3.1.2 SDK软件包下载

- 开发docker基础镜像：[点击前往官网下载Ubuntu开发镜像](https://developer.sophgo.com/site/index/material/11/all.html)，Ubuntu 18.04 with Python 3.7

```bash
wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/22/03/19/13/bmnnsdk2-bm1684-ubuntu-docker-py37.zip
```

- SDK软件包：[点击前往官网下载SDK软件包](https://developer.sophgo.com/site/index/material/17/all.html)，BMNNSDK 2.7.0_20220316_022200

```bash
wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/22/04/14/10/bmnnsdk2_bm1684_v2.7.0_20220316_patched_0413.zip
```

#### 3.1.3 创建docker开发环境

- 加载docker镜像:

```bash
docker load -i bmnnsdk2-bm1684-ubuntu.docker
```

- 解压缩SDK：

```bash
tar zxvf bmnnsdk2-bm1684_v2.7.0.tar.gz
```

- 创建docker容器，SDK将被挂载映射到容器内部供使用：

```bash
cd bmnnsdk2-bm1684_v2.7.0
# 若您没有执行前述关于docker命令免root执行的配置操作，需在命令前添加sudo
./docker_run_bmnnsdk.sh
```

- 进入docker容器中安装库：

```bash
# 进入容器中执行
cd  /workspace/scripts/
./install_lib.sh nntc
```

- 设置环境变量：

```bash
# 配置环境变量，这一步会安装一些依赖库，并导出环境变量到当前终端
# 导出的环境变量只对当前终端有效，每次进入容器都需要重新执行一遍，或者可以将这些环境变量写入~/.bashrc，这样每次登录将会自动设置环境变量
source envsetup_cmodel.sh
```

### 3.2 准备模型

从[yolact](https://github.com/dbolya/yolact#evaluation)下载所需的pt模型或者从我们准备好的相同来源的[pt模型](http://219.142.246.77:65000/sharing/Ib5nkB32t)。

**注意：**由于[yolact](https://github.com/dbolya/yolact#evaluation)源码包含了训练部分代码和切片操作，需要将训练部分和切片操作代码去掉，提前返回features。我们提供了修改好的代码可以直接转换。**在[模型转换](#4-模型转换)章节，我们提供了从pt模型下载，转换bmodel模型步骤。详细模型转换可参考[模型转换](#4-模型转换)。**

#### 3.2.1导出JIT模型

BMNNSDK2中的PyTorch模型编译工具BMNETP只接受PyTorch的JIT模型（TorchScript模型）。

JIT（Just-In-Time）是一组编译工具，用于弥合PyTorch研究与生产之间的差距。它允许创建可以在不依赖Python解释器的情况下运行的模型，并且可以更积极地进行优化。在已有PyTorch的Python模型（基类为torch.nn.Module）的情况下，通过torch.jit.trace就可以得到JIT模型，如`torch.jit.trace(python_model, torch.rand(input_shape)).save('jit_model')`。BMNETP暂时不支持带有控制流操作（如if语句或循环）的JIT模型，因此不能使用torch.jit.script，而要使用torch.jit.trace，它仅跟踪和记录张量上的操作，不会记录任何控制流操作。以yolact_base_54_800000模型为例，只需运行如下命令即可导出符合要求的JIT模型：

```bash
cd scripts/converter
python3 ./convert.py --input ${MODEL_DIR}/yolact_base_54_800000.pth --mode tstrace --cfg yolact_base
```

上述脚本会在scripts/converter文件夹下生成`yolact_base_54_800000.trace.pt`的JIT模型。

### 3.3 准备量化集

coming soon.

## 4. 模型转换

模型转换的过程需要在x86下的docker开发环境中完成。以下操作均在x86下的docker开发环境中完成。下面我们以`yolact_base_54_800000`模型为例，介绍如何完成模型的转换。

### 4.1 生成JIT模型

将[3.2 准备模型](#3.2-准备模型)下载好的`yolact_base_54_800000.pth`放到`data/models`文件夹下，**或者**通过运行`download.sh`将相关模型下载至`data/models`，`data/models/yolact_base_54_800000.pth`为训练好的原始模型。

```bash
cd ${YOLACT}/scripts
./download.sh
```

上述脚本会下载好的`yolact_base_54_800000.pth`，并放到`data/models`文件夹下。

执行以下命令生成JIT模型：

```bash
./10_gen_tstracemodel.sh
```

上述脚本会在`data/models`文件夹下生成`yolact_base_54_800000.trace.pt`文件，即转换好的JIT模型，并放到`data/models`目录下。

### 4.2 生成FP32 BModel

执行以下命令，使用bmnetp编译生成FP32 BModel：

```bash
./11_gen_fp32bmodel.sh
```

上述脚本会在`data/models`文件夹下根据`yolact_base_54_800000.trace.pt`生成`yolact_base_54_800000_b1.bmodel`文件，即转换好的FP32 BModel，放在文件夹yolact_base_54_800000_fp32_b1下。使用`bm_model.bin --info ${BModel}`查看的模型具体信息如下：

```bash
bmodel version: B.2.2
chip: BM1684
create time: Mon May 16 16:05:58 2022

==========================================
net 0: [yolact_base]  static
------------
stage 0:
input: x.1, [1, 3, 550, 550], float32, scale: 1
output: proto_out0.1, [1, 138, 138, 32], float32, scale: 1
output: loc.1, [1, 19248, 4], float32, scale: 1
output: mask.1, [1, 19248, 32], float32, scale: 1
output: conf.1, [1, 19248, 81], float32, scale: 1

device mem size: 306055232 (coeff: 224407936, instruct: 1457664, runtime: 80189632)
host mem size: 0 (coeff: 0, runtime: 0)
```

### 4.3 生成INT8 BModel

不量化模型可跳过本节。

coming soon.

## 5. 部署测试

请注意根据您使用的模型，选择相应的`.cfg`文件。

测试图片见`data/images`，测试视频见`data/videos`，转换好的bmodel文件可以放置于`data/models`

已经转换好的bmodel文件可从以下链接下载：

链接: http://219.142.246.77:65000/sharing/1EDAWPfqh

### 5.1 环境配置

#### x86 SC5

对于x86 SC5平台，程序执行所需的环境变量执行`source envsetup_pcie.sh`时已经配置完成

#### arm SE5

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

### 5.2 C++例程部署测试

coming soon.

### 5.3 Python例程部署测试

Python代码无需编译，无论是x86 SC平台还是arm SE5平台配置好环境之后就可直接运行。

样例中提供了一系列例程以供参考使用，具体情况如下：

| #    | 样例文件          | 说明                                                  |
| ---- | ----------------- | ----------------------------------------------------- |
| 1    | yolact_bmcv.py    | 使用SAIL解码、BMCV前处理、SAIL推理、OpenCV后处理      |
| 2    | yolact_sail.py    | 使用OpenCV解码、OpenCV前处理、SAIL推理、OpenCV后处理  |
| 3    | yolact_pytorch.py | 使用OpenCV读取图片和前处理、pytorch推理、OpenCV后处理 |

测试

```bash
cd ${YOLACT}/python
# yolact_sail.py使用方法与yolact_bmcv.py一致
# 如果使用yolact_pytorch.py测试，<model>为JIT模型路径
# yoloact base
# image
python3 yolact_bmcv.py --cfgfile configs/yolact_base.cfg --model ../data/models/yolact_base_54_800000_fp32_b1/yolact_base_54_800000_b1.bmodel --input_path ../data/images/

# video
python3 yolact_bmcv.py --cfgfile configs/yolact_base.cfg --model ../data/models/yolact_base_54_800000_fp32_b1/yolact_base_54_800000_b1.bmodel --is_video 1 --input_path ../data/videos/road.mp4
```

> **使用SAIL模块的注意事项：**对于INT8 BModel来说，当输入输出为int8时，含有scale，需要在处理时将输入输出乘以相应的scale。使用SAIL接口推理时，当sail.Engine.process()接口输入为numpy时，SAIL内部会自动乘以scale，用户无需操作；而输入为Tensor时，需要手动在数据送入推理接口前乘以scale。
>
> 这是因为Tensor作为输入的话，一般图像来源就是bm_image，这样就可以直接调用vpp进行scale等操作，所以推理之前由用户乘以scale更高效；而在python接口中，当numpy作为输入的话，推理之前没办法调用vpp，sail内部使用SSE指令进行了加速。

