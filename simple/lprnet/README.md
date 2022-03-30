# LPRNet

## 目录

* [LPRNet](#LPRNet)
  * [目录](#目录)
  * [1. 简介](#1-简介)
  * [2. 数据集](#2-数据集)
  * [3. 准备环境与数据](#3-准备环境与数据)
    * [3.1 准备环境](#31-准备环境)
    * [3.2 准备模型](#32-准备模型)
    * [3.3 准备量化集](#33-准备量化集)
  * [4. 模型转换](#4-模型转换)
    * [4.1 生成FP32 BModel](#41-生成fp32-bmodel)
    * [4.2 生成INT8 BModel](#42-生成int8-bmodel)
  * [5. 部署测试](#5-部署测试)
    * [5.1 环境配置](#51-环境配置)
    * [5.2 C++例程部署测试](#52-C++例程部署测试)
    * [5.3 Python例程部署测试](#53-Python例程部署测试)

## 1. 简介

LPRNet(License Plate Recognition via Deep Neural Networks)，是一种轻量级卷积神经网络，可实现无需进行字符分割的端到端自动车牌识别。LPRNet是第一种不使用回归神经网络的实时方法，足够轻量化使其可以在各种平台上运行，包括嵌入式设备。LPRNet在实际交通监控视频中的应用表明，可以处理多种困难的情况，例如透视和相机相关的失真，照明条件差和视点变化等。

**论文:** [LNRNet论文](https://arxiv.org/abs/1806.10447v1)

**参考repo:** [LNRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch)


## 2. 数据集

[CCPD](https://github.com/detectRecog/CCPD)，是由中科大团队构建的一个用于车牌识别的大型国内停车场车牌数据集。该数据集在合肥市的停车场采集得来，采集时间早上7:30到晚上10:00。停车场采集人员手持Android POS机对停车场的车辆拍照并手工标注车牌位置。拍摄的车牌照片涉及多种复杂环境，包括模糊、倾斜、阴雨天、雪天等等。CCPD数据集一共包含将近30万张图片，每种图片大小720x1160x3。

LNRNet_Pytorch中提供了一个节选至CCPD的车牌测试集，数量为1000张，图片名为车牌标签，且图片大小统一resize为24x94。

## 3. 准备环境与数据

### 3.1 准备开发环境

开发环境是指用于模型转换或验证以及程序编译等开发过程的环境，目前只支持x86，需要使用我们提供的基于Ubuntu16.04的docker镜像。

运行环境是具备Sophon设备的平台上实际使用设备进行算法应用部署的环境，有PCIe加速卡、SM5模组、SE5边缘计算盒子等，所有运行环境上的BModel都是一样的，SDK中各模块的接口也是一致的。

开发环境与运行环境可能是统一的（如插有SC5加速卡的x86主机，既是开发环境又是运行环境），也可能是分离的（如使用x86主机作为开发环境转换模型和编译程序，使用SE5盒子部署运行最终的算法应用）。

但是，无论使用的产品是SoC模式还是PCIe模式，都需要一台x86主机作为开发环境，模型的转换工作必须在开发环境中完成。

#### 3.1.1 开发主机准备：

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

#### 3.1.2 SDK软件包下载：

- 开发docker基础镜像：[点击前往官网下载Ubuntu开发镜像](https://sophon.cn/drive/44.html)，Ubuntu 16.04 with Python 3.7

```bash  
wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/22/03/19/13/bmnnsdk2-bm1684-ubuntu-docker-py37.zip
```

- SDK软件包：[点击前往官网下载SDK软件包](https://sophon.cn/drive/48.html)，bmnnsdk2_bm1684 2022.03.27

```bash
wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/22/03/27/23/bmnnsdk2_dailybuild_20220327.zip
```

> **注意：** LPRNet模型量化需在20220317之后版本的SDK中进行！

#### 3.1.3 创建docker开发环境：

- 加载docker镜像:

```bash
docker load -i bmnnsdk2-bm1684-ubuntu.docker
```

- 解压缩SDK：

```bash
tar zxvf bmnnsdk2-bm1684_v2.6.0.tar.gz
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
source envsetup_pcie.sh
```

### 3.2 准备模型

可通过运行`download.sh`将相关模型下载至`data/models`，将相关数据集下载至`data/images`。其中`data/models/Final_LPRNet_model.pth`为训练好的原始模型。

```bash
# 进入工程目录
cd  /workspace/examples/lprnet
./scripts/download.sh
```

#### 3.2.1 导出JIT模型

BMNNSDK2中的PyTorch模型编译工具BMNETP只接受PyTorch的JIT模型（TorchScript模型）。

JIT（Just-In-Time）是一组编译工具，用于弥合PyTorch研究与生产之间的差距。它允许创建可以在不依赖Python解释器的情况下运行的模型，并且可以更积极地进行优化。在已有PyTorch的Python模型（基类为torch.nn.Module）的情况下，通过torch.jit.trace就可以得到JIT模型，如`torch.jit.trace(python_model, torch.rand(input_shape)).save('jit_model')`。BMNETP暂时不支持带有控制流操作（如if语句或循环）的JIT模型，因此不能使用torch.jit.script，而要使用torch.jit.trace，它仅跟踪和记录张量上的操作，不会记录任何控制流操作。可在源码导入CPU模型后通过添加以下代码导出符合要求的JIT模型：

```python
....
# 导入CPU模型
lprnet.load_state_dict(torch.load("{PATH_TO_PT_MODEL}/Final_LPRNet_model.pth", map_location=torch.device('cpu')))
# jit.trace
model = torch.jit.trace(lprnet, torch.rand(1, 3, 24, 94))
# 保存JIT模型
torch.jit.save(model, "{PATH_TO_JIT_MODEL}/LPRNet_model.torchscript")
....
```

### 3.3 准备量化集

不量化模型可跳过本节。

量化集通过`download.sh`下载并解压至`data/images/test_md5`


## 4. 模型转换

模型转换的过程需要在x86下的docker开发环境中完成。以下操作均在x86下的docker开发环境中完成。

### 4.1 生成FP32 BModel

执行以下命令，使用bmnetp编译生成FP32 BModel，请注意修改`gen_fp32bmodel.sh`中的JIT模型路径、生成模型目录和输入大小shapes等参数：

```bash
cd scripts/
./gen_fp32bmodel.sh
```

上述脚本会在`fp32bmodel/`下生成`lprnet_fp32_1b.bmodel`文件，即转换好的FP32 BModel，使用`bm_model.bin --info`查看的模型具体信息如下：

```bash
bmodel version: B.2.2
chip: BM1684
create time: Thu Mar 24 20:35:20 2022

==========================================
net 0: [lprnet]  static
------------
stage 0:
input: x.1, [1, 3, 24, 94], float32, scale: 1
output: 237, [1, 68, 18], float32, scale: 1

device mem size: 5151696 (coeff: 1941776, instruct: 94464, runtime: 3115456)
host mem size: 0 (coeff: 0, runtime: 0)
```

### 4.2 生成INT8 BModel

不量化模型可跳过本节。

INT8 BModel的生成需要经历中间格式UModel，即：原始模型→FP32 UModel→INT8 UModel→INT8 BModel。执行以下命令，将生成INT8 BModel：

```shell
./gen_int8bmodel.sh
```

上述脚本会在`int8bmodel/`下生成`lprnet_int8_1b.bmodel`文件，即转换好的FP32 BModel，使用`bm_model.bin --info`查看的模型具体信息如下：

```bash
bmodel version: B.2.2
chip: BM1684
create time: Tue Mar 29 23:58:40 2022

==========================================
net 0: [LPRNet_model.torchscript_bmnetp]  static
------------
stage 0:
input: x.1, [1, 3, 24, 94], float32, scale: 1
output: 237, [1, 68, 18], float32, scale: 1

device mem size: 3232832 (coeff: 594944, instruct: 59008, runtime: 2578880)
host mem size: 0 (coeff: 0, runtime: 0)
```

## 5. 部署测试

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
sudo pip3 install numpy==1.17.2
```

### 5.2 C++例程部署测试

#### 5.2.1 x86平台SC5
工程目录下的cpp目录提供了一系列C++例程以供参考使用，具体情况如下：
| #    | 样例文件夹            | 说明                                 |
| ---- | -------------------- | -----------------------------------  |
| 1    | lprnet_cv_cv_bmrt    | 使用OpenCV解码、OpenCV前处理、BMRT推理 |
| 2    | lprnet_cv_bmcv_bmrt  | 使用OpenCV解码、BMCV前处理、BMRT推理   |

以lprnet_cv_cv_bmrt的编译及测试为例：

- 编译

```bash
cd cpp/lprnet_cv_cv_bmrt
make -f Makefile.pcie # 生成lprnet_cv_cv_bmrt.pcie
```

- 测试

编译完成后，会生成lprnet_cv_cv_bmrt.pcie,具体参数说明如下：

```bash
usage:./lprnet_cv_cv_bmrt.pcie <mode> <image path> <bmodel path> <device id>
mode:运行模型，可选择test或val，选择test时可将图片的推理结果打印出来，选择val时可将图片的推理结果打印出来并与标签进行对比，计算准确率，val只用于整个文件夹的推理且图片名以车牌标签命令；
image path:推理图片路径，可输入单张图片的路径，也可输入整个推理图片文件夹的路径；
bmodel path:用于推理的bmodel路径；
device id:用于推理的tpu设备id。
```

测试实例如下：

```bash
# 测试单张图片
./lprnet_cv_cv_bmrt.pcie test ../../data/images/test.jpg ../../data/models/lprnet_fp32_1b.bmodel 0
# 测试整个文件夹  
./lprnet_cv_cv_bmrt.pcie test ../../data/images/test/ ../../data/models/lprnet_fp32_1b.bmodel 0
# 测试整个文件夹，并计算准确率  
./lprnet_cv_cv_bmrt.pcie val ../../data/images/test/ ../../data/models/lprnet_fp32_1b.bmodel 0  
```

#### 5.2.2 arm平台SE5
对于arm平台SE5，需要在docker开发容器中使用交叉编译工具链编译生成可执行文件，而后拷贝到Soc目标平台运行。

- 在docker开发容器中交叉编译

```bash
cd cpp/lprnet_cv_cv_bmrt
make -f Makefile.arm  # 生成lprnet_cv_cv_bmrt.arm
```
- 将生成的可执行文件及所需的模型和测试图片拷贝到盒子中测试，测试方法与SC5相同。

### 5.3 Python例程部署测试

由于Python例程用到sail库，需安装Sophon Inference：

```bash
# 确认平台及python版本，然后进入相应目录，比如x86平台，python3.7
cd /workspace/lib/sail/python3/pcie/py37
pip3 install sophon-x.x.x-py3-none-any.whl
```

Python代码无需编译，无论是x86 SC平台还是arm SE5平台配置好环境之后就可直接运行。

工程目录下的python目录提供了一系列python例程以供参考使用，具体情况如下：

| #    | 样例文件                  | 说明                                 |
| ---- | ----------------------   | -----------------------------------  |
| 1    | lprnet_cv_cv_sail.py     | 使用OpenCV解码、OpenCV前处理、SAIL推理 |
| 2    | lprnet_sail_bmcv_sail.py | 使用SAIL解码、BMCV前处理、SAIL推理     |


> **使用SAIL模块的注意事项：** 对于INT8 BModel来说，当输入输出为int8时，含有scale，需要在处理时将输入输出乘以相应的scale。使用SAIL接口推理时，当sail.Engine.process()接口输入为numpy时，SAIL内部会自动乘以scale，用户无需操作；而输入为Tensor时，需要手动在数据送入推理接口前乘以scale。
> 这是因为Tensor作为输入的话，一般图像来源就是bm_image，这样就可以直接调用vpp进行scale等操作，所以推理之前由用户乘以scale更高效；而在python接口中，当numpy作为输入的话，推理之前没办法调用vpp，sail内部使用SSE指令进行了加速。

> **出现中文无法正常显示的解决办法**：Python例程在打印车牌时若出现中文无法正常显示，可参考以下操作进行解决：

```bash
# 1.安装中文支持包language-pack-zh-hans
apt install language-pack-zh-hans
# 2.修改/etc/environment，在文件的末尾追加：
LANG="zh_CN.UTF-8"
LANGUAGE="zh_CN:zh:en_US:en"
# 3.修改/var/lib/locales/supported.d/local，没有这个文件就新建，同样在末尾追加：
en_US.UTF-8 UTF-8
zh_CN.UTF-8 UTF-8
zh_CN.GBK GBK
zh_CN GB2312
# 4.最后，执行命令：
locale-gen
```

> **使用bm_opencv解码的注意事项：** lprnet_cv_cv_sail.py默认使用原生opencv解码和预处理，使用bm_opencv解码结果与原生opencv解码结果的差异可能会导致推理结果的差异，若要使用bm_opencv硬解码可在运行lprnet_cv_cv_sail.py时修改环境变量如下：

```bash
export PYTHONPATH=$PYTHONPATH:/workspace/lib/opencv/x86/opencv-python/
```


- 测试

以lprnet_cv_cv_sail.py的测试为例,具体参数说明如下：

```bash
usage:lprnet_cv_cv_sail.py [--mode MODE] [--img_path IMG_PATH] [--bmodel BMODEL] [--tpu_id TPU]
--mode:运行模型，可选择test或val，选择test时可将图片的推理结果打印出来，选择val时可将图片的推理结果打印出来并与标签进行对比，计算准确率，val只用于整个文件夹的推理且图片名以车牌标签命令；
--img_path:推理图片路径，可输入单张图片的路径，也可输入整个图片文件夹的路径；
--bmodel:用于推理的bmodel路径；
--tpu_id:用于推理的tpu设备id。
```

测试实例如下：
```bash
# 测试单张图片
python3 lprnet_cv_cv_sail.py --mode test --img_path ../data/images/test.jpg --bmodel ../scripts/fp32bmodel/lprnet_fp32_1b.bmodel --tpu_id 0  
# 测试整个文件夹
python3 lprnet_cv_cv_sail.py --mode test --img_path ../data/images/test --bmodel ../scripts/fp32bmodel/lprnet_fp32_1b.bmodel --tpu_id 0 
# 测试整个文件夹，并计算准确率 
python3 lprnet_cv_cv_sail.py --mode val --img_path ../data/images/test --bmodel ../scripts/fp32bmodel/lprnet_fp32_1b.bmodel --tpu_id 0  
```

使用原生opencv解码的测试结果与LNRNet_Pytorch测试精度一致：

| #      |  LNRNet_Pytorch    | lprnet_cv_cv_sail  |
| ------ | ----------------   | -----------------  |
| ACC    |       89.4%        |        89.4%       |

使用bm_opencv硬解码的测试精度如下：

| ACC                      |lprnet_fp32.bmodel| lprnet_int8.bmodel|
| ----------------------   | -------------    | --------------    |
| lprnet_cv_cv_sail.py     |      88%         |   87.7%           |
| lprnet_sail_bmcv_sail.py |      88.2%       |   87.4%           |
| cpp/lprnet_cv_cv_bmrt    |      88%         |   87.7%           |
| cpp/lprnet_cv_bmcv_bmrt  |      88%         |   87.7%           |