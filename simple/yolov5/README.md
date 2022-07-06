# YOLOv5

## 目录

* [YOLOv5](#YOLOv5)
  * [目录](#目录)
  * [1. 简介](#1-简介)
  * [2. 数据集](#2-数据集)
    * [2.1 测试数据](#2.1-测试数据)
    * [2.2 量化数据集](#2.2-量化数据集)
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

YOLOv5是非常经典的基于anchor的One Stage目标检测算法YOLO的改进版本，因其优秀的精度和速度表现，在工程实践应用中获得了非常广泛的应用。

**文档:** [YOLOv5文档](https://docs.ultralytics.com/)

**参考repo:** [yolov5](https://github.com/ultralytics/yolov5)

**实现repo：**[yolov5_demo](https://github.com/xiaotan3664/)

## 2. 数据集

### 2.1 测试数据

使用`scripts/01_prepare_test_data.sh`下载测试数据，下载完成后测试数据(图片和视频)将保存在`data`目录下：

```bash
cd scripts
bash ./01_prepare_test_data.sh
```

### 2.2 量化数据集

[MS COCO](http://cocodataset.org/#home)，是微软构建的一个包含分类、检测、分割等任务的大型的数据集。使用[yolov5](https://github.com/ultralytics/yolov5)基于COCO Detection 2017预训练好的80类通用目标检测模型。

> MS COCO提供了一些[API](https://github.com/cocodataset/cocoapi)，方便对数据集的使用和模型评估，您可以使用pip安装` pip3 install pycocotools`，并使用COCO提供的API进行下载。

## 3. 准备环境与数据


### 3.1 准备开发环境

开发环境是指用于模型转换或验证以及程序编译等开发过程的环境，目前只支持x86，需要使用我们提供的基于Ubuntu18.04的docker镜像。

运行环境是具备Sophon设备的平台上实际使用设备进行算法应用部署的环境，有PCIe加速卡、SM模组、SE边缘计算盒子等，所有运行环境上的BModel都是一样的，SDK中各模块的接口也是一致的。

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

- 开发docker基础镜像：[点击前往官网下载Ubuntu开发镜像](https://sophon.cn/drive/44.html)，请选择与SDK版本适配的docker镜像

- SDK软件包：[点击前往官网下载SDK软件包](https://sophon.cn/drive/45.html)，请选择与仓库代码分支对应的SDK版本


#### 3.1.3 创建docker开发环境：

- 安装工具

  ```bash
  sudo apt update
  sudo apt install unzip
  ```

- 加载docker镜像:

  ```bash
  unzip <docker_image_file>.zip
  cd <docker_image_file>
  docker load -i <docker_image>
  ```

- 解压缩SDK：

  ```bash
  unzip <sdk_zip_file>.zip
  cd <sdk_zip_file>/
  tar zxvf <sdk_file>.tar.gz
  ```

- 创建docker容器，SDK将被挂载映射到容器内部供使用：

  ```bash
  cd <sdk_path>/
  # 若您没有执行前述关于docker命令免root执行的配置操作，需在命令前添加sudo
  ./docker_run_<***>sdk.sh
  ```

- 进入docker容器中安装库：

  ```bash
  # 进入容器中执行
  cd  /workspace/scripts/
  ./install_lib.sh nntc
  ```

- 设置环境变量-[无PCIe加速卡]：

  ```bash
  # 配置环境变量,这一步会安装一些依赖库，并导出环境变量到当前终端
  # 导出的环境变量只对当前终端有效，每次进入容器都需要重新执行一遍，或者可以将这些环境变量写入~/.bashrc，这样每次登录将会自动设置环境变量
  source envsetup_cmodel.sh
  ```

- 设置环境变量-[有PCIe加速卡]：

  ```bash
  # 配置环境变量,这一步会安装一些依赖库,并导出环境变量到当前终端
  # 导出的环境变量只对当前终端有效，每次进入容器都需要重新执行一遍，或者可以将这些环境变量写入~/.bashrc，这样每次登录将会自动设置环境变量
  source envsetup_pcie.sh
  ```

- 安装python对应版本的sail包

  ```bash
  # the wheel package is in the SophonSDK:
  pip3 uninstall -y sophon
  # get your python version
  python3 -V
  # choose the same verion of sophon wheel to install
  # the following py3x maybe py35, py36, py37 or py38
  # for x86
  pip3 install ../lib/sail/python3/pcie/py3x/sophon-?.?.?-py3-none-any.whl --user
  ```

### 3.2 准备模型

从[yolov5 release](https://github.com/ultralytics/yolov5/releases/)下载所需的pt模型。

**注意：**YOLOv5有多个版本：1.0、2.0、3.0、3.1、4.0、5.0、6.0、6.1。YOLOv5不同版本的代码导出的YOLOv5模型的输出会有不同，主要取决于model/yolo.py文件中的class Detect的forward函数。根据不同的组合，可能会有1、2、3、4个输出的情况，v6.1版本默认会有4个输出。具体情况可参见docs目录下的说明文档《YOLOV5模型导出与优化.docx》。

| 模型名称 | [YOLOv5s v6.1](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt) |
| -------- | ------------------------------------------------------------ |
| 训练集   | MS COCO                                                      |
| 概述     | 80类通用目标检测                                             |
| 运算量   | 16.5 GFlops                                                  |
| 输入数据 | images, [batch_size, 3, 640, 640], FP32，NCHW，RGB planar    |
| 输出数据 | 339,  [batch_size, 3, 80, 80, 85], FP32<br />391,  [batch_size, 3, 40, 40, 85], FP32<br />443,  [batch_size, 3, 20, 20, 85], FP32<br />output, [batch_size, 25200, 85], FP32 |
| 其他信息 | YOLO_MASKS: [6, 7, 8, 3, 4, 5, 0, 1, 2]<br />YOLO_ANCHORS: [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326] |
| 前处理   | BGR->RGB、/255.0                                             |
| 后处理   | nms等                                                        |

#### 3.2.1 下载yolov5源码

```bash
# 在容器里, 以python3.7的docker为例
cd ${YOLOv5}

# 下载yolov5源码
git clone https://github.com/ultralytics/yolov5.git yolov5_github
# 切换到yolov5工程目录
cd yolov5_github
# 使用tag从远程创建本地v6.1 分支
git branch v6.1 v6.1

# 下载yolov5s v6.1版本
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt
```

#### 3.2.2 修改models/yolo.py

修改Detect类的forward函数的最后return语句，实现不同的输出

```python
# 以三个输出模型为例
    ....
    
    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
                
        return x if self.training else x                       # 3个输出
        # return x if self.training else (torch.cat(z, 1))     # 1个输出
        # return x if self.training else (torch.cat(z, 1), x)  # 4个输出
        
        ....
```

#### 3.2.3 导出JIT模型

SophonSDK中的PyTorch模型编译工具BMNETP只接受PyTorch的JIT模型（TorchScript模型）。

JIT（Just-In-Time）是一组编译工具，用于弥合PyTorch研究与生产之间的差距。它允许创建可以在不依赖Python解释器的情况下运行的模型，并且可以更积极地进行优化。在已有PyTorch的Python模型（基类为torch.nn.Module）的情况下，通过torch.jit.trace就可以得到JIT模型，如`torch.jit.trace(python_model, torch.rand(input_shape)).save('jit_model')`。BMNETP暂时不支持带有控制流操作（如if语句或循环）的JIT模型，因此不能使用torch.jit.script，而要使用torch.jit.trace，它仅跟踪和记录张量上的操作，不会记录任何控制流操作。这部分操作yolov5已经为我们写好，只需运行如下命令即可导出符合要求的JIT模型：

```bash
# 创建python虚拟环境virtualenv
pip3 install virtualenv
# 切换到虚拟环境
virtualenv -p python3 --system-site-packages env_yolov5
source env_yolov5/bin/activate

# 安装依赖
pip3 install -r requirements.txt
# 此过程遇到依赖冲突或者错误属正常现象

# 导出jit模型
python3 export.py --weights yolov5s.pt --include torchscript
# 退出虚拟环境
deactivate

# 将生成好的jit模型yolov5s.torchscript拷贝到${YOLOv5}/build文件夹下
mkdir ../build
cp yolov5s.torchscript ../build/yolov5s_coco_v6.1_3output.trace.pt

# 拷贝一份到${YOLOv5}/data/models文件夹下
mkdir ../data/models
cp yolov5s.torchscript ../data/models/yolov5s_coco_v6.1_3output.trace.pt

cd ..
```

上述脚本会在原始pt模型所在目录下生成导出的JIT模型，导出后可以修改模型名称以区分不同版本和输出类型，l例如`yolov5s_640_coco_v6.1_1output.trace.pt`表示仅带有1个融合后的输出的JIT模型。

同时，我们已经准备了转换好的JIT模型，也可以直接从[这里](http://219.142.246.77:65000/sharing/lrneolzC3)下载，放到`${YOLOv5}/build`文件夹下。

### 3.3 准备量化集

不量化模型可跳过本节。

量化集使用COCO Detection 2017的验证集，可在YOLOv5 release v1.0的附件中下载：

```bash
cd scripts
./00_prepare.sh
```

上述脚本会创建build目录，并在其中下载解压`coco2017val.zip`。若您要使用自己的数据集，请修改`scripts/model_info.sh`中的参数，或者根据自己的需求修改脚本文件即可。


## 4. 模型转换

模型转换的过程需要在x86下的docker开发环境中完成。以下操作均在x86下的docker开发环境中完成。下面我们以3个output的情况为例，介绍如何完成模型的转换。

### 4.1 生成FP32 BModel

执行以下命令，使用bmnetp编译生成FP32 BModel，请注意修改`model_info.sh`中的模型名称、生成模型目录和输入大小shapes、使用的量化LMDB文件目录、batch_size、img_size等参数：

```bash
./1_gen_fp32bmodel.sh
```

上述脚本会在`${YOLOv5}/data/models`下生成`yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel`文件，即转换好的FP32 BModel，使用`bm_model.bin --info`查看的模型具体信息如下：

```bash
bmodel version: B.2.2
chip: BM1684
create time: Wed Jul  6 22:13:38 2022

==========================================
net 0: [yolov5s]  static
------------
stage 0:
input: input.1, [1, 3, 640, 640], float32, scale: 1
output: 147, [1, 3, 80, 80, 85], float32, scale: 1
output: 148, [1, 3, 40, 40, 85], float32, scale: 1
output: 149, [1, 3, 20, 20, 85], float32, scale: 1
```

### 4.2 生成INT8 BModel

不量化模型可跳过本节。

INT8 BModel的生成请注意修改`model_info.sh`中的模型名称、生成模型目录和输入大小shapes、使用的量化LMDB文件目录、batch_size、img_size等参数。

执行以下命令，使用一键量化工具cali_model，生成INT8 BModel：

```shell
./2_gen_int8bmodel.sh
```

上述脚本会在`${YOLOv5}/data/models`下生成`yolov5s_640_coco_v6.1_3output_int8_1b.bmodel`，即转换好的INT8 BModel，使用`bm_model.bin --info`查看的模型具体信息如下：

```bash
# for yolov5s_640_coco_v6.1_3output_int8_1b.bmodel
bmodel version: B.2.2
chip: BM1684
create time: Wed Jul  6 16:29:33 2022

==========================================
net 0: [yolov5s]  static
------------
stage 0:
input: input.1, [1, 3, 640, 640], int8, scale: 148.504
output: 147, [1, 3, 80, 80, 85], int8, scale: 0.152651
output: 148, [1, 3, 40, 40, 85], int8, scale: 0.119441
output: 149, [1, 3, 20, 20, 85], int8, scale: 0.106799
```

由于量化模型通常存在精度损失，当使用默认脚本生成的量化模型精度不能满足需求时，可能需要修改量化策略并借助自动量化工具auto-calib寻找最优结果，甚至在必要时需要将某些量化精度损失较大的层单独设置为使用fp32推理，相关调试方法请参考[《量化工具用户开发手册》](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/calibration-tools/html/index.html)。

> 注意：当使用1个output输出时，由于最后是把还原的box和score结果concat到一起去，而box和score的数值差异比较大（box中x，y取值范围是0-640， w和h相对小些， score的取值范围是0-1），在量化过程中会导致精度损失严重，可以考虑把box和score分别作为输出。相关内容参见docs目录下的说明文档《YOLOV5模型导出与优化.docx》。

## 5. 部署测试

请注意根据您使用的模型，修改`cpp/yolov5.cpp`或者`python/yolov5_your_script.py`中的anchors信息以及使用的`coco.names`文件；类别数量是根据模型输出Tensor的形状自动计算得到的，无需修改。

2.1节下载测试数据后，测试图片见`data/images`，测试视频见`data/videos`，转换好的bmodel文件可以放置于`data/models`。

> 已经转换好的bmodel文件可从[这里](http://219.142.246.77:65000/sharing/YtGpzqDfP)下载
>

### 5.1 环境配置

#### 5.1.1 x86 PCIe

对于x86 PCIe平台，程序执行所需的环境变量执行`source envsetup_pcie.sh`时已经配置完成

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

### 5.2 C++例程部署测试

C++例程适用于多种输出Tensor的情形。

#### 5.2.1 x86 PCIe

- 编译

```bash
$ cd ${YOLOv5}/cpp
$ make -f Makefile.pcie # 生成yolov5_demo.pcie
```

- 测试

```bash
 $ ./yolov5_demo.pcie --input=../data/images/dog.jpg --bmodel=../data/models/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel # use your own yolov5 bmodel
 # $ ./yolov5_demo.pcie --input=../data/videos/dance.mp4 --bmodel=../data/models/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel --is_video=true # use video as input, and process all frames
 # $ ./yolov5_demo.pcie --input=../data/videos/dance.mp4 --bmodel=../data/models/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel --is_video=true --frame_num=4 # use video as input, and process the first 4 frames
 # $ ./yolov5_demo.pcie --help # see detail help info
```

#### 5.2.2 arm SoC

对于arm SoC平台，需要在docker开发容器中使用交叉编译工具链编译生成可执行文件，而后拷贝到Soc目标平台运行。

- 在docker开发容器中交叉编译

```bash
$ cd ${YOLOv5}/cpp
$ make -f Makefile.arm # 生成yolov5_demo.arm
```

- 将生成的可执行文件及所需的模型和测试图片或视频文件拷贝到盒子中测试

```bash
 $ ./yolov5_demo.arm --input=../data/images/dog.jpg --bmodel=../data/models/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel # use your own yolov5 bmodel
 # $ ./yolov5_demo.arm --input=../data/videos/dance.mp4 --bmodel=../data/models/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel --is_video=true # use video as input, and process all frames
 # $ ./yolov5_demo.arm --input=../data/videos/dance.mp4 --bmodel=../data/models/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel --is_video=true --frame_num=4 # use video as input, and process the first 4 frames
 # $ ./yolov5_demo.arm --help # see detail help info
```

### 5.3 Python例程部署测试

Python代码无需编译，无论是x86 PCIe平台还是arm SoC平台配置好环境之后就可直接运行。

> 运行之前需要安装sail包

样例中提供了一系列例程以供参考使用，具体情况如下：

| #    | 样例文件                 | 说明                                                         |
| ---- | ------------------------ | ------------------------------------------------------------ |
| 1    | yolov5_bmcv_1output.py   | 使用SAIL解码、BMCV前处理、SAIL推理、OpenCV后处理，适用模型为1个输出 |
| 2    | yolov5_bmcv_3output.py   | 使用SAIL解码、BMCV前处理、SAIL推理、OpenCV后处理，适用模型为3个输出 |
| 3    | yolov5_opencv_1output.py | 使用OpenCV解码、OpenCV前处理、SAIL推理、OpenCV后处理，适用模型为1个输出 |
| 4    | yolov5_opencv_3output.py | 使用OpenCV解码、OpenCV前处理、SAIL推理、OpenCV后处理，适用模型为3个输出 |
| 5    | yolov5_pytorch.py        | 使用OpenCV读取图片和前处理、pytorch推理、OpenCV后处理        |

#### 5.3.1 x86平台PCIe模式

测试步骤如下：

```bash
# 在容器里, 以python3.7的docker为例
pip3 install /workspace/lib/sail/python3/pcie/py37/sophon-3.0.0-py3-none-any.whl

cd ${YOLOv5}/python

# yolov5_bmcv_3output.py使用3output的bmodel
python3 yolov5_bmcv_3output.py --bmodel ../data/models/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel --input ../data/images/dog.jpg
# yolov5_opencv_3output.py使用3output的bmodel
python3 yolov5_opencv_3output.py --bmodel ../data/models/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel --input ../data/images/dog.jpg

# yolov5_bmcv_1output.py使用1output的bmodel，需要自行转换1output的bmodel
python3 yolov5_bmcv_1output.py --bmodel ../data/models/yolov5s_640_coco_v6.1_1output_fp32_1b.bmodel --input ../data/images/dog.jpg
# yolov5_opencv_1output.py使用1output的bmodel，需要自行转换1output的bmodel
python3 yolov5_opencv_1output.py --bmodel ../data/models/yolov5s_640_coco_v6.1_1output_fp32_1b.bmodel --input ../data/images/dog.jpg

# pytorch推理,兼容1output和3output的trace后的jit模型
python3 yolov5_pytorch.py --model ../data/models/yolov5s_coco_v6.1_3output.trace.pt --img_size 640 --input ../data/images/dog.jpg
```

> **使用SAIL模块的注意事项：**对于INT8 BModel来说，当输入输出为int8时，含有scale，需要在处理时将输入输出乘以相应的scale。使用SAIL接口推理时，当sail.Engine.process()接口输入为numpy时，SAIL内部会自动乘以scale，用户无需操作；而输入为Tensor时，需要手动在数据送入推理接口前乘以scale。
>
> 这是因为Tensor作为输入的话，一般图像来源就是bm_image，这样就可以直接调用vpp进行scale等操作，所以推理之前由用户乘以scale更高效；而在python接口中，当numpy作为输入的话，推理之前没办法调用vpp，sail内部使用SSE指令进行了加速。

5.3.2 SE5智算盒SoC模式

> 将python文件夹和data文件夹拷贝到SE5中同一目录下

```bash
cd ${YOLOv5}/python

# yolov5_bmcv_3output.py使用3output的bmodel
python3 yolov5_bmcv_3output.py --bmodel ../data/models/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel --input ../data/images/dog.jpg
# yolov5_opencv_3output.py使用3output的bmodel
python3 yolov5_opencv_3output.py --bmodel ../data/models/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel --input ../data/images/dog.jpg

# yolov5_bmcv_1output.py使用1output的bmodel，需要自行转换1output的bmodel
python3 yolov5_bmcv_1output.py --bmodel ../data/models/yolov5s_640_coco_v6.1_1output_fp32_1b.bmodel --input ../data/images/dog.jpg
# yolov5_opencv_1output.py使用1output的bmodel，需要自行转换1output的bmodel
python3 yolov5_opencv_1output.py --bmodel ../data/models/yolov5s_640_coco_v6.1_1output_fp32_1b.bmodel --input ../data/images/dog.jpg
```

