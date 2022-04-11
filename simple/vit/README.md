# CLIP-ViT

## 目录

* [ViT](#ViT)
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

ViT算法中尝试将标准的Transformer结构直接应用于图像，并对整个图像分类流程进行最少的修改。具体来讲，ViT算法中，会将整幅图像拆分成小图像块，然后把这些小图像块的线性嵌入序列作为Transformer的输入送入网络，然后使用监督学习的方式进行图像分类的训练。

**文档:** [ViT文档](https://openai.com/blog/clip/)

**参考repo:** [ViT](https://github.com/openai/CLIP)


## 2. 数据集

[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)，CIFAR-10是一个更接近普适物体的彩色图像数据集。CIFAR-10 是由Hinton 的学生Alex Krizhevsky 和Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含10 个类别的RGB 彩色图片。

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

- 开发docker基础镜像：[点击前往官网下载Ubuntu开发镜像](https://sophon.cn/drive/44.html)，Ubuntu 16.04 with Python 3.5

  ```bash
  wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/22/01/18/10/bmnnsdk2-bm1684-ubuntu-docker-py35.zip
  ```

- SDK软件包：[点击前往官网下载SDK软件包](https://sophon.cn/drive/45.html)，BMNNSDK 2.6.0_20220130_042200

  ```bash
  wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/22/02/10/18/bmnnsdk2_bm1684_v2.6.0.zip
  ```

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
  cd bmnnsdk2-bm1684_v2.6.0
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

需要从[ViT](https://github.com/openai/CLIP)trace所需的pt模型。


| 模型名称 | [ViT](https://github.com/openai/CLIP) |
| -------- | ------------------------------------------------------------ |
| 训练集   | CIFAR100                                                      |
| 概述     | 使用监督学习的方式进行图像分类的训练网络                            |
| 运算量   | 4.8 GFlops                                                  |
| 输入数据 | images, [batch_size, 3, 224, 224], FP32，NCHW，RGB planar    |
| 输出数据 | output,  [batch_size, 512] |
| 前处理   | resize, center crop, BGR->RGB, nomarlize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))   |
| 后处理   | softmax等                                              |

#### 3.2.1 下载ViT源码

```bash
# 下载ViT源码
git clone https://github.com/openai/CLIP
# 切换到ViT工程目录
cd CLIP
# 使用anaconda创建1个Python==3.8.12的虚拟环境并激活这个环境
conda create -n py38vit python==3.8.12
conda activate py38vit
# 安装依赖
pip install -r requirements.txt
```

#### 3.2.1 修改clip/model.py

修改`CLIP`类的`forward`函数:

```python
def forward(self, image):
    image_features = self.encode_image(image)
    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features / image_features.norm(dim=-1, keepdim=True)
```

#### 3.2.2 导出JIT模型

BMNNSDK2中的PyTorch模型编译工具BMNETP只接受PyTorch的JIT模型（TorchScript模型）。

JIT（Just-In-Time）是一组编译工具，用于弥合PyTorch研究与生产之间的差距。它允许创建可以在不依赖Python解释器的情况下运行的模型，并且可以更积极地进行优化。在已有PyTorch的Python模型（基类为torch.nn.Module）的情况下，通过torch.jit.trace就可以得到JIT模型，如`torch.jit.trace(python_model, torch.rand(input_shape)).save('jit_model')`。BMNETP暂时不支持带有控制流操作（如if语句或循环）的JIT模型，因此不能使用torch.jit.script，而要使用torch.jit.trace，它仅跟踪和记录张量上的操作，不会记录任何控制流操作。

新建`export.py`, 并输入以下内容

```python
import clip
import torch

# Load the model
device = "cpu" 
model, preprocess = clip.load('ViT-B/32', device)
input=torch.rand([1,3,224,224])
trace_model = torch.jit.trace(model, (input))
torch.jit.save(trace_model, "ViT_224_1b.pt")
```

```bash
python export.py
```

上述脚本会在原始pt模型所在目录下生成导出的JIT模型。

### 3.3 准备量化集


## 4. 模型转换

模型转换的过程需要在x86下的docker开发环境中完成。以下操作均在x86下的docker开发环境中完成。下面我们以3个output的情况为例，介绍如何完成模型的转换。

### 4.1 生成FP32 BModel

执行以下命令，使用bmnetp编译生成FP32 BModel，请注意修改`model_info.sh`中的模型名称、生成模型目录和输入大小shapes、使用的量化LMDB文件目录、batch_size、img_size等参数：

```bash
./gen_fp32bmodel.sh
```

上述脚本会在`fp32model/`下生成`*_fp32_1b.bmodel`文件，即转换好的FP32 BModel，使用`bm_model.bin --info`查看的模型具体信息如下：



### 4.2 生成INT8 BModel

不量化模型可跳过本节。

INT8 BModel的生成需要经历中间格式UModel，即：原始模型→FP32 UModel→INT8 UModel→INT8 BModel。请注意修改`model_info.sh`中的模型名称、生成模型目录和输入大小shapes、使用的量化LMDB文件目录、batch_size、img_size等参数。

执行以下命令，将依次调用以下步骤中的脚本，生成INT8 BModel：

```shell
./2_gen_int8bmodel.sh
```

### 4.2.0 生成LMDB

需要将原始量化数据集转换成lmdb格式，供后续校准量化工具Quantization-tools 使用。更详细信息请参考：[准备LMDB数据集](https://doc.sophgo.com/docs/docs_latest_release/calibration-tools/html/module/chapter4.html#lmdb)。

在docker开发容器中使用`ufw.io ` 工具从数据集图片生成LMDB文件，具体操作参见`tools/convert_imageset.py`, 相关操作已被封装在 `scripts/20_create_lmdb.sh`中，执行如下命令即可：

```
./20_create_lmdb.sh
```

上述脚本会在指定目录中生成lmdb的文件夹，其中存放着量化好的LMDB文件：`data.mdb`。请注意根据模型输入要求修改脚本中`convert_imageset`命令中的`resize_width`和`resize_height`等参数。

#### 4.2.1 生成FP32 UModel

执行以下命令，使用`ufw.pt_to_umodel`生成FP32 UModel，若不指定-D参数，可以在生成prototxt文件以后修改：

```bash
./21_gen_fp32umodel.sh
```

上述脚本会在`int8model/`下生成`*_bmnetp_test_fp32.prototxt`、`*_bmnetp.fp32umodel`文件，即转换好的FP32 UModel。

#### 4.2.2 修改FP32 UModel

执行以下命令，修改FP32 UModel的prototxt文件即`_bmnetp_test_fp32.prototxt`，将输入层替换为Data层指向LMDB文件位置（若上一步已经指定-D参数，则无需操作），并使用`transform_op`完成需要进行的预处理；如果`transform_op`无法完成要求的预处理，那么可以使用Python程序来生成LMDB文件：

```bash
./22_modify_fp32umodel.sh
```


#### 4.2.3 生成INT8 UModel

执行以下命令，使用修改后的FP32 UModel文件生成INT8 UModel：

```
./23_gen_int8umodel.sh
```

上述脚本会在`int8model/`下生成`*_bmnetp_deploy_fp32_unique_top.prototxt`、`*_bmnetp_deploy_int8_unique_top.prototxt`和`*_bmnetp.int8umodel`文件，即转换好的INT8 UModel。

#### 4.2.4 生成INT8 BModel

执行以下命令，使用生成的INT8 UModel文件生成INT8 BModel：

```
./24_gen_int8bmodel.sh
```

上述脚本会在`int8model/`下生成`*_int8_1b.bmodel`，即转换好的INT8 BModel，使用`bm_model.bin --info`查看的模型具体信息如下：



由于量化模型通常存在精度损失，当使用默认脚本生成的量化模型精度不能满足需求时，可能需要修改量化策略并借助自动量化工具auto-calib寻找最优结果，甚至在必要时需要将某些量化精度损失较大的层单独设置为使用fp32推理，相关调试方法请参考[《量化工具用户开发手册》](https://doc.sophgo.com/docs/docs_latest_release/calibration-tools/html/index.html)。


## 5. 部署测试


### 5.1 环境配置

#### 5.1.1 x86 SC5

对于x86 SC5平台，程序执行所需的环境变量执行`source envsetup_pcie.sh`时已经配置完成

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

C++例程适用于多种输出Tensor的情形。

#### 5.2.1 x86平台SC5

- 编译



- 测试



#### 5.2.2 arm平台SE5

对于arm平台SE5，需要在docker开发容器中使用交叉编译工具链编译生成可执行文件，而后拷贝到Soc目标平台运行。

- 在docker开发容器中交叉编译



- 将生成的可执行文件及所需的模型和测试图片或视频文件拷贝到盒子中测试



### 5.3 Python例程部署测试

Python代码无需编译，无论是x86 SC平台还是arm SE5平台配置好环境之后就可直接运行。

样例中提供了一系列例程以供参考使用，具体情况如下：




> **使用SAIL模块的注意事项：**对于INT8 BModel来说，当输入输出为int8时，含有scale，需要在处理时将输入输出乘以相应的scale。使用SAIL接口推理时，当sail.Engine.process()接口输入为numpy时，SAIL内部会自动乘以scale，用户无需操作；而输入为Tensor时，需要手动在数据送入推理接口前乘以scale。
>
> 这是因为Tensor作为输入的话，一般图像来源就是bm_image，这样就可以直接调用vpp进行scale等操作，所以推理之前由用户乘以scale更高效；而在python接口中，当numpy作为输入的话，推理之前没办法调用vpp，sail内部使用SSE指令进行了加速。