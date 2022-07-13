# CenterNet

## 目录

* [CenterNet](#CenterNet)
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

CenterNet 是一种 anchor-free 的目标检测网络，不仅可以用于目标检测，还可以用于其他的一些任务，如姿态识别或者 3D 目标检测等等。

**文档:** [CenterNet论文](https://arxiv.org/pdf/1904.07850.pdf)

**参考repo:** [CenterNet](https://github.com/xingyizhou/CenterNet)



## 2. 数据集

[MS COCO](http://cocodataset.org/#home)，是微软构建的一个包含分类、检测、分割等任务的大型的数据集。使用[CenterNet](https://github.com/xingyizhou/CenterNet)基于COCO Detection 2017预训练好的80类通用目标检测模型。

## 3. 准备环境与数据


### 3.1 准备开发环境

开发环境是指用于模型转换或验证以及程序编译等开发过程的环境，目前只支持x86，需要使用我们提供的基于Ubuntu18.04的docker镜像。

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

- 开发docker基础镜像：[点击前往官网下载Ubuntu开发镜像](https://developer.sophgo.com/site/index/material/11/all.html)，请选择与SDK版本适配的docker镜像

- SDK软件包：[点击前往官网下载SDK软件包](https://developer.sophgo.com/site/index/material/17/all.html)，请选择与仓库代码分支对应的SDK版本


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
#### 3.1.4 拷贝examples代码到容器
请从[github](https://github.com/sophon-ai-algo/examples)下载代码，拷贝到容器内编译测试。
github上centernet例程位于examples/simple/centernet/。

`git clone https://github.com/sophon-ai-algo/examples.git`

下载后，通过`docker cp <examples-dir> <container-id>:/workspace`的方式，拷贝到上一步的docker容器中
> 下面操作，默认centernet路径在容器中路径为/workspace/examples_test

### 3.2 准备模型

从[CenterNet GoogleDrive](https://drive.google.com/drive/folders/1px-Xg7jXSC79QqgsD1AAGJQkuf5m0zh_)下载所需的pt模型。

> **注意：**本示例展示的是使用CenterNet进行目标检测。由于工具链目前对DeformConv可变卷积还未支持，所以选用dlav0作为主干网, 从官方ModelZoo中下载对应的与训练pt文件。



#### 3.2.1 JIT环境准备
SophonSDK中的PyTorch模型编译工具BMNETP只接受PyTorch的JIT模型（TorchScript模型）。

JIT（Just-In-Time）是一组编译工具，用于弥合PyTorch研究与生产之间的差距。它允许创建可以在不依赖Python解释器的情况下运行的模型，并且可以更积极地进行优化。在已有PyTorch的Python模型（基类为torch.nn.Module）的情况下，通过torch.jit.trace就可以得到JIT模型，如`torch.jit.trace(python_model, torch.rand(input_shape)).save('jit_model')`。BMNETP暂时不支持带有控制流操作（如if语句或循环）的JIT模型，因此不能使用torch.jit.script，而要使用torch.jit.trace，它仅跟踪和记录张量上的操作，不会记录任何控制流操作。可在源码导入CPU模型后通过添加以下代码导出符合要求的JIT模型：

```bash
# 下载dlav0作为主干网的预训练模型
apt update
apt install curl
cd /workspace/examples_test/simple/centernet/data/scripts/
./download_pt.sh
# 下载成功后，文件位于../build/ctdet_coco_dlav0_1x.pth

# 切换目录
cd ../build
```

#### dlav0.py网络修改说明
当前目录下dlav0.py，是从[CenterNet源码](https://github.com/xingyizhou/CenterNet)中，修改dlav0.py中DLASeg类forward方法的返回值后得到的。
```python
#return [ret]
return torch.cat((ret['hm'], ret['wh'], ret['reg']), 1) 
```
将heatmap, wh, reg三个head的特征图concat到一起，方便后续bmodel的转换

#### JIT模型生成
直接运行export.py即可
```bash
python3 export.py
cp ctdet_coco_dlav0_1x.torchscript.pt ../models
```
当前目录下生成了一份ctdet_coco_dlav0_1x.torchscript.pt文件


### 3.3 准备量化集

不量化模型可跳过本节。

量化集使用COCO Detection 2017的验证集
我们选取其中的200张图片进行量化

```bash
apt install unzip
cd ../scripts
./00_prepare.sh
# 下载成功后，JPG文件位于../images文件夹中
```


## 4. 模型转换

模型转换的过程需要在x86下的docker开发环境中完成。以下操作均在x86下的docker开发环境中完成。

### 4.1 生成FP32 BModel

```bash
# SDK_PATH改为您SDK的根路径，如果您在docker内，则默认为/workspace
pushd $SDK_PATH/scripts
./install_lib.sh nntc
source envsetup_pcie.sh
popd
./1_gen_fp32bmodel.sh
```

上述脚本会在`../models/`下生成`ctdet_coco_dlav0_1output_512_fp32_1batch.bmodel`文件，即转换好的FP32 BModel，使用`bm_model.bin --info`查看的模型具体信息如下：

```bash
bmodel version: B.2.2
chip: BM1684
create time: Fri Apr  1 16:36:25 2022

==========================================
net 0: [ctdet_dlav0]  static
------------
stage 0:
input: input.1, [1, 3, 512, 512], float32, scale: 1
output: 40, [1, 84, 128, 128], float32, scale: 1

device mem size: 113864072 (coeff: 72048648, instruct: 134528, runtime: 41680896)
host mem size: 0 (coeff: 0, runtime: 0)
```

### 4.2 生成INT8 BModel

不量化模型可跳过本节。


执行以下命令，使用一键量化工具cali_model，生成INT8 BModel：

```shell
./2_gen_int8bmodel.sh
```
上述脚本会在`../models/`下生成`ctdet_coco_dlav0_1output_512_int8_4batch.bmodel`，即转换好的INT8 BModel，使用`bm_model.bin --info`查看的模型具体信息如下


```bash
bmodel version: B.2.2
chip: BM1684
create time: Sat Apr  2 00:09:40 2022

==========================================
net 0: [ctdet_coco_dlav0_1x.torchscript_bmnetp]  static
------------
stage 0:
input: input.1, [4, 3, 512, 512], int8, scale: 60.4494
output: 40, [4, 84, 128, 128], float32, scale: 1

device mem size: 78307080 (coeff: 18616328, instruct: 147200, runtime: 59543552)
host mem size: 0 (coeff: 0, runtime: 0)
```

## 5. 部署测试

测试图片见`/workspace/examples_test/simple/centernet/data/`，转换好的bmodel文件可以放置于`/workspace/examples_test/simple/centernet/data/models`。


### 5.1 环境配置

#### 5.1.1 x86 PCIe

对于x86 with PCIe加速卡平台，程序执行所需的环境变量执行`source envsetup_pcie.sh`时已经配置完成。

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

#### 5.2.1 x86平台 PCIe加速卡

- 编译

```bash
$ cd /workspace/examples_test/simple/centernet/cpp_bmcv_sail
# 先手动修改Makefile.pcie里的top_dir地址，指向实际SDK的根路径
# docker容器中，默认为/workspace，请根据实际挂载的SDK路径进行修改
$ make -f Makefile.pcie # 生成centernet_bmcv_sail.pcie
```

- 测试

```bash
# 1batch
$ ./centernet_bmcv_sail.pcie --bmodel=../data/models/ctdet_coco_dlav0_1output_512_fp32_1batch.bmodel --image=../data/ctdet_test.jpg
# 执行完毕后，在当前目录生成ctdet_result_20xx-xx-xx-xx-xx-xx.jpg格式的图片
# 图片上检测出11个目标

# 4batch
$ ./centernet_bmcv_sail.pcie --bmodel=../data/models/ctdet_coco_dlav0_1output_512_int8_4batch.bmodel --image=../data/ctdet_test.jpg
# 执行完毕后，在当前目录生成ctdet_result_20xx-xx-xx-xx-xx-xx-bx.jpg格式的图片
# 按照量化结果差异，图片上检测出11-13个目标，均属正常范围
```

#### 5.2.2 arm平台SE5

对于arm平台SE5，需要在docker开发容器中使用交叉编译工具链编译生成可执行文件，而后拷贝到Soc目标平台运行。

- 在docker开发容器中交叉编译

```bash
$ cd cpp_bmcv_sail
# 先手动修改Makefile.arm里的top_dir地址，指向实际SDK的根路径
# docker容器中，默认为/workspace，请根据实际挂载的SDK路径进行修改
$ make -f Makefile.arm # 生成centernet_bmcv_sail.arm
```

- 将以下文件拷贝到盒子中同一个目录中，进行测试
1. `centernet_bmcv_sail.arm`
2. `/workspace/examples_test/simple/centernet/data/models/ctdet_coco_dlav0_1output_512_fp32_1batch.bmodel`
3. `/workspace/examples_test/simple/centernet/data/models/ctdet_coco_dlav0_1output_512_int8_4batch.bmodel`
4. `/workspace/examples_test/simple/centernet/data/ctdet_test.jpg`
5. `/workspace/examples_test/simple/centernet/data/coco_classes.txt`
```bash
# 1batch
$ ./centernet_bmcv_sail.arm --bmodel=ctdet_coco_dlav0_1output_512_fp32_1batch.bmodel --image=ctdet_test.jpg
# 执行完毕后，在当前目录生成ctdet_result_20xx-xx-xx-xx-xx-xx.jpg格式的图片
# 图片上检测出11个目标

# 4batch
$ ./centernet_bmcv_sail.arm --bmodel=ctdet_coco_dlav0_1output_512_int8_4batch.bmodel --image=ctdet_test.jpg
# 执行完毕后，在当前目录生成ctdet_result_20xx-xx-xx-xx-xx-xx_bx.jpg格式的图片
# 按照量化结果差异，图片上检测出11-13个目标，均属正常范围
```

### 5.3 Python例程部署测试

Python代码无需编译，无论是x86 SC平台还是arm SE5平台配置好环境之后就可直接运行。
> 运行之前需要安装sail包

#### 5.3.1 x86平台PCIe模式
```bash
# 在容器里, 以python3.7的docker为例
cd /workspace/lib/sail/python3/pcie/py37
pip3 install sophon-2.7.0-py3-none-any.whl

cd /workspace/examples_test/simple/centernet/py_bmcv_sail

# 1batch
python3 det_centernet_bmcv_sail_1b_4b.py --bmodel=../data/models/ctdet_coco_dlav0_1output_512_fp32_1batch.bmodel --input=../data/ctdet_test.jpg
# 执行完毕后，在当前目录生成ctdet_result_20xx-xx-xx-xx-xx-xx_b_x.jpg格式的图片
# 图片上检测出11个目标

# 4batch
python3 det_centernet_bmcv_sail_1b_4b.py --bmodel=../data/models/ctdet_coco_dlav0_1output_512_int8_4batch.bmodel --input=../data/ctdet_test.jpg
# 执行完毕后，在当前目录生成ctdet_result_20xx-xx-xx-xx-xx-xx_b_x.jpg格式的图片
# 按照量化结果差异，图片上检测出11-13个目标，均属正常范围
```

1. 如果是fp32的模型，图片有11个框
2. 如果是int8的模型，按照量化结果差异，图片上检测出11-13个目标，均属正常范围

> **使用SAIL模块的注意事项：**对于INT8 BModel来说，当输入输出为int8时，含有scale，需要在处理时将输入输出乘以相应的scale。使用SAIL接口推理时，当sail.Engine.process()接口输入为numpy时，SAIL内部会自动乘以scale，用户无需操作；而输入为Tensor时，需要手动在数据送入推理接口前乘以scale。
> 这是因为Tensor作为输入的话，一般图像来源就是bm_image，这样就可以直接调用vpp进行scale等操作，所以推理之前由用户乘以scale更高效；而在python接口中，当numpy作为输入的话，推理之前没办法调用vpp，sail内部使用SSE指令进行了加速。
#### 5.3.2 SE5智算盒SoC模式
> 将py_bmcv_sail整个文件夹拷贝到SE5中，和`5.2.2`中bmodel和jpg文件同一目录下
```bash
cd py_bmcv_sail
# 1batch
python3 det_centernet_bmcv_sail_1b_4b.py --bmodel=../ctdet_coco_dlav0_1output_512_int8_4batch.bmodel --input=../ctdet_test.jpg --class_path=../coco_classes.txt
# 4batch
python3 det_centernet_bmcv_sail_1b_4b.py --bmodel=../ctdet_coco_dlav0_1output_512_fp32_1batch.bmodel --input=../ctdet_test.jpg --class_path=../coco_classes.txt
```
成功后，在当前目录下生成和`5.3.1`相同的`ctdet_result_20xx-xx-xx-xx-xx-xx_b_x.jpg`图片
