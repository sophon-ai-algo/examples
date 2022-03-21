# YOLOv5

## 目录

* [YOLOv5](#YOLOv5)
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

YOLOv5是非常经典的基于anchor的One Stage目标检测算法YOLO的改进版本，因其优秀的精度和速度表现，在工程实践应用中获得了非常广泛的应用。

**文档:** [YOLOv5文档](https://docs.ultralytics.com/)

**参考repo:** [yolov5](https://github.com/ultralytics/yolov5)

**实现repo：**[yolov5_demo](https://github.com/xiaotan3664/)


## 2. 数据集

[MS COCO](http://cocodataset.org/#home)，是微软构建的一个包含分类、检测、分割等任务的大型的数据集。使用[yolov5](https://github.com/ultralytics/yolov5)基于COCO Detection 2017预训练好的80类通用目标检测模型。

> MS COCO提供了一些[API](https://github.com/cocodataset/cocoapi)，方便对数据集的使用和模型评估，您可以使用pip安装` pip3 install pycocotools`，并使用COCO提供的API进行下载。

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

从[yolov5 release](https://github.com/ultralytics/yolov5/releases/)下载所需的pt模型。

**注意：**YOLOv5有多个版本：1.0、2.0、3.0、3.1、4.0、5.0、6.0、6.1。YOLOv5不同版本的代码导出的YOLOv5模型的输出会有不同，主要取决于model/yolo.py文件中的class Detect的forward函数。根据不同的组合，可能会有1、2、3、4个输出的情况，v6.1版本默认会有4个输出。具体情况可参见docs目录下的说明文档《YOLOV5模型导出与优化.docx》。

```bash
# 下载yolov5s v6.1版本
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -P build/
```

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
# 下载yolov5源码
git clone https://github.com/ultralytics/yolov5.git
# 切换到yolov5工程目录
cd yolov5
# 使用tag从远程创建本地v6.1 分支
git branch v6.1 v6.1
# 使用anaconda创建1个Python==3.8.12的虚拟环境并激活这个环境
conda create -n py38yolov5 python==3.8.12
conda activate py38yolov5
# 安装依赖
pip install -r requirements.txt
```

#### 3.2.2 修改model/yolo.py

修改Detect类的forward函数的最后return语句，实现不同的输出

```python
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

BMNNSDK2中的PyTorch模型编译工具BMNETP只接受PyTorch的JIT模型（TorchScript模型）。

JIT（Just-In-Time）是一组编译工具，用于弥合PyTorch研究与生产之间的差距。它允许创建可以在不依赖Python解释器的情况下运行的模型，并且可以更积极地进行优化。在已有PyTorch的Python模型（基类为torch.nn.Module）的情况下，通过torch.jit.trace就可以得到JIT模型，如`torch.jit.trace(python_model, torch.rand(input_shape)).save('jit_model')`。BMNETP暂时不支持带有控制流操作（如if语句或循环）的JIT模型，因此不能使用torch.jit.script，而要使用torch.jit.trace，它仅跟踪和记录张量上的操作，不会记录任何控制流操作。这部分操作yolov5已经为我们写好，只需运行如下命令即可导出符合要求的JIT模型：

```bash
python export.py --weights ${PATH_TO_YOLOV5S_MODEL}/yolov5s.pt --include torchscript
```

上述脚本会在原始pt模型所在目录下生成导出的JIT模型，导出后可以修改模型名称以区分不同版本和输出类型，如`yolov5s_640_coco_v6.1_1output.torchscript`表示仅带有1个融合后的输出的JIT模型。

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
./gen_fp32bmodel.sh
```

上述脚本会在`fp32model/`下生成`*_fp32_1b.bmodel`文件，即转换好的FP32 BModel，使用`bm_model.bin --info`查看的模型具体信息如下：

```bash
bmodel version: B.2.2
chip: BM1684
create time: Tue Mar  8 11:00:45 2022

==========================================
net 0: [yolov5s_coco_v6.1]  static
------------
stage 0:
input: x.1, [1, 3, 640, 640], float32, scale: 1
output: 172, [1, 3, 80, 80, 85], float32, scale: 1
output: 173, [1, 3, 40, 40, 85], float32, scale: 1
output: 174, [1, 3, 20, 20, 85], float32, scale: 1
```

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

执行以下命令，修改FP32 UModel的prototxt文件即`yolov5s_coco_v6.1_3output.torchscript_bmnetp_test_fp32.prototxt`，将输入层替换为Data层指向LMDB文件位置（若上一步已经指定-D参数，则无需操作），并使用`transform_op`完成需要进行的预处理；对于yolov5s来说，需要设置scale和bgr2rgb；如果`transform_op`无法完成要求的预处理，那么可以使用Python程序来生成LMDB文件：

```bash
./22_modify_fp32umodel.sh
```

修改后的prototxt如下：

```
model_type: BMNETP2UModel
output_whitelist: "172"
output_whitelist: "173"
output_whitelist: "174"
inputs: "x.1"
outputs: "172"
outputs: "173"
outputs: "174"
layer {
  name: "x.1"
  type: "Data"
  top: "x.1"
  include {
    phase: TEST
  }
  transform_param {
      transform_op {
         op: STAND
         mean_value: 0
         mean_value: 0
         mean_value: 0
         scale: 0.00392156862745
         bgr2rgb: true
      }
   }
  data_param {
    source: "./images/test/img_lmdb"
    batch_size: 1
    backend: LMDB
  }
}
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

```bash
# for yolov5s_640_coco_v6.1_3output_int8_1b.bmodel
bmodel version: B.2.2
chip: BM1684
create time: Tue Mar  8 17:13:15 2022

==========================================
net 0: [yolov5s_coco_v6.1_3output.torchscript_bmnetp]  static
------------
stage 0:
input: x.1, [1, 3, 640, 640], int8, scale: 63.8269
output: 172, [1, 3, 80, 80, 85], int8, scale: 0.139196
output: 173, [1, 3, 40, 40, 85], int8, scale: 0.116644
output: 174, [1, 3, 20, 20, 85], int8, scale: 0.106422
```

由于量化模型通常存在精度损失，当使用默认脚本生成的量化模型精度不能满足需求时，可能需要修改量化策略并借助自动量化工具auto-calib寻找最优结果，甚至在必要时需要将某些量化精度损失较大的层单独设置为使用fp32推理，相关调试方法请参考[《量化工具用户开发手册》](https://doc.sophgo.com/docs/docs_latest_release/calibration-tools/html/index.html)。

> 注意：当使用1个output输出时，由于最后是把还原的box和score结果concat到一起去，而box和score的数值差异比较大（box中x，y取值范围是0-640， w和h相对小些， score的取值范围是0-1），在量化过程中会导致精度损失严重，可以考虑把box和score分别作为输出。相关内容参见docs目录下的说明文档《YOLOV5模型导出与优化.docx》。

## 5. 部署测试

请注意根据您使用的模型，修改`cpp/yolov5.cpp`或者`python/yolov5_***.py`中的anchors信息以及使用的`coco.names`文件；类别数量是根据模型输出Tensor的形状自动计算得到的，无需修改。

测试图片见`data/images`，测试视频见`data/videos`，转换好的bmodel文件可以放置于`data/models`。

> 已经转换好的bmodel文件可从以下百度网盘下载：
>
> 链接: https://pan.baidu.com/s/1d3f8CjzC3BF2-2I2OF0q1g 提取码: lt59 

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

```bash
$ cd cpp
$ make -f Makefile.pcie # 生成yolov5_demo.pcie
```

- 测试

```bash
 $ ./yolov5_demo.pcie --input=path/to/image --bmodel=xxx.bmodel # use your own yolov5 bmodel
 # $ ./yolov5_demo.pcie --input=../data/images/dance.mp4 --is_video=true # use video as input, and process all frames
 # $ ./yolov5_demo.pcie --input=../data/videos/dance.mp4 --is_video=true --frame_num=4 # use video as input, and process the first 4 frames
 # $ ./yolov5_demo.pcie --help # see detail help info
```

#### 5.2.2 arm平台SE5

对于arm平台SE5，需要在docker开发容器中使用交叉编译工具链编译生成可执行文件，而后拷贝到Soc目标平台运行。

- 在docker开发容器中交叉编译

```bash
$ cd cpp
$ make -f Makefile.arm # 生成yolov5_demo.arm
```

- 将生成的可执行文件及所需的模型和测试图片或视频文件拷贝到盒子中测试

```bash
 $ ./yolov5_demo.arm --input=path/to/image --bmodel=xxx.bmodel # use your own yolov5 bmodel
 # $ ./yolov5_demo.arm --input=../data/images/dance.mp4 --is_video=true # use video as input, and process all frames
 # $ ./yolov5_demo.arm --input=../data/videos/dance.mp4 --is_video=true --frame_num=4 # use video as input, and process the first 4 frames
 # $ ./yolov5_demo.arm --help # see detail help info
```

### 5.3 Python例程部署测试

Python代码无需编译，无论是x86 SC平台还是arm SE5平台配置好环境之后就可直接运行。

样例中提供了一系列例程以供参考使用，具体情况如下：

| #    | 样例文件                 | 说明                                                         |
| ---- | ------------------------ | ------------------------------------------------------------ |
| 1    | yolov5_bmcv_1output.py   | 使用SAIL解码、BMCV前处理、SAIL推理、OpenCV后处理，适用模型为1个输出 |
| 2    | yolov5_bmcv_3output.py   | 使用SAIL解码、BMCV前处理、SAIL推理、OpenCV后处理，适用模型为3个输出 |
| 3    | yolov5_opencv_1output.py | 使用OpenCV解码、OpenCV前处理、SAIL推理、OpenCV后处理，适用模型为1个输出 |
| 4    | yolov5_opencv_3output.py | 使用OpenCV解码、OpenCV前处理、SAIL推理、OpenCV后处理，适用模型为3个输出 |
| 5    | yolov5_pytorch.py        | 使用OpenCV读取图片和前处理、pytorch推理、OpenCV后处理，适用模型为old/yolov5s.torchscript.640.1.pt |
| 6    | yolov5_sail.py           | 使用OpenCV读取图片和前处理、SAIL推理、OpenCV后处理，适用模型为old/yolov5s_fp32_640_1.bmodel和yolov5s_fix8b_640_1.bmodel |



> **使用SAIL模块的注意事项：**对于INT8 BModel来说，当输入输出为int8时，含有scale，需要在处理时将输入输出乘以相应的scale。使用SAIL接口推理时，当sail.Engine.process()接口输入为numpy时，SAIL内部会自动乘以scale，用户无需操作；而输入为Tensor时，需要手动在数据送入推理接口前乘以scale。
>
> 这是因为Tensor作为输入的话，一般图像来源就是bm_image，这样就可以直接调用vpp进行scale等操作，所以推理之前由用户乘以scale更高效；而在python接口中，当numpy作为输入的话，推理之前没办法调用vpp，sail内部使用SSE指令进行了加速。