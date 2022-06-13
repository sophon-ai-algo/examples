# YOLOx

## 目录
* [YOLOx](#YOLOx)
  * [目录](#目录)
  * [1.简介](#1-简介)
  * [2.数据集](#2-数据集)
  * [3.准备环境与数据](#3-准备环境与数据)
    * [3.1 准备开发环境](#31-准备开发环境)
    * [3.2 准备模型](#32-准备模型)
    * [3.3 准备量化集](#33-准备量化集)
  * [4.模型转换](#4-模型转换)
    * [4.1 生成FP32 BModel](#41-生成fp32-bmodel)
    * [4.2 生成INT8 BModel](#42-生成int8-bmodel)
    * [4.3 使用脚本快速生成INT8 BMODEL](#43-使用脚本楷书生成int8-bmodel)
  * [5.部署测试]
    * [5.1 环境配置](#51-环境配置)
    * [5.2 C++例程部署测试](#52-C++例程部署测试)
    * [5.3 Python例程部署测试](#53-Python例程部署测试)
    * [5.4 使用脚本对示例代码进行自动测试](#54-使用脚本对示例代码进行自动测试)

## 1.简介

YOLOx由旷世研究提出,是基于YOLO系列的改进。

**论文地址** (https://arxiv.org/abs/2107.08430)

**官方源码地址** (https://github.com/Megvii-BaseDetection/YOLOX)


## 2. 数据集

[MS COCO](http://cocodataset.org/#home),是微软构建的一个包含分类、检测、分割等任务的大型的数据集.

> MS COCO提供了一些[API](https://github.com/cocodataset/cocoapi),方便对数据集的使用和模型评估,您可以使用pip安装` pip3 install pycocotools`,并使用COCO提供的API进行下载.

## 3. 准备环境与数据


### 3.1 准备开发环境

开发环境是指用于模型转换或验证以及程序编译等开发过程的环境，目前只支持x86，需要使用我们提供的基于Ubuntu18.04的docker镜像。

运行环境是具备Sophon设备的平台上实际使用设备进行算法应用部署的环境，有PCIe加速卡、SM模组、SE微服务器等，所有运行环境上的BModel都是一样的，SDK中各模块的接口也是一致的。

开发环境与运行环境可能是统一的（如插有SC5加速卡的x86主机,既是开发环境又是运行环境），也可能是分离的（如使用x86主机作为开发环境转换模型和编译程序，使用盒子部署运行最终的算法应用）。

但是,无论使用的产品是SoC模式还是PCIe模式，都需要一台x86主机作为开发环境，模型的转换工作必须在开发环境中完成。

#### **3.1.1 开发主机准备：**

- 开发主机：一台安装了Ubuntu16.04/18.04/20.04的x86主机,运行内存建议12GB以上

- 安装docker：参考《[官方教程](https://docs.docker.com/engine/install/)》,若已经安装请跳过

  ```bash
  # 安装docker
  sudo apt-get install docker.io
  # docker命令免root权限执行
  # 创建docker用户组,若已有docker组会报错,没关系可忽略
  sudo groupadd docker
  # 将当前用户加入docker组
  sudo gpasswd -a ${USER} docker
  # 重启docker服务
  sudo service docker restart
  # 切换当前会话到新group或重新登录重启X会话
  newgrp docker
  ```

#### **3.1.2 SDK软件包下载：**

- 开发docker基础镜像：[点击前往官网下载Ubuntu开发镜像](https://developer.sophgo.com/site/index/material/11/all.html)，请选择与SDK版本适配的docker镜像


- SDK软件包：[点击前往官网下载SDK软件包](https://developer.sophgo.com/site/index/material/17/all.html)，请选择与仓库代码分支对应的SDK版本

#### **3.1.3 创建docker开发环境：**

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

- 因为上述docker已经安装了pytorch，但是版本较yolox版本要求的版本低一些，所以此步骤不建议在docker内进行，最好在物理机上直接进行

- YOLOx模型的模型参数  
  
| Model                                       | size | mAP<sup>val<br>0.5:0.95 | mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) | FLOPs<br>(G) |                           weights                            |
| ------------------------------------------- | :--: | :---------------------: | :----------------------: | :----------------: | :-----------: | :----------: | :----------------------------------------------------------: |
| [YOLOX-s](./exps/default/yolox_s.py)        | 640  |          40.5           |           40.5           |        9.8         |      9.0      |     26.8     | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth) |
| [YOLOX-m](./exps/default/yolox_m.py)        | 640  |          46.9           |           47.2           |        12.3        |     25.3      |     73.8     | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth) |
| [YOLOX-l](./exps/default/yolox_l.py)        | 640  |          49.7           |           50.1           |        14.5        |     54.2      |    155.6     | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) |
| [YOLOX-x](./exps/default/yolox_x.py)        | 640  |          51.1           |           51.5           |        17.3        |     99.1      |    281.9     | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth) |
| [YOLOX-Darknet53](./exps/default/yolov3.py) | 640  |          47.7           |           48.0           |        11.1        |     63.7      |    185.3     | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_darknet.pth) |


#### **3.2.1 下载yolovx源码**

 ```bash
 # 下载yolox源码
 git clone https://github.com/Megvii-BaseDetection/YOLOX
 # 切换到yolox工程目录
 cd YOLOX
 # 安装依赖
 pip3 install -r requirements.txt
 ```

#### **3.2.2 导出JIT模型**

 SophonSDK中的PyTorch模型编译工具BMNETP只接受PyTorch的JIT模型（TorchScript模型）.

 JIT（Just-In-Time）是一组编译工具，用于弥合PyTorch研究与生产之间的差距。它允许创建可以在不依赖Python解释器的情况下运行的模型，并且可以更积极地进行优化。在已有PyTorch的Python模型（基类为torch.nn.Module）的情况下，通过torch.jit.trace就可以得到JIT模型,如`torch.jit.trace(python_model, torch.rand(input_shape)).save('jit_model')`。BMNETP暂时不支持带有控制流操作（如if语句或循环）的JIT模型，因此不能使用`torch.jit.script`，而要使用`torch.jit.trace`，它仅跟踪和记录张量上的操作，不会记录任何控制流操作。这部分操作YOLOX已经为我们写好，只需运行如下命令即可导出符合要求的JIT模型：

- YOLOX-s
  ```bash
    python3 tools/export_torchscript.py -n yolox-s -c ${PATH_TO_YOLOX_MODEL}/yolox_s.pth --output-name ${PATH_TO_YOLOX_MODEL}/yolox_s.trace.pt
  ```
- YOLOX-m
  ```bash
    python3 tools/export_torchscript.py -n yolox-m -c ${PATH_TO_YOLOX_MODEL}/yolox_m.pth --output-name ${PATH_TO_YOLOX_MODEL}/yolox_m.trace.pt
  ```
- YOLOX-l
  ```bash
    python3 tools/export_torchscript.py -n yolox-l -c ${PATH_TO_YOLOX_MODEL}/yolox_l.pth --output-name ${PATH_TO_YOLOX_MODEL}/yolox_l.trace.pt
  ```
- YOLOX-x
  ```bash
    python3 tools/export_torchscript.py -n yolox-x -c ${PATH_TO_YOLOX_MODEL}/yolox_x.pth --output-name ${PATH_TO_YOLOX_MODEL}/yolox_x.trace.pt
  ```
- YOLOX-Darknet53
  ```bash
    python3 tools/export_torchscript.py -n yolov3 -c ${PATH_TO_YOLOX_MODEL}/yolox_darknet.pth --output-name ${PATH_TO_YOLOX_MODEL}/yolox_darknet.trace.pt
  ```

上述脚本会在 `${PATH_TO_YOLOX_MODEL}` 下生成相应的JIT模型


### 3.3 准备量化集

此步骤需要在开发主机的docker内进行，不量化模型可以跳过本节。

#### **3.3.1 准备量化图片**

示例从coco数据集中随机选取了部分图片，保存在docker内的路径为：${OST_DATA_PATH}

#### **3.3.2 使用不同的resize(opencv/tpu/vpp)方法对图片进行扩展**

- 使用opencv做resize and padding

  ```bash
  python3 image_resize.py --ost_path=${OST_DATA_PATH} --dst_path=${RESIZE_DATA_PATH} --dst_width=640 --dst_height=640
  ```
  结果图片将保存在`${RESIZE_DATA_PATH}`中

- 使用tpu和vpp做resize and padding

  因为此操作用到了PCIe加速卡硬件，所以如果没有可跳过此步骤，除设置环境变量外，此操作还需要安装好sail包：

  ```bash
  python3 image_resize_sophgo.py --ost_path=${OST_DATA_PATH} --dst_path=${RESIZE_DATA_PATH} --dst_width=640--dst_height=640
  ```
  结果图片将保存在`${RESIZE_DATA_PATH}`中

### 3.3.3 生成lmdb数据

  ```bash
    python3 ../../calibration/create_lmdb_demo/convert_imageset.py \
        --imageset_rootfolder=${RESIZE_DATA_PATH} \
        --imageset_lmdbfolder=${LMDB_PATH} \
        --resize_height=640 \
        --resize_width=640 \
        --shuffle=True \
        --bgr2rgb=False \
        --gray=False
  ```
  结果lmdb将保存的`${LMDB_PATH}`中


## 4.模型转换

模型转换的过程需要在x86下的docker开发环境中完成。fp32模型的运行验证可以在挂载有PCIe加速卡的x86-docker开发环境中进行，也可以在盒子中进行，且使用的原始模型为JIT模型。下面以YOLOX-s为例，介绍如何完成模型的转换。

### 4.1 生成FP32 BModel

#### **4.1.1 生成FP32 BModel**

  ```bash
  python3 -m bmnetp --net_name=yolox_s --target=BM1684 --opt=1 --cmp=true --shapes="[1,3,640,640]" --model=${OST_MODEL_NAME} --outdir=${OUTPUT_MODEL_PATH} --dyn=false
  ```
  其中 `${OST_MODEL_NAME}` 表示原始模型的路径及名称,结果会在`${OUTPUT_MODEL_PATH}`文件夹下面生成,文件夹内的compilation.bmodel即为fp32 bmodel


#### **4.1.2 查看FP32 BModel**

  此步骤可以在开发的docker内进行,也可以在盒子上进行

  ```bash
  bm_model.bin --info ${BModel_NAME}
  ```
  使用`bm_model.bin --info`查看的模型具体信息如下：

  ```bash
  bmodel version: B.2.2
  chip: BM1684
  create time: Tue Mar 29 12:04:18 2022

  ==========================================
  net 0: [yolox_s]  static
  ------------
  stage 0:
  input: x.1, [1, 3, 640, 640], float32, scale: 1
  output: 15, [1, 8400, 85], float32, scale: 1
  ```

#### **4.1.3 转换精度验证**

- 在yolox的源码下面生成原始模型的推理结果

  此步骤是使用经过trace的pytorch模型，在pytorch框架下面随机产生一批数据进行推理，然后保存推理的feature map

  ```docker内部已经安装了pytorch，但是由于版本的问题，yolox的源码在docker内执行会报错```

  将scirpts/pytorch.py文件拷贝至yolox源码目录下，执行下面命令

  ```bash
  python3 pytorch.py --model_path=${OST_MODEL_NAME} --feature_savepath=${FEATURE_SAVE_PATH}
  ```
  `${OST_MODEL_NAME}`表示torch模型名称，生成的feature map将保存在`${FEATURE_SAVE_PATH}`文件夹

- 在挂载有PCIe加速卡的x86 dev docker开发环境或者SE/SM微服务器上使用fp32 bmodel推理验证结果

  将上一步模型的推理结果拷贝至docker或者边缘盒子中,拷贝之后的路径为`${TORCH_FEATURE_PATH}`

  ```bash
  python3 python/sail.py –bmodel_path=${FP32_BMODEL_NAME} –feature_savepath=${TORCH_FEATURE_PATH} –max_error=0.00001
  ```

  如果打印`Verification successed!`则表示转换成功,否则转换失败

  参数说明：

    bmodel_path：fp32 bmodel路径

    feature_savepath：torch模型推理结果路径

    max_error：最大误差,如果不指定,默认0.00001


### 4.2 生成INT8 BModel

此过程需要在x86下的docker开发环境中完成，不量化模型可跳过本节。

INT8 BModel的生成需要经历中间格式UModel，即：原始模型→FP32 UModel→INT8 UModel→INT8 BModel。

#### **4.2.1 生成FP32 UModel**

  执行以下命令，将依次调用以下步骤中的脚本，生成INT8 BModel：

  ```bash
  python3 gen_fp32_umodel.py \
    --trace_model=${OST_MODEL_NAME} \
    --data_path=${LMDB_PATH}/data.mdb \
    --dst_width=640 \
    --dst_height=640
  ```
  结果将在`${OST_MODEL_NAME}`的文件夹下面创建一个以`${OST_MODEL_NAME}`模型名称命名的文件夹`${UMODEL_PATH}`，文件夹内存放的是fp32 umodel。

#### **4.2.2 修改FP32 UModel**

  如果输出成量化成int8会导致最终结果无法检出，所以最后一层不量化。在生成fp32umodel之后，会生成一个*bmnetp_test_fp32.prototxt，修改此文件，使网络最后的输出类型为float，具体操作为在最后一层中添加“forward_with_float:true”。

  修改之后最后一层如下：
```
layer {
  name: "15"
  type: "Transpose"
  bottom: "< 1 >137"
  top: "15"
  tag: "layer-1418@#aten::permute"
  transpose_param {
    order: 0
    order: 2
    order: 1
    order: 0
    order: 0
    order: 0
    order: 0
    order: 0
  }
  forward_with_float: true
}
```

#### **4.2.3 生成INT8 UModel**

  ```bash
  calibration_use_pb \
    quantize \
    -model=${UMODEL_PATH}/*_bmnetp_test_fp32.prototxt \
    -weights=${UMODEL_PATH}/*_bmnetp.fp32umodel \
    -iterations=100 \
    -bitwidth=TO_INT8
  ```
  ```注意：不同的模型的bmnetp_test_fp32.prototxt和bmnetp.fp32umodel文件名称不同,实际使用时需要替换命令行中的*```

  ```int8 umodel将保存在${UMODEL_PATH}文件夹下```

#### **4.2.4 生成INT8 BModel**

  ```bash
  bmnetu 
    -model=${UMODEL_PATH}/*_bmnetp_deploy_int8_unique_top.prototxt \
    -weight=${UMODEL_PATH}*_bmnetp.int8umodel \
    -max_n=4 \
    -prec=INT8 \
    -dyn=0 \
    -cmp=1 \
    -target=BM1684 \
    -outdir=${OUTPUT_BMODEL_PATH}
  ```
  ```注意：不同的模型的bmnetp_deploy_int8_unique_top.prototxt和bmnetp.int8umodel文件名称不同,实际使用时需要替换命令行中的*```

  ```命令参数中max_n表示生成模型的batchsize,结果bmodel将保存在${OUTPUT_BMODEL_PATH}下```

#### **4.2.4 查看INT8 BModel**

此步骤可以在开发的docker内进行,也可以在盒子上进行

  ```bash
    bm_model.bin --info ${BModel_NAME}
  ```
使用`bm_model.bin --info`查看的模型具体信息如下：

```bash
bmodel version: B.2.2
chip: BM1684
create time: Wed Mar 30 19:24:41 2022

==========================================
net 0: [yolox_s.trace_bmnetp]  static
------------
stage 0:
input: x.1, [4, 3, 640, 640], int8, scale: 0.498161
output: 15, [4, 8400, 85], float32, scale: 1
```

### 4.3 使用脚本快速生成INT8 BMODEL

```bash
  ./auto_gen.sh ${SDK_PATH} ${OST_DATA_PATH} ${OST_MODEL_NAME} ${NET_WIDTH} ${NET_HEIGHT} ${BATCH_SIZE} ${WITH_VPP_TPU}
```

参数说明：

  `${SDK_PATH}`: docker中SDK的路径

  `${OST_DATA_PATH}`: 原始量化图片的路径

  `${OST_MODEL_NAME}`: JIT模型名称

  `${NET_WIDTH}`: 网络输出宽度

  `${NET_HEIGHT}`: 网络输入的高度

  `${BATCH_SIZE}`: 生成bmodel的batch size

  `${WITH_VPP_TPU}`: 是否使用VPP和TPU对数据进行resize and padding

最终输出`Passed: convert to int8 bmodel`表示转换成功,否则转换失败

结果将在`${OST_MODEL_NAME}`所在目录中创建一个和原始模型同名的文件夹,并且内部将生成一个`bmodel_int8_bs${BATCH_SIZE}`,文件夹内的bmodel即为结果模型

## 5. 部署测试

请注意根据您使用的模型，所有例程都使用sail进行推理，内部对batch size和int8 or fp32做了自适应。

### 5.1 环境配置

#### **5.1.1 x86 PCIe**

对于x86 with PCIe加速卡平台，程序执行所需的环境变量执行`source envsetup_pcie.sh`时已经配置完成。

#### **5.1.2 arm SoC**

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

详细步骤参考cpp_sail下Readme.md

### 5.3 Python例程部署测试

详细步骤参考py_sail下Readme.md

### 5.4 使用脚本对示例代码进行自动测试

此自动测试脚本需要在挂载有PCIe加速卡的x86-pice-docker内进行

自动测试脚本会使用curl自动下载模型，所以如果没有安装curl的话需要先进行安装：

```bash
apt-get install curl
```

配置好环境变量安装好对应版本的sail之后执行：

```bash
./auto_test.sh ${SDK_PATH}
```
其中`${SDK_PATH}`指SDK的路径，如果最终输出 `Failed:`则表示执行失败，否则表示成功。