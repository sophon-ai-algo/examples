# YOLOv3/v4 demo for bmopencv decode + bmcv preprocess

## 1.开发环境:

开发环境是指用于模型转换或验证以及程序编译等开发过程的环境，目前只支持x86，需要使用我们提供的基于Ubuntu16.04的docker镜像。

运行环境是具备Sophon设备的平台上实际使用设备进行算法应用部署的环境，有PCIe加速卡、SM5模组、SE5边缘计算盒子等，所有运行环境上的BModel都是一样的，SDK中各模块的接口也是一致的。

开发环境与运行环境可能是统一的（如插有SC5加速卡的x86主机，既是开发环境又是运行环境），也可能是分离的（如使用x86主机作为开发环境转换模型和编译程序，使用SE5盒子部署运行最终的算法应用）。

但是，无论使用的产品是SoC模式还是PCIe模式，都需要一台x86主机作为开发环境，模型的转换工作必须在开发环境中完成。

### 1.1 开发主机准备

开发主机：一台安装了Ubuntu16.04/18.04/20.04的x86主机，运行内存建议12GB以上

安装docker：参考《[官方教程](https://docs.docker.com/engine/install/)》，若已经安装请跳过

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

### 1.2 SDK软件包下载

- 开发docker基础镜像：[点击前往官网下载Ubuntu开发镜像](https://developer.sophgo.com/site/index/material/11/all.html)，Ubuntu 18.04 with Python 3.7

```bash
wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/22/03/19/13/bmnnsdk2-bm1684-ubuntu-docker-py37.zip
```

- SDK软件包：[点击前往官网下载SDK软件包](https://developer.sophgo.com/site/index/material/17/all.html)，BMNNSDK 2.7.0_20220316_022200

```bash
wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/22/04/14/10/bmnnsdk2_bm1684_v2.7.0_20220316_patched_0413.zip
```

### 1.3 创建docker开发环境

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
source envsetup_pcie.sh
```

## 2. 模型与数据:

- 模型：[yolov4_608_coco_fp32_1b.bmodel](http://219.142.246.77:65000/sharing/EO0XZJNnH)

- 数据：[test images and videos](http://219.142.246.77:65000/sharing/pnZylgE2T)

**备注**：[test images and videos](http://219.142.246.77:65000/sharing/pnZylgE2T)点击【下载文件夹】下载data.zip，解压data.zip后在data文件夹下的images文件夹包含测试图片和记录图片相对路径的`imageslist.txt`文件和videos文件夹包含测试视频和记录视频相对路径的`videolist.txt`文件。使用时，需要将data文件夹放到运行目录下，\<image list\>或\<video url\>指定为images路径下的`imageslist.txt`或videos路径下的`videolist.txt`即可完成测试。

## 3. 编译:

```bash
cd /workspace/examples/YOLOv3_object/cpp_cv_bmcv_bmrt_postprocess
```

* x86 pcie

```shell
make -f Makefile.pcie

# yolo_test.pcie will be generated
```
* arm pcie

```shell
make -f Makefile.arm_pcie

# yolo_test.arm_pcie will be generated
```
* SOC

```shell
make -f Makefile.arm

# yolo_test.arm will be generated, copy the file to soc product and run
```
* x86 pcie

```shell
make -f Makefile.mips

# yolo_test.mips will be generated
```

## 4. 使用:

```shell
# image list or video url is a txt file with image path or video path in each line
# yolo_text.xxx differs on different platform
./yolo_test.xxx image <image list> <bmodel file> 
./yolo_test.xxx video <video url>  <bmodel file>

# result images with drawn prediction bboxes will be saved in result_imgs directory.
```

**备注**：\<image list\>和\<video url\>均为记录图片或视频路径的`txt`文件

image list的`txt`文件内容示例：

```bash
# 测试图片路径
data/images/dog.jpg
data/images/horse.jpg
```

video url的`txt`文件内容示例

```bash
# 测试视频路径
data/videos/dance.mp4
```

**使用下载的测试数据进行测试**：

```bash
# 将模型和data文件夹放到运行目录下
# image
./yolo_test.xxx image ./data/images/imagelist.txt yolov4_608_coco_fp32_1b.bmodel

# video
./yolo_test.xxx image ./data/videos/videolist.txt yolov4_608_coco_fp32_1b.bmodel
```
