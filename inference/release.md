# 运行说明
## 1. 环境部署

### 1.1 PCIe模式
首先，需要从官网下载`docker`镜像和`BMNNSDK2`开发包，可以从[算能官网](https://sophgo-doc.gitbook.io/bmnnsdk2-bm1684/bmnnsdk2/get)下载获得。BMNNSDK2的安装方式可以查看[官方文档](https://sophgo-doc.gitbook.io/bmnnsdk2-bm1684/bmnnsdk2/setup/on-linux)的PCIe部分

docker、驱动和sdk安装完后，启动并进入容器
```bash
cd /workspace/scripts
# 初始化环境
source envsetup_pcie.sh
```

### 1.2 SoC模式
以SE5智算盒为例，其中已经内置了运行库等，用户只需要初始化环境变量
```bash
export PATH=$PATH:/system/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/system/lib/:/system/usr/lib/aarch64-linux-gnu
export PYTHONPATH=$PYTHONPATH:/system/lib
```

## 2. 运行

### 2.1 Demo说明
release文件夹下包含了`可执行程序`，`配置文件`, `模型文件`，具体程序如下
难易程度 | 目录 | 说明 | 模型个数
|---|---|---|---|
入门 | [face_detect](https://github.com/sophon-ai-algo/examples/tree/main/inference/examples/face_detect) | Sequeezenet 人脸检测  | 1
入门 | [retinaface](https://github.com/sophon-ai-algo/examples/tree/main/inference/examples/retinaface) | Retinaface 人脸检测 | 1
入门 | [yolov5](https://github.com/sophon-ai-algo/examples/tree/main/inference/examples/yolov5) | yolov5s 对象检测 | 1
进阶 | [openpose](https://github.com/sophon-ai-algo/examples/tree/main/inference/examples/openpose) | OpenPose 18/25个关键点 | 1
进阶 | [multi](https://github.com/sophon-ai-algo/examples/tree/main/inference/examples/multi)  | 两个yolov5模型通过配置文件并行 | 2
高级 | [face_recognition](https://github.com/sophon-ai-algo/examples/tree/main/inference/examples/face_recognition) | 演示多个模型如何串联 | 3
高级 | [video_stitch](./examples/video_stitch) | 4路检检测+拼接+编码+RTSP服务 | 1

每个demo文件夹下，有一个`run.sh`的启动脚本，用户根据PCIe和SoC模式分别运行
```bash
# PCIe模式
run.sh x86
# SoC模式
run.sh soc
```
> 此外，配置文件cameras.json里包含了接入的视频流文件, 由`address`配置项决定。
> 用户可以配置`RTSP`、`纯264文件`、`纯265文件`作为视频输入。


**每个demo可能包含特定的命令行参数，以yolov5s_demo为例，用户可以通过执行./x86/yolov5s_demo --help具体查看**


### 2.2 结果可视化
对于大多数demo，--help中带有`--output`参数，代表服务端可以向指定地址推流，并且客户端负责接收带有结果的流，然后显示。
具体配置方法，请参考[SE5-OpenPose-Demo配置文档.docx](https://github.com/sophon-ai-algo/examples/blob/main/inference/SE5-OpenPose-Demo-Config.docx)  

#### 2.2.1 windows
解压`face_demo-winx64.zip`，里面有编译好的exe客户端

#### 2.2.2 Ubuntu&Mac
从源码编译, 请参考如下链接：https://github.com/sophon-ai-algo/face_demo_client
