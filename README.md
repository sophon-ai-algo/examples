## BMMNSDK2样例仓介绍

BMNNSDK2是算能科技基于其自主研发的 AI 芯片所定制的深度学习SDK，涵盖了神经网络推理阶段所需的模型优化、高效运行时支持等能力，为深度学习应用开发和部署提供易用、高效的全栈式解决方案。

算能样例仓就是以BMNNSDK2接口进行开发，制作的一系列给开发者进行参考学习的样例。在开发者朋友们开发自己的样例时，也可以就样例仓的相关案例进行参考。

## 版本说明

**master分支样例版本适配情况请参见[样例表单及适配说明](#Version-of-samples)。     
历史版本请参考[表1 版本说明](#Version-Description)下载对应发行版**。

**表1** 版本说明<a name="Version-Description"></a>
| BMNNSDK2版本 | 否维护 | Examples 获取方式 |
|---|---|---|
| [BMNNSDK2-2.7.0](https://sophon-file.sophon.cn/sophon-prod-s3/drive/22/03/18/11/bmnnsdk2_bm1684_v2.7.0.zip) | 是 |  |
| [BMNNSDK2-2.6.0](https://sophon-file.sophon.cn/sophon-prod-s3/drive/21/12/16/16/bmnnsdk2_bm1684_v2.6.0.zip) | 是 | Release 2.6.0发行版，[点击跳转](https://github.com/sophon-ai-algo/examples/releases/v2.6.0) |


## 目录结构与说明
| 目录 | 说明 |
|---|---|
| [inference](./inference) | 简易的推理框架及其使用示例 |
| [multimedia](./multimedia) | 基于ffmpeg/opencv的示例目录 |
| [pipeline](./pipeline) | 基于Pipeline的全流程示例目录 | 
| [simple](./simple) | 单个模型推理简单样例目录 |


## 使用指南

1.根据设备形态按按照如下步骤搭建合设环境

请参考在线入门文档：https://sophgo-doc.gitbook.io/bmnnsdk2-bm1684/

## 样例表单&适配说明<a name="Version-of-samples"></a>

| 样例名称 | 语言 | 适配BMNNSDK2版本 | 简介 |
|---|---|---|---|
| [YoloV5](./simple/yolov5) |  c++/python | >=2.6.0 | 使用bmcv/opencv做前处理，bmrt推理的示例程序 |
|[centernet](./simple/centernet) | c++/python | >=2.6.0 | CenterNet 推理示例，采用BMCV做前后处理。 |
|[retinaface](./simple/retinaface) | c++/python | >=2.6.0 | RetinaFace 推理示例，采用BMCV做前后处理。 |
|[yolox](./simple/yolox) | c++/python | >=2.6.0 | YOLOX 推理示例，采用BMCV做前后处理。 |
|[yolov4](./simple/yolov4) | c++/python | >=2.6.0 | YOLOV3/YOLOV4 推理示例，采用BMCV做前后处理。 |
## 文档

参考社区网站[产品文档](https://developer.sophon.cn/document/index.html)获取相关文档。

## 社区

算能社区鼓励开发者多交流，共学习。开发者可以通过以下渠道进行交流和学习。

算能社区网站：sophgo.com

算能开发者论坛：https://developer.sophon.cn/forum/view/43.html

算能官方qq群：【待发布】

## 贡献

欢迎参与贡献。更多详情，请参阅我们的[贡献者Wiki](./CONTRIBUTING_CN.md)。

## 许可证
[Apache License 2.0](LICENSE)