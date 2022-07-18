## SophonSDK样例仓介绍

SophonSDK是算能科技基于其自主研发的 AI 芯片所定制的深度学习SDK，涵盖了神经网络推理阶段所需的模型优化、高效运行时支持等能力，为深度学习应用开发和部署提供易用、高效的全栈式解决方案。

算能样例仓就是以SophonSDK接口进行开发，制作的一系列给开发者进行参考学习的样例。在开发者朋友们开发自己的样例时，也可以就样例仓的相关案例进行参考。

## 版本说明

**历史版本请参考[表1 版本说明](#Version-Description)下载对应发行版**。

**表1** <a name="Version-Description">版本说明</a>

| Examples分支 | 是否维护 | 适用SDK版本 |
|---|---|---|
| 3.0.0 | 是 | [SDK 3.0.0](https://sophon-file.sophon.cn/sophon-prod-s3/drive/22/07/18/11/sophonsdk_v3.0.0_20220716.zip) |
| 2.7.0 | 是 | [SDK-2.7.0](https://sophon-file.sophon.cn/sophon-prod-s3/drive/22/05/31/11/bmnnsdk2_bm1684_v2.7.0_20220531patched.zip) |
| 2.6.0 | 否 | [SDK-2.6.0](https://sophon-file.sophon.cn/sophon-prod-s3/drive/22/02/10/18/bmnnsdk2_bm1684_v2.6.0.zip) |


## 目录结构与说明
| 目录 | 说明 |
|---|---|
| [calibration](./calibration) | 量化工具使用示例 |
| [inference](./inference) | 简易的推理框架Inference framework及其使用示例 |
| [multimedia](./multimedia) | 基于ffmpeg/opencv的使用示例 |
| [nntc](./nntc) | NNToolChain模型转换工具使用示例 |
| [other](./other) | TODO，其他参考示例，如推流、websocket推图片流到web显示等 |
| [simple](./simple) | 单个模型推理示例 |


## 使用指南

1.根据设备形态按照如下步骤搭建合适环境

请参考在线入门文档：[https://sophgo-doc.gitbook.io/sophonsdk3/](https://sophgo-doc.gitbook.io/sophonsdk3/)

## 文档

参考社区网站[产品文档](https://developer.sophon.cn/site/index/document/all/all.html)获取相关文档。

## 社区

算能社区鼓励开发者多交流，共学习。开发者可以通过以下渠道进行交流和学习。

算能社区网站：https://www.sophgo.com/

算能开发者论坛：[https://developer.sophgo.com/forum/index.html](https://developer.sophgo.com/forum/index.html)

算能官方qq群：【待发布】

## 贡献

欢迎参与贡献。更多详情，请参阅我们的[贡献者Wiki](./CONTRIBUTING_CN.md)。

## 许可证
[Apache License 2.0](LICENSE)