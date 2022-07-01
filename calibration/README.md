# calibration examples

本examples包含如下子模块：

- yolov5s_demo，包含使用auto_cali量化yolov5s和使用可视化工具查看yolov5s量化结果两个demo
- caffemodel_to_fp32model_demo, 使用ufw将 caffe 网络编译为fp32umodel的demo
- dn_to_fp32umodel_demo, 使用ufw将 darknet 网络编译为fp32umodel的demo
- mx_to_fp32umodel_demo, 使用ufw将 mxnet 网络编译为fp32umodel的demo
- on_to_fp32umodel_demo, 使用ufw将 onnx 网络编译为fp32umodel的demo
- pp_to_fp32umodel_demo, 使用ufw将 paddlepaddle 网络编译为fp32umodel的demo
- pt_to_fp32umodel_demo, 使用ufw将 pytorch 网络编译为fp32umodel的demo
- tf_to_fp32umodel_demo, 使用ufw将 tensorflow 网络编译为fp32umodel的demo
- create_lmdb_demo, 使用ufwio制作lmdb数据集的demo
- classify_demo, 分类网络量化的demo
- face_demo, 人脸检测网络量化，量化网络后可以把图片上人脸框出
- object_detection_python_demo, python物体检测demo

## 说明
本目录下的demo都设计在docker容器中的sophonsdk3中工作，关于如何选择正确的docker image和sophonsdk版本
请参考sophonsdk3的在线入门文档：[https://sophgo-doc.gitbook.io/sophonsdk3/](https://sophgo-doc.gitbook.io/sophonsdk3/)

## 准备工作
启动和配置sophonsdk3的docker容器以sophonsdk3文档为准，如果运行本目录中的demo需要在启动容器进入sophonsdk3
根目录后执行以下操作:
  1. cd scripts
  2. ./install_lib.sh nntc
  3. source ./envsetup_cmodel.sh
本目录中的demo需要较多数据和原始网络作为演示，所以在准备好以上环境后进入本目录，然后运行
  bash ./prepare.sh
此脚本将下载examples所需模型和数据。运行某个demo之前请首先阅读其中的README.md，下载或准备更多的数据和资料。

