#face_demo
-----
请先进入docker环境，再执行下述操作
## 配置运行环境
```bash
cd sophonsdk3-<version>/scripts  #进入对应版本的sdk脚本文件夹
./install_lib.sh nntc #安装nntoolchain
source ./envsetup_cmodel.sh  #配置不同平台的运行环境
```
##运行脚本
```bash
cd ../examples/calibration/face_demo
source face_demo.sh
detect_squeezenet_fp32 #执行后打印：
   final predict 19 bboxes
   并生成detection.png，其中19张人脸被正确框出。
convert_squeezenet_to_int8
#执行后生成squeezenet_21k_deploy_int8_unique_top.prototxt和squeezenet_21k_deploy_fp32_unique_top.prototxt
detect_squeezenet_int8 #执行后打印：
   final predict 19 bboxes
   并生成detection_int8.png，其中19张人脸被正确框出。
```
