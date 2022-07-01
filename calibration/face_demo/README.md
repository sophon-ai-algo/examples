#face_demo
-----
请先进入docker环境，再执行下述操作
## 配置运行环境
```bash
cd <sdk_path>/scripts  #进入对应版本的sdk脚本文件夹
./install_lib.sh nntc #安装nntoolchain
source ./envsetup_cmodel.sh  #配置不同平台的运行环境
```
##进入examples所目录
```bash
cd {EXAMPLES_TOP}/calibration/face_demo
```
##运行脚本
```bash
source face_demo.sh
detect_squeezenet_fp32 #测试fp32网络，执行后打印：
   final predict 19 bboxes
   并生成detection.png，其中19张人脸被正确框出。
convert_squeezenet_to_int8 #量化fp32网络为int8网络，执行后
                           #在models/squeezenet目录下生成
                           #squeezenet_21k_deploy_int8_unique_top.prototxt和squeezenet_21k_deploy_fp32_unique_top.prototxt
                           #以及squeezenet_21k.int8umodel
detect_squeezenet_int8 #使用量化生成的int8网络进行测试，执行后打印：
   final predict 19 bboxes
   并生成detection_int8.png，其中19张人脸被正确框出。
```
