#object_detection_python_demo
-----
请先进入docker环境，再执行下述操作
## 配置运行环境
```bash
cd bmnnsdk2-<version>/scripts  #进入对应版本的sdk脚本文件夹
./install_lib.sh nntc #安装nntoolchain
source ./envsetup_cmodel.sh  #配置不同平台的运行环境
```
##运行脚本
```bash
cd ../examples/calibration/object_detection_python_demo
python3 ssd_vgg300_fp32_test.py #生成person_fp32_detected.jpg
python3 ssd_vgg300_int8_test.py #生成person_int8_detected.jpg
```