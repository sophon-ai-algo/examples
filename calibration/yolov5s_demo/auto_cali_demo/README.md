#yolov5 auto_cali_demo
-----
请先进入docker环境，再执行下述操作
## 配置运行环境
```bash
cd <sdk_path>/scripts  #进入对应版本的sdk脚本文件夹
./install_lib.sh nntc #安装nntoolchain
source ./envsetup_cmodel.sh  #配置不同平台的运行环境
```
##运行脚本
```bash
本例演示使用auto_cali量化yolov5s网络，需要先下载coco128数据集制作量化数据，如果已经执行过create_lmdb_demo可跳过：
cd {EXAMPLES_TOP}/calibration/create_lmdb_demo
# 一定要先下载数据集
bash download_coco128.sh

执行auto_cali量化网络：
cd {EXAMPLES_TOP}/calibration/yolov5s_demo/auto_cali_demo
bash ./auto_cali.sh

执行结果打印：
  calib finished。
在当前目录的yolov5s目录下生成compilation.bmodel。
```
