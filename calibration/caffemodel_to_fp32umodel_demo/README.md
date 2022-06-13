#caffemodel_to_fp32umodel_demo
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
cd ../examples/calibration/caffemodel_to_fp32umodel_demo
python3 resnet50_to_umodel.py  #指定当前虚拟环境的python解释器来运行脚本
```
##执行结果
当前文件夹下会生成如下3个文件：
- compilation文件夹
- *.fp32umodel
- *.prototxt
