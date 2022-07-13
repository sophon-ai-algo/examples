#caffemodel_to_fp32umodel_demo
-----
请先进入docker环境，再执行下述操作
## 配置运行环境
```bash
cd <sdk_path>/scripts  #进入对应版本的sdk脚本文件夹
./install_lib.sh nntc #安装nntoolchain
source ./envsetup_cmodel.sh  #配置不同平台的运行环境
```
##进入examples所在目录
```bash
cd {EXAMPLES_TOP}/calibration/caffemodel_to_fp32umodel_demo
```
##运行脚本
```bash
python3 resnet50_to_umodel.py  #指定当前虚拟环境的python解释器来运行脚本
```
##执行结果
当前文件夹下会生成compilation文件夹并在其中包含转换结果：
- compilation文件夹
  - *.fp32umodel
  - *.prototxt
