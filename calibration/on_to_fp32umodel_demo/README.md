#on_to_fp32umodel_demo
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
cd {EXAMPLES_TOP}/calibration/on_to_fp32umodel_demo
```
##运行脚本
```bash
python3 postnet_to_umodel.py   #指定当前虚拟环境的python解释器来运行脚本
```
##执行结果
在当前目录生成一个compilation文件夹，内容包括:
- io_info.dat
- *.fp32umodel
- *.prototxt
