#create_lmdb_demo
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
cd {SDK_TOP}/examples/calibration/create_lmdb_demo
# 一定要先下载数据集
bash download_coco128.sh
cd ../yolov5s_demo/auto_cali_demo
bash ./auto_cali.sh
pip3 install pycocotools
python3 -m dfn --url https://disk.sophgo.vip/sharing/ivVtP2yIg
bash ./regression.sh
```
