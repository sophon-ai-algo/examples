#create_lmdb_demo
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
cd {EXAMPLES_TOP}/calibration/create_lmdb_demo
```
##运行脚本
```bash
bash download_coco128.sh
python3 convert_imageset.py \
        --imageset_rootfolder=./coco128/images/train2017 \
        --imageset_lmdbfolder=./lmdb \
        --resize_height=640 \
        --resize_width=640 \
        --shuffle=True \
        --bgr2rgb=False \
        --gray=False
```
##执行结果
在./lmdb目录生成一个data.mdb文件
