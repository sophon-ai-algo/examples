#auto_cali
-----
请先进入docker环境，再执行下述操作
## 配置运行环境
```bash
cd bmnnsdk2-<version>/scripts  #进入对应版本的sdk脚本文件夹
./install_lib.sh nntc #安装nntoolchain
source ./envsetup_cmodel.sh  #配置不同平台的运行环境
```
##解压缩测试数据
进入examples/calibration/auto_cali/test_models目录，执行extract_tar.sh解压缩测试文件

##运行脚本
```bash
python3 -m ufw.cali.cali_model --model pytorch/resnet18.pt  --cali_lmdb imagenet_preprocessed_by_pytorch_100/ --input_shapes '(1,3,224,224)' --test_iterations 50 --net_name resnet18  --postprocess_and_calc_score_class topx_accuracy_for_classify --cali_iterations=100
```
##执行结果
在test_models/pytorch/下生成相应bmodel

