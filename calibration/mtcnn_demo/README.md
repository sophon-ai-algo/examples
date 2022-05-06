#mtcnn_demo
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
cd ../examples/calibration/mtcnn_demo
source mtcnn_demo.sh
mtcnn_build #编译代码
dump_fddb_lmdb #生成各网络的lmdb数据集
#当前文件夹下生成pnet, rnet, onet的lmdb数据集
```
##量化各网络
```bash
convert_mtcnn_demo_pnet_to_int8_pb #量化pnet
convert_mtcnn_demo_rnet_to_int8_pb #量化rnet
convert_mtcnn_demo_onet_to_int8_pb #量化onet
#./models目录下生成量化后的模型
```

##运行demo
```bash
run_demo_float #运行fp32网络的demo
run_demo_int8  #运行int8网络的demo
```