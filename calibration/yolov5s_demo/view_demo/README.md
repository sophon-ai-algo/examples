# create_lmdb_demo
-----
请先进入docker环境，再执行下述操作
## 配置运行环境
```bash
cd bmnnsdk2-<version>/scripts  #进入对应版本的sdk脚本文件夹
./install_lib.sh nntc #安装nntoolchain
source ./envsetup_cmodel.sh  #配置不同平台的运行环境
```
## 运行脚本
```bash
bash ./jupyter_server.sh
```
## 执行结果
在浏览器输入localhost:80 正常打开可视化工具，注：如果使用此可视化工具，需要在docker创建的时候将端口映射到主机。