Inference framework based BMNNSDK2.
> Ubuntu 16.04 安装QT依赖：
````
sudo apt install qtbase5-dev
````

> Tracker 功能需要安装 Eigen 依赖：
```
sudo apt-get install -y libeigen3-dev
```

> Retinaface 模块需要安装 glog/exiv2 依赖：
```
sudo apt-get install -y libgoogle-glog-dev libexiv2-dev
```

1. export REL_TOP=$bmnnsdk_dir 根据实际位置修改为BMNNSDK跟路径
2. 各个平台编译

   > TARGET_ARCH=x86 表示x86平台
   TARGET_ARCH=soc 表示小盒子上编译
   TARGET_ARCH=arm-pcie 表示国产ARM CPU上编译
   
   > appname=face_detect 人脸检测
     appname=openpose    人体骨骼检测
     appname=yolov5      物体检测
     appname=retinaface  人脸检测
     
   ```` 
   ./compile.sh [appname] [target_arch]
   example: ./compile.sh face_detect x86-pcie
   
3. 运行方法
   > cd ./release/face_detect

   > ./x86/example_demo --help 查看命行帮助信息     

4. 相关模型请从下面网盘下载，如有问题，请联系技术支持 
   
   链接：https://pan.baidu.com/s/16d5E_NTj4ubVPkPmR6GG5A 
   提取码：sp2w 
