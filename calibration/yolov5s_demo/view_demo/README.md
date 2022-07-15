# yolov5s_view_demo
-----
请先进入sdk所目录，在启动docker脚本中增加-p 8080:8080 的端口映射配置，再执行下述操作。
##docker启动脚本示例
```bash
  CMD="docker run \
      --network=host \
      --workdir=/workspace \
      --privileged=true \
      ${mount_options} \
      --device=/dev/bmdev-ctl:/dev/bmdev-ctl \
      -v /dev/shm --tmpfs /dev/shm:exec \
      -v $WORKSPACE:/workspace \
      -v /dev:/dev \
      -p 8080:8080 \
      -v /etc/localtime:/etc/localtime \
      -e LOCAL_USER_ID=`id -u` \
      -itd $REPO/$IMAGE:$TAG \
      bash
  "

  container_sha=`eval $CMD`
  container_id=${container_sha:0:12}
  CMD="docker exec -it ${container_id} bash"

  [[ ! -z "$container_id" ]] && eval $CMD
```

## 配置运行环境
```bash
cd <sdk_path>/scripts  #进入对应版本的sdk脚本文件夹
./install_lib.sh nntc #安装nntoolchain
source ./envsetup_cmodel.sh  #配置不同平台的运行环境
```
## 首先参照 {EXAMPLES_TOP}/calibration/yolov5s_demo/auto_cali_demo/README.md描述将网络进行量化。

## 进入auto_cali量化网络所在目录
```bash
cd {EXAMPLES_TOP}/calibration/yolov5s_demo/auto_cali_demo
```
## 运行脚本
```bash
python3 -m ufw.tools.app -p 8080
```
## 执行结果
在浏览器输入：容器镜像计算机ip:8080 正常打开可视化工具显示yolov5s网络。

注：如果使用此可视化工具，需要在docker创建的时候将端口映射到主机。
