**操作说明**

1. 在某一台SE5盒子能访问的主机上安装并启动RTSP Server：

   推荐EasyDarwin：http://www.easydarwin.org/

   如果想在盒子上使用，请下载源码编译

2. 运行Python程序，通过管道将待推送的图像数据发送到管道，由ffmpeg命令推流到RTSP/RTMP Server

   RTSP Server可使用部署的EasyDarwin RTSP Server

   RTMP Server可使用如下公开Server测试：rtmp://8.136.105.93:10035/app/test
