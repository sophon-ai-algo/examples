# JPEG/H264/HEVC 编码相关例子介绍
## 1 总体介绍
本模块包含三个测试例子，功能如下：
|名称 | 描述 |
|---|---|
|simple_ffmpeg_jpeg_enc_from_file | 演示读取文件后进行JPEG编码的实现过程 
| simple_ffmpeg_video_enc | 演示读取YUV视频流，然后编码成264/265的实现过程。
| simpple_ffmpeg_video_jpeg_dec | 演示常规的视频或者图片解码的实现流程

具体的实现细节请参考实现代码。

## 2 编译方法
本模块编译采用了cmake进行编译，请使用者自行安装cmake，此处不提供。  
需要提前设置好环境变量：export REL_TOP=$bmnnsdk2_dir 此处要求是实际的bmnnsdk2的地址。     
ubuntu平台编译指令：
``` bash 
mkdir build
cmake -DPRODUCTFROM=x86 ..
make -j4
```
注意事项：
> PRODUCTFORM：为平台的类型，可选为：x86 | soc | arm_pcie | sw64 | loongarch64  

## 3 运行方法

> 请参考各个程序的帮助指令。
