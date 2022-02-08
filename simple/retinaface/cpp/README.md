# BMRetinaFaceDemo
这个是在比特大陆BM1684系列AI硬件(SC5+/SE5/SM5)上运行RetinaFace人脸检测模型的demo

1. 前提
  
    已有了bmnnsdk2-bm1684_v2.x.x的sdk，并且安装完lib。

2. 使用说明
 
    a)  打开makefile，正确设置: top_dir ?= /workspace, sdk_dir = $(top_dir)/, install_dir = $(top_dir)/install
  
    b)  编译程序
    
        如果是SC5+板卡 : make -f Makefile.pcie -j4
    
        如果是SE5或SM5 : make -f Makefile.arm -j4
  
    c)  运行程序
        
        如果是图片模式， ./face_test 0 imagelist.txt ../data/models/face_detection_fp32_b1.bmodel
        
        如果是视频模式， ./face_test 1 videolist.txt ../data/models/face_detection_int8_b4.bmodel
        
        * 1 imagelist.txt的每一行是图片的路径
        * 2 videolist.txt的每一行是一个mp4视频路径或者一个rtsp url
        * 3 如果模型是多batch的，那么图片模式，会每攒够batch数的图片做一次推理。视频模式，
            要求videolist.txt的视频数和模型的batch数相等。当然也可以按着自己需求的逻辑来
            修改。只是demo里这样写的。
  
    d） 结束后会在当前目录生成一个名为result_imgs的文件夹，里面可以看到结果图片
 
