#!/bin/bash

mkdir -p output/det2
rm -rf output/det2/*

# --model 指定caffe模型结构文件
# --weight 指定caffe模型权重文件
# --dyn 是否是动态编译
# --target 指定目标设备
# --outdir 指定输出目录
# --opt 编译优化级别：0,1,2
# --enable_profile 是否要记录profile信息到bmodel, 默认不记录
# --v 日志输出级别：0,1,2,3,4。0最少，4最详细

bmnetc --model=models/mtcnndet2/det2.prototxt \
       --weight=models/mtcnndet2/det2.caffemodel \
       --dyn=false \
       --target=BM1684 \
       --outdir=output/det2 \
       --opt=2 \
       --enable_profile=1 \
       --v=4
