#!/bin/bash

mkdir -p output/yolov3-tiny
rm -rf output/yolov3-tiny/*

# --model 指定模型结构文件
# --weight 指定模型权重文件
# --target 指定目标设备
# --dyn 是否是动态编译, 可选，默认是False
# --outdir 指定输出目录, 可选，默认是compilation
# --opt 编译优化级别：0,1,2, 可选，默认是2

bmnetd --model=models/yolov3-tiny/yolov3-tiny.cfg \
       --weight=models/yolov3-tiny/yolov3-tiny.weights \
       --dyn=false \
       --target=BM1684 \
       --outdir=output/yolov3-tiny \
       --opt=2
