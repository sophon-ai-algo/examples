#!/bin/bash

mkdir -p output/yolov5s
rm -rf output/yolov5s/*

# --model 指定模型结构文件
# --shape 指定模型输入shape
# --target 指定目标设备
# --outdir 指定输出目录, 可选，默认是compilation
# --net_name 设置网络名称，可选，默认是network
# --opt 编译优化级别：0,1,2, 可选，默认是2
# --dyn 是否是动态编译, 可选，默认是False
python3 -m bmneto --model=models/yolov5s/yolov5s.onnx \
                  --shapes="[1,3,640,640]" \
                  --target=BM1684 \
                  --outdir=output/yolov5s \
                  --opt=2 \
                  --net_name=yolov5s \
                  --dyn=false
