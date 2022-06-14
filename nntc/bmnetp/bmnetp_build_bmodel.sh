#!/bin/bash

mkdir -p output/anchors
rm -rf output/anchors/*

# --model 指定模型结构文件
# --target 指定目标设备
# --shapes 指定网络输入shapes
# --dyn 是否是动态编译, 可选，默认是False
# --outdir 指定输出目录, 可选，默认是compilation
# --opt 编译优化级别：0,1,2, 可选，默认是2
# --net_name 设置网络名称，可选，默认是network
# --cmp 是否开启比对，可选，默认开启
python3 -m bmnetp --model=models/anchors/anchors.pth \
                  --target=BM1684 \
                  --shapes="[3,100],[5,10]" \
                  --opt=1 \
                  --cmp=true \
                  --net_name=anchors \
                  --outdir=output/anchors \
                  --dyn=false
