#!/bin/bash

mkdir -p output/lenet
rm -rf output/lenet/*

# --model 指定模型结构文件
# --weight 指定模型权重文件
# --target 指定目标设备
# --dyn 是否是动态编译, 可选，默认是False
# --outdir 指定输出目录, 可选，默认是compilation
# --opt 编译优化级别：0,1,2, 可选，默认是2
# --net_name 设置网络名称，可选，默认是network
python3 -m bmnetm --model=models/lenet/lenet-symbol.json \
                  --weight=models/lenet/lenet-0100.params \
                  --shapes=[1,1,28,28] \
                  --target=BM1684 \
                  --net_name=lenet \
                  --outdir=output/lenet \
                  --opt=2 \
                  --dyn=False

