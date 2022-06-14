#!/bin/bash

mkdir -p output/mobilenet
rm -rf output/mobilenet/*

# --model 指定模型结构文件
# --weight 指定模型权重文件
# --target 指定目标设备
# --shapes 设置输入shape，可选，默认用模型里的shape
# --dyn 是否是动态编译, 可选，默认是False
# --outdir 指定输出目录, 可选，默认是compilation
# --opt 编译优化级别：0,1,2, 可选，默认是2
# --net_name 设置网络名称，可选，默认是network
# --cmp 是否开启比对，可选，默认开启

bmnetu --model models/mobilenet/mobilenet_deploy_int8_unique_top.prototxt \
       --weight models/mobilenet/mobilenet.int8umodel \
       --target=BM1684 \
       --shapes=[1,3,224,224] \
       --dyn=0  \
       --outdir=output/mobilenet \
       --cmp=1
