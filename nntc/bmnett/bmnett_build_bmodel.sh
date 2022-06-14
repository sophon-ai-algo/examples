#!/bin/bash

mkdir -p output/vqvae
rm -rf output/vqvae/*

# --model 指定模型结构文件
# --input_names 指定输入名称
# --shapes 指定网络输入shapes
# --target 指定目标设备
# --output_names 指定输出名称，默认以所有悬空的tensor作为输出
# --dyn 是否是动态编译, 可选，默认是False
# --outdir 指定输出目录, 可选，默认是compilation
# --opt 编译优化级别：0,1,2, 可选，默认是2
# --net_name 设置网络名称，可选，默认是network
# --cmp 是否开启比对，可选，默认开启

python3 -m bmnett --model=models/vqvae/vqvae.pb \
                  --input_names=Placeholder \
                  --shapes=[1,90,180,3] \
                  --target=BM1684 \
                  --output_names=valid/forward/ArgMin \
                  --net_name=vqvae \
                  --opt=2 \
                  --outdir=output/vqvae \
                  --dyn=false
