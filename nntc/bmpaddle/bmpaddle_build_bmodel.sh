#!/bin/bash

mkdir -p output/ch_ppocr_mobile_v2
rm -rf output/ch_ppocr_mobile_v2/*

# --model 指定模型结构文件
# --shapes 指定网络输入shapes
# --target 指定目标设备
# --dyn 是否是动态编译, 可选，默认是False
# --outdir 指定输出目录, 可选，默认是compilation
# --opt 编译优化级别：0,1,2, 可选，默认是2
# --net_name 设置网络名称，可选，默认是network
# --cmp 是否开启比对，可选，默认开启

python3 -m bmpaddle --model=models/ch_ppocr_mobile_v2.0_cls_infer \
                    --shapes="[1,3,32,100]" \
                    --target=BM1684 \
                    --opt=2 \
                    --cmp=true \
                    --outdir=output/ch_ppocr_mobile_v2 \
                    --net_name=ocr-cls \
                    --dyn=false

