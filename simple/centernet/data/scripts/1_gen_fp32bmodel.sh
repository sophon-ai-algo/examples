#!/bin/bash

source model_info.sh

SCRIPT_DIR=`pwd`/`dirname $0`
MODEL_DIR=$SCRIPT_DIR/../models

python3 \
    -m bmnetp \
    --net_name=ctdet_dlav0 \
    --target=BM1684 \
    --opt=2 \
    --cmp=true \
    --enable_profile=true \
    --shapes=[1,3,512,512] \
    --model=$MODEL_DIR/ctdet_coco_dlav0_1x.torchscript.pt \
    --outdir=$fp32model_dir \
    --dyn=false

cp $fp32model_dir/compilation.bmodel $MODEL_DIR/${dst_model_prefix}_${img_size}_fp32_1batch.bmodel

echo "[Success] $MODEL_DIR/${dst_model_prefix}_${img_size}_fp32_1batch.bmodel generated."