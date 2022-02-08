#!/bin/bash

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
    --outdir=$MODEL_DIR/fp32_centernet_bmodel \
    --dyn=false

cp $MODEL_DIR/fp32_centernet_bmodel/compilation.bmodel $MODEL_DIR/ctdet_coco_dlav0_1x_fp32.bmodel

echo "[Success] $MODEL_DIR/ctdet_coco_dlav0_1x_fp32.bmodel generated."