#!/bin/bash

source model_info.sh

# src_model_file="yolov5s_coco_v6.1_3output.torchscript"
# src_model_name=`basename ${src_model_file}`
# dst_model_prefix="yolov5s_coco_v6.1_3output"
# fp32model_dir="fp32model"
# int8model_dir="int8model"
# lmdb_dir="./images/test/img_lmdb"
# img_size=${1:-640}
# batch_size=${2:-1}

export BMNETP_LOG_LEVEL=3

pushd $build_dir

python3 -m ufw.tools.pt_to_umodel \
    -m="${src_model_file}" \
    -s="(${batch_size},3,${img_height},${img_width})" \
    -d="${int8model_dir}/${batch_size}" \
    -D="${lmdb_dst_dir}" \
    --cmp
    
popd
