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

pushd $build_dir

check_file $src_model_file

python3 -m bmnetp --mode="compile" \
      --model="${src_model_file}" \
      --outdir="${fp32model_dir}/${batch_size}" \
      --target="BM1684" \
      --shapes=[[${batch_size},3,${img_height},${img_width}]] \
      --net_name=${src_model_name} \
      --opt=2 \
      --dyn=False \
      --cmp=True \
      --enable_profile=True 
 
cp "${fp32model_dir}/${batch_size}/compilation.bmodel" "${fp32model_dir}/${dst_model_prefix}_fp32_${batch_size}b.bmodel"

popd
