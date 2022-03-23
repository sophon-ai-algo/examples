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

sdk_dir=$REL_TOP

echo "sdk dir: "$sdk_dir

export LD_LIBRARY_PATH=${sdk_dir}/lib/bmcompiler:${sdk_dir}/lib/bmlang:${sdk_dir}/lib/thirdparty/x86:${sdk_dir}/lib/bmnn/cmodel:${sdk_dir}/lib/calibration-tools
export PATH=$PATH:${sdk_dir}/bmnet/bmnetd:${sdk_dir}/bin/x86/calibration-tools

function gen_int8bmodel()
{
  #1batch bmodel
  bmnetu -model="${int8model_dir}/${batch_size}/${src_model_name}_bmnetp_deploy_int8_unique_top.prototxt" \
       -weight="${int8model_dir}/${batch_size}/${src_model_name}_bmnetp.int8umodel" \
       -max_n=1 \
       -prec=INT8 \
       -dyn=0 \
       -cmp=1 \
       -target=BM1684 \
       -outdir="${int8model_dir}/${batch_size}"

  cp ${int8model_dir}/${batch_size}/compilation.bmodel ${int8model_dir}/${dst_model_prefix}_int8_${batch_size}b.bmodel

}

pushd $build_dir

gen_int8bmodel

popd