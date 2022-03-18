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
# iteration=${3:-2}

sdk_dir=$REL_TOP

echo "sdk dir: "$sdk_dir

export LD_LIBRARY_PATH=${sdk_dir}/lib/bmcompiler:${sdk_dir}/lib/bmlang:${sdk_dir}/lib/thirdparty/x86:${sdk_dir}/lib/bmnn/cmodel:${sdk_dir}/lib/calibration-tools
export PATH=$PATH:${sdk_dir}/bmnet/bmnetd:${sdk_dir}/bin/x86/calibration-tools

function calibration() 
{
  calibration_use_pb quantize \
      -model="${int8model_dir}/${batch_size}/${src_model_name}_bmnetp_test_fp32.prototxt"   \
      -weights="${int8model_dir}/${batch_size}/${src_model_name}_bmnetp.fp32umodel"  \
      -iterations=$iteration \
      -winograd=false \
      -save_test_proto=false \
      -save_test_proto=false \
      -bitwidth=TO_INT8
}

pushd $build_dir

calibration

popd
