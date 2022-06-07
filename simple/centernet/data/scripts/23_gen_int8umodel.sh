#!/bin/bash

source model_info.sh


function calibration() 
{
  calibration_use_pb quantize \
      -model="${int8model_dir}/${src_model_file%.*}_bmnetp_test_fp32.prototxt"   \
      -weights="${int8model_dir}/${src_model_file%.*}_bmnetp.fp32umodel"  \
      -iterations=$iteration \
      -winograd=true \
      -save_test_proto=false \
      -bitwidth=TO_INT8 \
      -graph_transform=true
}

pushd $build_dir

calibration

popd
