#!/bin/bash

source model_info.sh


function gen_int8bmodel()
{
  #1batch bmodel
  bmnetu -model="${int8model_dir}/${src_model_file%.*}_bmnetp_deploy_int8_unique_top.prototxt" \
       -weight="${int8model_dir}/${src_model_file%.*}_bmnetp.int8umodel" \
       -max_n=4 \
       -prec=INT8 \
       -dyn=0 \
       -cmp=1 \
       --v 4 \
       -target=BM1684 \
       -outdir="${int8model_dir}"
  cp ${int8model_dir}/compilation.bmodel ${root_dir}/models/${dst_model_prefix}_${img_size}_int8_4batch.bmodel

}

pushd $build_dir

gen_int8bmodel
echo "[Success] ${root_dir}/models/${dst_model_prefix}_${img_size}_int8_4batch.bmodel done."
popd