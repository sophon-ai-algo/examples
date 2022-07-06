#!/bin/bash

source model_info.sh

pushd $build_dir

check_file $src_model_file

python3 -m bmnetp --mode="compile" \
      --model="${src_model_file}" \
      --outdir="${fp32model_dir}/${batch_size}" \
      --target="BM1684" \
      --shapes=[[${batch_size},3,${img_height},${img_width}]] \
      --net_name=$dst_model_prefix \
      --opt=2 \
      --dyn=False \
      --cmp=True \
      --enable_profile=True 

dst_model_dir=${root_dir}/data/models
if [ ! -d "$dst_model_dir" ]; then
    echo "create data dir: $dst_model_dir"
    mkdir -p $dst_model_dir
fi
cp "${fp32model_dir}/${batch_size}/compilation.bmodel" "${dst_model_dir}/${dst_model_prefix}_${img_size}_${dst_model_postfix}_fp32_${batch_size}b.bmodel"

popd
