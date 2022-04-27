#!/bin/bash

source model_info.sh


export BMNETP_LOG_LEVEL=3

pushd $build_dir

python3 -m ufw.tools.pt_to_umodel \
    -m="${src_model_path}" \
    -s="(1,3,${img_size},${img_size})" \
    -d="${int8model_dir}" \
    -D="${lmdb_dst_dir}" \
    --cmp

popd
