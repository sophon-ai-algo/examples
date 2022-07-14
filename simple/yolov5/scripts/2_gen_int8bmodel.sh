#!/bin/bash

source model_info.sh

check_file $src_model_file
check_dir $lmdb_src_dir

auto_cali_dir=$build_dir/auto_cali

if [ ! -d "${auto_cali_dir}" ]; then
    echo "create data dir: ${auto_cali_dir}"
    mkdir -p ${auto_cali_dir}
fi

pushd ${auto_cali_dir}

#!/bin/bash
python3 -m ufw.cali.cali_model \
 --net_name $dst_model_prefix \
 --model ${src_model_file} \
 --cali_image_path ${image_src_dir} \
 --cali_image_preprocess 'resize_h=640,resize_w=640;scale=0.003921569,bgr2rgb=True' \
 --input_shapes "[${batch_size},3,${img_height},${img_width}]" \
 --cali_iterations=1 \

dst_model_dir=${root_dir}/data/models
if [ ! -d "$dst_model_dir" ]; then
    echo "create data dir: $dst_model_dir"
    mkdir -p $dst_model_dir
fi

cp "${build_dir}/${dst_model_prefix}_batch${batch_size}/compilation.bmodel" "$dst_model_dir/${dst_model_prefix}_${img_size}_${dst_model_postfix}_int8_${batch_size}b.bmodel"
echo "[Success] $dst_model_dir/${dst_model_prefix}_${img_size}_${dst_model_postfix}_int8_${batch_size}b.bmodel is generated."


popd
