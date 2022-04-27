#!/bin/bash

source model_info.sh



pushd $build_dir

# modify fp32 prototxt file to assign data source
proto_file="${int8model_dir}/${src_model_file%.*}_bmnetp_test_fp32.prototxt"
has_transform=`grep transform_param $proto_file`
if [[ ! -n $has_transform ]]
then
     patch $proto_file -i data_layer.patch
fi
echo "$proto_file patched"
popd