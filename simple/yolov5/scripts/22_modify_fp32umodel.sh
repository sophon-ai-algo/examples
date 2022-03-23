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

# modify fp32 prototxt file to assign data source
proto_file="${int8model_dir}/${batch_size}/${src_model_name}_bmnetp_test_fp32.prototxt"
tmp_file=`mktemp`
sed -n '1,8p' $proto_file >> $tmp_file
echo "layer {
  name: \"data\"
  type: \"Data\"
  top: \"x.1\"
  include {
    phase: TEST
  }
  transform_param {
      transform_op {
         op: STAND
         mean_value: 0
         mean_value: 0
         mean_value: 0
         scale: 0.00392156862745
         bgr2rgb: true
      }
   }
  data_param {
    source: \"$lmdb_dst_dir\"
    batch_size: $batch_size
    backend: LMDB
  }
}
">> $tmp_file

sed -n '22,$p' $proto_file >> $tmp_file
cat $tmp_file

mv $tmp_file $proto_file

popd