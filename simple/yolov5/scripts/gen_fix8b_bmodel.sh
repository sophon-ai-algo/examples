#!/bin/bash

root_dir=$(cd `dirname $BASH_SOURCE[0]`/../ && pwd)
mkdir -p $root_dir/build
pushd $root_dir/build

dataset_path=$1
if [ -z "$dataset_path" ]; then
  echo "Usage: gen_fix8b_bmodel.sh dataset_path [iteration] [img_size] [batch_size]"
  exit -1
fi
iteration=${2:-1}
img_size=${3:-640}
batch_size=${4:-1}

model_name=$root_dir/data/models/yolov5s.torchscript.$img_size.$batch_size.pt
in_name=yolov5s_fp32_${img_size}_${batch_size}

# generate binary fp32 umodel
#echo "--------------------------------------------------"
python3 -m bmnetp --mode=GenUmodel --model $model_name --shapes [$batch_size,3,$img_size,$img_size] --target BM1684 --outdir $in_name --v 4  2>&1 | tee $in_name.log

# generate text fp32 prototxt
python3 -m ufw.tools.to_umodel -u $in_name/bm_network_bmnetp.fp32umodel -T

# modify fp32 prototxt file to assign data source
proto_file=$in_name/bm_network_bmnetp_test_fp32.prototxt
tmp_file=`mktemp`
sed -n '1,3p' $proto_file >>$tmp_file
echo "layer {
  name: \"data\"
  type: \"Data\"
  top: \"x.1\"
  include {
    phase: TEST
  }
  data_param {
    source: \"$dataset_path\"
    batch_size: $batch_size 
    backend: LMDB
  }
}
">> $tmp_file

sed -n '21,$p' $proto_file >> $tmp_file
cat $tmp_file

mv $tmp_file $prototxt


# generate lmdb datasets
rm -rf img_lmdb
ls $dataset_path/*.jpg > $dataset_path/imageset.txt
convert_imageset --shuffle --resize_height=$img_size --resize_width=$img_size $dataset_path/ $dataset_path/imageset.txt img_lmdb

#
# calibration fp32 umodel
calibration_use_pb release -model $in_name/bm_network_bmnetp_test_fp32.prototxt -weights $in_name/bm_network_bmnetp.fp32umodel -iterations=$iteration -bitwidth=TO_INT8 2>&1 | tee cali_$in_name.log

#
# generate fix8b bmodel
out_name=yolov5s_fix8b_${img_size}_${batch_size}
bmnetu -model $in_name/bm_network_bmnetp_deploy_int8_unique_top.prototxt -weight $in_name/bm_network_bmnetp.int8umodel --cmp=false --outdir $out_name --v 4 2>&1 | tee umodel_$in_name.log
mv $out_name/compilation.bmodel $root_dir/data/models/$out_name.bmodel

popd
