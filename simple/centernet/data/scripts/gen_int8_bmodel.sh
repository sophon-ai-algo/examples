#!/bin/bash

if [[ $# -ne 3 ]]
then
     echo "usage: $0 <batch_size> <img_size> <validation_image_dir>"
     echo "eg. gen_int8_bmodel.sh 4 512 ../val2017"
     exit 0
fi

SCRIPT_DIR=`pwd`/`dirname $0`
MODEL_DIR=$SCRIPT_DIR/../models

batch_size=${1:-1}
img_size=${2:-512}
model_name=$MODEL_DIR/ctdet_coco_dlav0_1x.torchscript.pt
dataset_path=${3:-"../val2017"}

if [[ ! -f $model_name ]]
then
     echo "model file:$model_name not exist."
     exit 0
fi

if [[ ! -d $dataset_path ]]
then
     echo "$dataset_path directory:$dataset_path not exist."
     exit 0
fi

# rm -rf img_lmdb
# ls $dataset_path/*.jpg | head -n 200 | cut -d '/' -f3 > $dataset_path/imageset.txt
# convert_imageset --shuffle --resize_height=$img_size --resize_width=$img_size $dataset_path/ $dataset_path/imageset.txt img_lmdb

# rm -rf compilation
# python3 $SCRIPT_DIR/centernet_pt_to_fp32umodel.py $model_name $SCRIPT_DIR/img_lmdb $batch_size $img_size

# has_transform=`grep transform_param compilation/ctdet_coco_dlav0_1x.torchscript_bmnetp_test_fp32.prototxt`
# if [[ ! -n $has_transform ]]
# then
#      patch compilation/ctdet_coco_dlav0_1x.torchscript_bmnetp_test_fp32.prototxt -i data_layer.patch
# fi

# calibration_use_pb \
#     quantize \
#     -model=$SCRIPT_DIR/compilation/ctdet_coco_dlav0_1x.torchscript_bmnetp_test_fp32.prototxt \
#     -weights=$SCRIPT_DIR/compilation/ctdet_coco_dlav0_1x.torchscript_bmnetp.fp32umodel \
#     -iterations=100 \
#     -bitwidth=TO_INT8

bmnetu -model $SCRIPT_DIR/compilation/ctdet_coco_dlav0_1x.torchscript_bmnetp_deploy_int8_unique_top.prototxt \
     -weight $SCRIPT_DIR/compilation/ctdet_coco_dlav0_1x.torchscript_bmnetp.int8umodel \
     -max_n $batch_size \
     -prec=INT8 \
     -dyn=0 \
     -cmp=1 \
     -target=BM1684 \
     --v 4 \
     -outdir=$MODEL_DIR/int8_centernet_bmodel

# rm -rf $SCRIPT_DIR/compilation
cp $MODEL_DIR/int8_centernet_bmodel/compilation.bmodel $MODEL_DIR/ctdet_coco_dlav0_1x_int8_b$batch_size.bmodel

echo "[Success] $MODEL_DIR/ctdet_coco_dlav0_1x_int8_b$batch_size.bmodel generated."
