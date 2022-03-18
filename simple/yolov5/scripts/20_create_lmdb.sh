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

root_dir=$(cd `dirname $BASH_SOURCE[0]`/../ && pwd)
build_dir=$root_dir/build
if [ ! -d "$lmdb_src_dir" ]; then
  echo "invalid source images dir"
  exit 1
fi
if [ ! -d "$lmdb_dst_dir" ]; then
  mkdir -p $lmdb_dst_dir
fi
if [ -f "${lmdb_dst_dir}data.mdb" ]; then
  echo "${lmdb_dst_dir}data.mdb exist."
  exit 1
fi

# Generate lmdb

# usage: convert_imageset.py
# optional arguments:
#   -h, --help            show this help message and exit
#   --imageset_rootfolder IMAGESET_ROOTFOLDER
#                         please setting images source path
#   --imageset_lmdbfolder IMAGESET_LMDBFOLDER
#                         please setting lmdb path
#   --shuffle SHUFFLE     shuffle order of images
#   --resize_height RESIZE_HEIGHT
#                         target height
#   --resize_width RESIZE_WIDTH
#                         target width
#   --bgr2rgb BGR2RGB     convert bgr to rgb
#   --gray GRAY           if True, read image as gray

python3 ../tools/convert_imageset.py \
        --imageset_rootfolder=$lmdb_src_dir \
        --imageset_lmdbfolder=$lmdb_dst_dir \
        --shuffle=True \
        --resize_height=$img_height \
        --resize_width=$img_width \
        --bgr2rgb=False \
        --gray=False

if [ ! $# -eq 0 ];then
  echo "failed to create LMDB for calibration!"
  exit -1
fi
