#!/bin/bash

source ./model_info.sh

if [ ! -d "$lmdb_src_dir" ]; then
  echo "invalid source images dir $lmdb_src_dir"
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
rm -rf ${lmdb_src_dir}/*.mdb
python3 convert_imageset.py \
        --imageset_rootfolder=${lmdb_src_dir} \
        --imageset_lmdbfolder=${lmdb_src_dir} \
        --resize_height=${img_size} \
        --resize_width=${img_size} \
        --shuffle=True \
        --bgr2rgb=False \
        --gray=False

if [ ! $# -eq 0 ];then
  echo "failed to create LMDB for calibration!"
  exit -1
fi
