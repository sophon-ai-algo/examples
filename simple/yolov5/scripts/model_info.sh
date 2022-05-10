#!/bin/bash
root_dir=$(cd `dirname $BASH_SOURCE[0]`/../ && pwd)
build_dir=$root_dir/build
src_model_file=$build_dir/"yolov5s_coco_v6.1_3output.torchscript"
src_model_name=`basename ${src_model_file}`
dst_model_prefix="yolov5s_coco_v6.1_3output"
fp32model_dir="fp32model"
int8model_dir="int8model"
lmdb_src_dir="${build_dir}/coco/images/val2017/"
# lmdb_src_dir="${build_dir}/coco2017val/coco/images/"
lmdb_dst_dir="${build_dir}/lmdb/"
img_size=${1:-640}
batch_size=${2:-1}
iteration=${3:-2}
img_width=640
img_height=640

function check_file()
{
    if [ ! -f $1 ]; then
        echo "$1 not exist."
        exit 1
    fi
}

function check_dir()
{
    if [ ! -d $1 ]; then
        echo "$1 not exist."
        exit 1
    fi
}
