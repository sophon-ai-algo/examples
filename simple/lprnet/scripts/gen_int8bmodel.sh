#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
echo $model_dir

function create_lmdb()
{
    rm ../data/images/test_md5_lmdb/*
    python3 ../tools/convert_imageset.py \
        --imageset_rootfolder=../data/images/test_md5 \
        --imageset_lmdbfolder=../data/images/test_md5_lmdb/ \
        --resize_height=24 \
        --resize_width=94 \
        --shuffle=True \
        --bgr2rgb=True \
        --gray=False
}
function gen_fp32umodel()
{
    python3 -m ufw.tools.pt_to_umodel \
        -m ../data/models/LPRNet_model.torchscript \
        -s '(1, 3, 24, 94)' \
        -d compilation \
        -D ../data/images/test_md5_lmdb/ \
        --cmp
}
function gen_int8umodel()
{
    calibration_use_pb quantize \
        -model=compilation/LPRNet_model.torchscript_bmnetp_test_fp32.prototxt \
        -weights=compilation/LPRNet_model.torchscript_bmnetp.fp32umodel \
        -iterations=200 \
        -save_test_proto=true \
        -th_method=JSD \
        -fpfwd_blocks="< 4 0 >19,< 8 0 >27"
}
function gen_int8bmodel()
{
    bmnetu -model=compilation/LPRNet_model.torchscript_bmnetp_deploy_int8_unique_top.prototxt \
           -weight=compilation/LPRNet_model.torchscript_bmnetp.int8umodel \
           -outdir=int8bmodel/
    cp int8bmodel/compilation.bmodel int8bmodel/lprnet_int8_1b.bmodel
}

pushd $model_dir
create_lmdb
gen_fp32umodel
gen_int8umodel
gen_int8bmodel
popd