#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
echo $model_dir

function gen_fp32bmodel()
{
    python3 -m bmnetp --net_name=lprnet \
                      --target=BM1684 \
                      --opt=1 \
                      --cmp=true \
                      --shapes="[1,3,24,94]" \
                      --model=../data/models/LPRNet_model.torchscript \
                      --outdir=./fp32bmodel \
                      --dyn=false
    cp fp32bmodel/compilation.bmodel fp32bmodel/lprnet_fp32_1b.bmodel

}

pushd $model_dir
gen_fp32bmodel
popd