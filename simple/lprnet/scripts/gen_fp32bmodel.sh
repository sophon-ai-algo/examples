#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
outdir=../data/models/fp32bmodel

function gen_fp32bmodel()
{
    python3 -m bmnetp --net_name=lprnet \
                      --target=BM1684 \
                      --opt=1 \
                      --cmp=true \
                      --shapes=[$1,3,24,94] \
                      --model=../data/models/LPRNet_model.torchscript \
                      --outdir=$outdir \
                      --dyn=false
    cp $outdir/compilation.bmodel $outdir/lprnet_fp32_$1b.bmodel

}

pushd $model_dir
#batch_size=1
gen_fp32bmodel 1
#batch_size=4
gen_fp32bmodel 4
bm_model.bin --combine $outdir/lprnet_fp32_1b.bmodel $outdir/lprnet_fp32_4b.bmodel -o $outdir/lprnet_fp32_1b4b.bmodel 
popd