#!/bin/bash

model_dir=$(dirname $(readlink -f "$0"))
echo $model_dir
sdk_dir=$REL_TOP


export LD_LIBRARY_PATH=${sdk_dir}/lib/bmcompiler:${sdk_dir}/lib/bmlang:${sdk_dir}/lib/thirdparty/x86:${sdk_dir}/lib/bmnn/cmodel
export PATH=$PATH:${sdk_dir}/bmnet/bmnetc

# modify confidence_threshold to improve inference performance
sed -i "s/confidence_threshold:\ 0.01/confidence_threshold:\ 0.2/g" ${model_dir}/ssd300_deploy.prototxt

#generate 1batch bmodel
mkdir -p out/ssd300
bmnetc --model=${model_dir}/ssd300_deploy.prototxt \
       --weight=${model_dir}/ssd300.caffemodel \
       --shapes=[1,3,300,300] \
       --outdir=./out/ssd300 \
       --target=BM1684
cp out/ssd300/compilation.bmodel out/ssd300/f32_1b.bmodel

#generate 4 batch bmodel
mkdir -p out/ssd300_4batch
bmnetc --model=${model_dir}/ssd300_deploy.prototxt \
       --weight=${model_dir}/ssd300.caffemodel \
       --shapes=[4,3,300,300] \
       --outdir=./out/ssd300_4batch \
       --target=BM1684
cp out/ssd300_4batch/compilation.bmodel out/ssd300_4batch/f32_4b.bmodel

#combine bmodel
bm_model.bin --combine out/ssd300/f32_1b.bmodel out/ssd300_4batch/f32_4b.bmodel -o out/fp32_ssd300.bmodel
