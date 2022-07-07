#!/bin/bash

model_dir=$(dirname $(readlink -f "$0"))
echo $model_dir
sdk_dir=$REL_TOP

export LD_LIBRARY_PATH=${sdk_dir}/lib/bmcompiler:${sdk_dir}/lib/bmlang:${sdk_dir}/lib/thirdparty/x86:${sdk_dir}/lib/bmnn/cmodel
export PATH=$PATH:${sdk_dir}/bmnet/bmnetc

pushd $model_dir
# modify confidence_threshold to improve inference performance
sed -i "s/confidence_threshold:\ 0.01/confidence_threshold:\ 0.2/g" ../data/models/ssd300_deploy.prototxt

#generate 1batch bmodel
mkdir -p ../data/models/fp32bmodel
bmnetc --model=../data/models/ssd300_deploy.prototxt \
       --weight=../data/models/ssd300.caffemodel \
       --shapes=[1,3,300,300] \
       --outdir=compilation \
       --target=BM1684
cp compilation/compilation.bmodel ../data/models/fp32bmodel/ssd300_fp32_1b.bmodel

generate 4 batch bmodel
bmnetc --model=../data/models/ssd300_deploy.prototxt \
       --weight=../data/models/ssd300.caffemodel \
       --shapes=[4,3,300,300] \
       --outdir=compilation \
       --target=BM1684
cp compilation/compilation.bmodel ../data/models/fp32bmodel/ssd300_fp32_4b.bmodel

#combine bmodel
# bm_model.bin --combine ../data/models/fp32bmodel/ssd300_fp32_1b.bmodel ../data/models/fp32bmodel/ssd300_fp32_4b.bmodel -o ../data/models/fp32bmodel/ssd300_fp32_1b4b.bmodel

rm -rf compilation
popd