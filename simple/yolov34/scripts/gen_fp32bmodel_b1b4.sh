#!/bin/bash
script_dir=$(dirname $(readlink -f "$0"))
model_dir=$script_dir/../data/models
echo $model_dir
src_model_name="yolov4"
dst_model_name="yolov4_416_coco"

#generate 1batch bmodel
mkdir -p out/${dst_model_name}
bmnetd --model=${model_dir}/${src_model_name}.cfg \
       --weight=${model_dir}/${src_model_name}.weights \
       --shapes=[1,3,416,416] \
       --outdir=./out/${dst_model_name} \
       --target=BM1684
cp out/${dst_model_name}/compilation.bmodel out/${dst_model_name}/fp32_1b.bmodel

#generate 4 batch bmodel
mkdir -p out/${dst_model_name}_4batch
bmnetd --model=${model_dir}/${src_model_name}.cfg \
       --weight=${model_dir}/${src_model_name}.weights \
       --shapes=[4,3,416,416] \
       --outdir=./out/${dst_model_name}_4batch \
       --target=BM1684 \
       --v=4
cp out/${dst_model_name}_4batch/compilation.bmodel out/${dst_model_name}_4batch/fp32_4b.bmodel

# combine bmodel
bm_model.bin --combine out/${dst_model_name}/fp32_1b.bmodel out/${dst_model_name}_4batch/fp32_4b.bmodel -o out/${dst_model_name}_fp32_1b_4b.bmodel
