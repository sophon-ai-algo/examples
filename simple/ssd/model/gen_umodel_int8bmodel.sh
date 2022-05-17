#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
echo $model_dir

function gen_fp32umodel()
{
python3 -m ufw.tools.cf_to_umodel \
    -m ssd300_deploy.prototxt \
    -w ssd300.caffemodel \
    -s '(1, 3, 300, 300)' \
    -d compilation \
    -D data/VOC0712/ \
    --cmp
}
function gen_int8umodel()
{

calibration_use_pb quantize \
    -model=compilation/ssd300_bmnetc_test_fp32.prototxt \
    -weights=compilation/ssd300_bmnetc.fp32umodel \
    -iterations=200 \
    -bitwidth=TO_INT8
}

function gen_int8bmodel()
{
#1batch bmodel
  mkdir int8model
  bmnetu -model compilation/ssd300_bmnetc_deploy_int8_unique_top.prototxt \
       -weight compilation/ssd300_bmnetc.int8umodel \
       -max_n 1 \
       -prec=INT8 \
       -dyn=0 \
       -cmp=1 \
       -target=BM1684 \
       -outdir=./int8model
  if [ $? -ne 0 ]; then
      echo "bmnetu error for batch 1"
      exit 1
  else
      echo "bmnetu ok for batch 1"
  fi

  cp int8model/compilation.bmodel int8model/compilation_1.bmodel

#4batch bmodel
  bmnetu -model compilation/ssd300_bmnetc_deploy_int8_unique_top.prototxt \
       -weight compilation/ssd300_bmnetc.int8umodel \
       -max_n 4 \
       -prec=INT8 \
       -dyn=0 \
       -cmp=1 \
       -target=BM1684 \
       -outdir=./int8model
  
  if [ $? -ne 0 ]; then
      echo "bmnetu error for batch 4"
      exit 1
  else
      echo "bmnetu ok for batch 4"
  fi

  cp int8model/compilation.bmodel int8model/compilation_4.bmodel

#combine bmodel
  mkdir -p ./out
  bm_model.bin --combine int8model/compilation_4.bmodel int8model/compilation_1.bmodel -o out/int8_ssd300.bmodel
  if [ $? -ne 0 ]; then
      echo "combine bmodel error"
      exit 1
  else
      echo "combine bmodel ok"
  fi
}

pushd $model_dir
gen_fp32umodel
gen_int8umodel
gen_int8bmodel
popd
