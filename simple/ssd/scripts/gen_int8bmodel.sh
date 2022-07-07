#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
echo $model_dir

function gen_fp32umodel()
{
python3 -m ufw.tools.cf_to_umodel \
    -m ../data/models/ssd300_deploy.prototxt \
    -w ../data/models/ssd300.caffemodel \
    -s '(1, 3, 300, 300)' \
    -d compilation \
    -D ../data/images/lmdb/ \
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
  mkdir -p ../data/models/int8bmodel
  bmnetu -model compilation/ssd300_bmnetc_deploy_int8_unique_top.prototxt \
       -weight compilation/ssd300_bmnetc.int8umodel \
       -max_n 1 \
       -prec=INT8 \
       -dyn=0 \
       -cmp=1 \
       -target=BM1684 \
       -outdir=compilation
  if [ $? -ne 0 ]; then
      echo "bmnetu error for batch 1"
      exit 1
  else
      echo "bmnetu ok for batch 1"
  fi

  cp compilation/compilation.bmodel ../data/models/int8bmodel/ssd300_int8_1b.bmodel

#4batch bmodel
  bmnetu -model compilation/ssd300_bmnetc_deploy_int8_unique_top.prototxt \
       -weight compilation/ssd300_bmnetc.int8umodel \
       -max_n 4 \
       -prec=INT8 \
       -dyn=0 \
       -cmp=1 \
       -target=BM1684 \
       -outdir=compilation
  
  if [ $? -ne 0 ]; then
      echo "bmnetu error for batch 4"
      exit 1
  else
      echo "bmnetu ok for batch 4"
  fi

  cp compilation/compilation.bmodel ../data/models/int8bmodel/ssd300_int8_4b.bmodel

#combine bmodel
#   bm_model.bin --combine ../data/models/int8bmodel/ssd300_int8_1b.bmodel ../data/models/int8bmodel/ssd300_int8_4b.bmodel -o ../data/models/int8bmodel/ssd300_int8_1b4b.bmodel
#   if [ $? -ne 0 ]; then
#       echo "combine bmodel error"
#       exit 1
#   else
#       echo "combine bmodel ok"
#   fi
}

pushd $model_dir
gen_fp32umodel
gen_int8umodel
gen_int8bmodel
rm -rf compilation
popd
