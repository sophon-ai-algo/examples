#!/bin/bash

root_dir=$(cd `dirname $BASH_SOURCE[0]`/../ && pwd)
mkdir -p $root_dir/build
pushd $root_dir/build

img_size=${1:-640}
batch_size=${2:-1}

model_name=$root_dir/data/models/yolov5s.torchscript.$img_size.$batch_size.pt
outname=yolov5s_fp32_${img_size}_${batch_size}

python3 -m bmnetp --model $model_name --shapes [$batch_size,3,$img_size,$img_size] --target BM1684 --outdir $outname --v 4  --enable_profile True 2>&1 | tee $outname.log

mv $outname/compilation.bmodel $root_dir/data/models/$outname.bmodel

popd
