#!/bin/bash
root_dir=$(cd `dirname $BASH_SOURCE[0]`/../ && pwd)
model_dir=$root_dir/data/models

function gen_fp32bmodel()
{
	python3 -m bmnetp --net_name=yolact_base \
			  --target=BM1684 \
			  --opt=1 \
			  --cmp=true \
			  --shapes="[1,3,550,550]" \
			  --model=yolact_base_54_800000.trace.pt \
			  --outdir=./yolact_base_54_800000_fp32_b1 \
			  --dyn=false
	cp yolact_base_54_800000_fp32_b1/compilation.bmodel yolact_base_54_800000_fp32_b1/yolact_base_54_800000_fp32_b1.bmodel
}

pushd $model_dir
gen_fp32bmodel
popd
