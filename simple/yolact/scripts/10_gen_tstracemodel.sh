#!/bin/bash
root_dir=$(cd `dirname $BASH_SOURCE[0]`/../ && pwd)
cvt_dir=$root_dir/scripts/converter
 
function gen_tstracemodel()
{
	python3 convert.py --input $root_dir/data/models/yolact_base_54_800000.pth \
			   --mode tstrace \
			   --cfg yolact_base \
			   --output $root_dir/data/models/
}
 
pushd $cvt_dir
gen_tstracemodel
popd
