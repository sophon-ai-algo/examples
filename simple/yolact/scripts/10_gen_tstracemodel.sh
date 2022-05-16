!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))
echo $model_dir
 
function gen_tstracemodel()
{
	cd converter
	python3 convert.py --input ../../data/models/yolact_base_54_800000.pth \
			   --mode tstrace \
			   --cfg yolact_base \
			   --output ../../data/models/
}
 
pushd $model_dir
gen_tstracemodel
popd
