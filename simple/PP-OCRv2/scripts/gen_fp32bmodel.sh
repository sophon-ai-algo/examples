#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
echo $scripts_dir

pushd $scripts_dir

model_det_dir=../data/models/ch_PP-OCRv2_det_infer
model_cls_dir=../data/models/ch_ppocr_mobile_v2.0_cls_infer
model_rec_dir=../data/models/ch_PP-OCRv2_rec_infer

out_dir=../data/models/fp32bmodel 
mkdir -p ${out_dir}
#rm -rf fp32bmodel/*

#PP-OCRv2_det
#generate 1batch bmodel
python3 -m bmpaddle --model=${model_det_dir} --outdir=${out_dir} --shapes="[1, 3, 960, 960]" --net_name="PP-OCRv2_det"  --target=BM1684   --dyn=false --opt=2 --cmp=true
mv ${out_dir}/compilation.bmodel ${out_dir}/ch_PP-OCRv2_det_1b.bmodel
#generate 4batch bmodel
python3 -m bmpaddle --model=${model_det_dir} --outdir=${out_dir} --shapes="[4, 3, 960, 960]" --net_name="PP-OCRv2_det"  --target=BM1684   --dyn=false --opt=2 --cmp=true
mv ${out_dir}/compilation.bmodel ${out_dir}/ch_PP-OCRv2_det_4b.bmodel
bm_model.bin --combine ${out_dir}/ch_PP-OCRv2_det_1b.bmodel ${out_dir}/ch_PP-OCRv2_det_4b.bmodel -o ${out_dir}/ch_PP-OCRv2_det_fp32_b1b4.bmodel

#ppocr_mobile_v2.0_cls
#generate 1batch bmodel
python3 -m bmpaddle --model=${model_cls_dir} --outdir=${out_dir} --shapes="[1, 3, 48, 192]" --net_name="ppocr_mobile_v2.0_cls"  --target=BM1684   --dyn=false --opt=1 --cmp=true
mv ${out_dir}/compilation.bmodel ${out_dir}/ch_ppocr_mobile_v2.0_cls_1b.bmodel
#generate 4batch bmodel
python3 -m bmpaddle --model=${model_cls_dir} --outdir=${out_dir} --shapes="[4, 3, 48, 192]" --net_name="ppocr_mobile_v2.0_cls"  --target=BM1684   --dyn=false --opt=1 --cmp=true
mv ${out_dir}/compilation.bmodel ${out_dir}/ch_ppocr_mobile_v2.0_cls_4b.bmodel
bm_model.bin --combine ${out_dir}/ch_ppocr_mobile_v2.0_cls_1b.bmodel ${out_dir}/ch_ppocr_mobile_v2.0_cls_4b.bmodel -o ${out_dir}/ch_ppocr_mobile_v2.0_cls_fp32_b1b4.bmodel

#PP-OCRv2_rec
#generate [1, 3, 32, 320] bmodel
python3 -m bmpaddle --model=${model_rec_dir} --outdir=${out_dir} --shapes="[1, 3, 32, 320]" --net_name="PP-OCRv2_rec"  --target=BM1684   --dyn=false --opt=1 --cmp=true
mv ${out_dir}/compilation.bmodel ${out_dir}/ch_PP-OCRv2_rec_320_1b.bmodel
#generate [4, 3, 32, 320] bmodel
python3 -m bmpaddle --model=${model_rec_dir} --outdir=${out_dir} --shapes="[4, 3, 32, 320]" --net_name="PP-OCRv2_rec"  --target=BM1684   --dyn=false --opt=1 --cmp=true
mv ${out_dir}/compilation.bmodel ${out_dir}/ch_PP-OCRv2_rec_320_4b.bmodel
#generate [1, 3, 32, 640] bmodel
python3 -m bmpaddle --model=${model_rec_dir} --outdir=${out_dir} --shapes="[1, 3, 32, 640]" --net_name="PP-OCRv2_rec"  --target=BM1684   --dyn=false --opt=1 --cmp=true
mv ${out_dir}/compilation.bmodel ${out_dir}/ch_PP-OCRv2_rec_640_1b.bmodel
#generate [4, 3, 32, 640] bmodel
python3 -m bmpaddle --model=${model_rec_dir} --outdir=${out_dir} --shapes="[4, 3, 32, 640]" --net_name="PP-OCRv2_rec"  --target=BM1684   --dyn=false --opt=1 --cmp=true
mv ${out_dir}/compilation.bmodel ${out_dir}/ch_PP-OCRv2_rec_640_4b.bmodel
#generate [1, 3, 32, 1280] bmodel
python3 -m bmpaddle --model=${model_rec_dir} --outdir=${out_dir} --shapes="[1, 3, 32, 1280]" --net_name="PP-OCRv2_rec"  --target=BM1684   --dyn=false --opt=1 --cmp=true
mv ${out_dir}/compilation.bmodel ${out_dir}/ch_PP-OCRv2_rec_1280_1b.bmodel

bm_model.bin --combine ${out_dir}/ch_PP-OCRv2_rec_320_1b.bmodel ${out_dir}/ch_PP-OCRv2_rec_320_4b.bmodel ${out_dir}/ch_PP-OCRv2_rec_640_1b.bmodel ${out_dir}/ch_PP-OCRv2_rec_640_4b.bmodel ${out_dir}/ch_PP-OCRv2_rec_1280_1b.bmodel -o ${out_dir}/ch_PP-OCRv2_rec_fp32_b1b4.bmodel

popd