#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
echo $scripts_dir

pushd $scripts_dir
# images
./download_from_nas.sh http://219.142.246.77:65000/sharing/zMHZeauPP
tar -xvf ppocr_img.tar -C ../data/images/
rm ppocr_img.tar

# paddle model
mkdir ../data/models
# ch_PP-OCRv2_det_infer
./download_from_nas.sh http://219.142.246.77:65000/sharing/sny6uBdN9
tar -xvf ch_PP-OCRv2_det_infer.tar -C ../data/models
rm ch_PP-OCRv2_det_infer.tar
# ch_PP-OCRv2_rec_infer 
./download_from_nas.sh http://219.142.246.77:65000/sharing/fbIAcI119
tar -xvf ch_PP-OCRv2_rec_infer.tar -C ../data/models
rm ch_PP-OCRv2_rec_infer.tar 
# ch_ppocr_mobile_v2.0_cls_infer
./download_from_nas.sh http://219.142.246.77:65000/sharing/IxA5hKuLf
tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar -C ../data/models
rm ch_ppocr_mobile_v2.0_cls_infer.tar
popd