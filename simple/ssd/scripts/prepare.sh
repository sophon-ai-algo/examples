#!/bin/bash

target_dir=$(dirname $(readlink -f "$0"))

pushd $target_dir
mkdir -p ../data/models
# models_VGGNet_VOC0712_SSD_300x300.tar.gz
./download_from_nas.sh http://219.142.246.77:65000/sharing/LYE9kdKTA
tar -xvzf models_VGGNet_VOC0712_SSD_300x300.tar.gz
mv ./models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel ../data/models/ssd300.caffemodel
mv ./models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt ../data/models/ssd300_deploy.prototxt
rm -rf models_VGGNet_VOC0712_SSD_300x300.tar.gz ./models
# data.mdb
mkdir -p ../data/images/lmdb
./download_from_nas.sh http://219.142.246.77:65000/sharing/PgMtlQcxe
mv data.mdb ../data/images/lmdb
# test_car_person.mp4
mkdir -p ../data/videos
./download_from_nas.sh http://219.142.246.77:65000/sharing/hjUKjOoyv
mv test_car_person.mp4 ../data/videos
echo "All done!"
popd
