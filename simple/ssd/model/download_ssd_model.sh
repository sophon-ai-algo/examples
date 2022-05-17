#!/bin/bash

target_dir=$(dirname $(readlink -f "$0"))

pushd $target_dir
# models_VGGNet_VOC0712_SSD_300x300.tar.gz
./download_from_nas.sh http://219.142.246.77:65000/sharing/LYE9kdKTA
tar -xvzf models_VGGNet_VOC0712_SSD_300x300.tar.gz
sync
ln -sf ./models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel ssd300.caffemodel
ln -sf ./models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt ssd300_deploy.prototxt
# data.mdb
mkdir -p data/VOC0712/
./download_from_nas.sh http://219.142.246.77:65000/sharing/PgMtlQcxe
mv data.mdb data/VOC0712/

echo "All done!"
popd
