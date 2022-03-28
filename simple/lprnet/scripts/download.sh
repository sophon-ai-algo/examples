#!/bin/bash
apt update
apt install curl

scripts_dir=$(dirname $(readlink -f "$0"))
echo $scripts_dir

pushd $scripts_dir
# test
./download_from_nas.sh https://disk.sophgo.vip/sharing/4EVs06DoX
tar -xvf test.tar -C ../data/images/
rm test.tar
# test_md5
./download_from_nas.sh https://disk.sophgo.vip/sharing/l9OfrBf6y
tar -xvf test_md5.tar -C ../data/images/
rm test_md5.tar
# Final_LPRNet_model.pth
./download_from_nas.sh https://disk.sophgo.vip/sharing/2GPOnD1pj
mv Final_LPRNet_model.pth ../data/models
# LPRNet_model.torchscript
./download_from_nas.sh https://disk.sophgo.vip/sharing/xe2BiHGkV
mv LPRNet_model.torchscript ../data/models
# lprnet_fp32_1b.bmodel
./download_from_nas.sh https://disk.sophgo.vip/sharing/h4NxSzjHy
mv lprnet_fp32_1b.bmodel ../data/models
# lprnet_int8_1b.bmodel
./download_from_nas.sh https://disk.sophgo.vip/sharing/AIRJCf2BS
mv lprnet_int8_1b.bmodel ../data/models
popd