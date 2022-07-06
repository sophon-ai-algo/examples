#!/bin/bash
#apt update
#apt install curl

scripts_dir=$(dirname $(readlink -f "$0"))
echo $scripts_dir

pushd $scripts_dir
# test
./download_from_nas.sh http://219.142.246.77:65000/sharing/zPdGaKdL4
tar -xvf lprnet_test.tar -C ../data/images/
rm lprnet_test.tar
# test_md5_lmdb
./download_from_nas.sh http://219.142.246.77:65000/sharing/ui69UstuD
tar -xvf lprnet_test_md5_lmdb.tar -C ../data/images/
rm lprnet_test_md5_lmdb.tar
# test_md5
# ./download_from_nas.sh http://219.142.246.77:65000/sharing/hbeVFlhTU
# tar -xvf lprnet_test_md5.tar -C ../data/images/
# rm lprnet_test_md5.tar
# Final_LPRNet_model.pth
mkdir -p ../data/models
./download_from_nas.sh http://219.142.246.77:65000/sharing/xkBLp6HMm ../data/models/Final_LPRNet_model.pth
# LPRNet_model.torchscript
./download_from_nas.sh http://219.142.246.77:65000/sharing/6GjUqWhii ../data/models/LPRNet_model.torchscript
# lprnet_fp32_1b4b.bmodel
./download_from_nas.sh http://219.142.246.77:65000/sharing/QDabxU6uN ../data/models/lprnet_fp32_1b4b.bmodel
# lprnet_int8_1b4b.bmodel
./download_from_nas.sh http://219.142.246.77:65000/sharing/qQcMS6Orm ../data/models/lprnet_int8_1b4b.bmodel
popd