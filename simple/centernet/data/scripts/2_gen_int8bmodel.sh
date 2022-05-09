#!/bin/bash

source model_info.sh

check_file $src_model_path
check_dir  $lmdb_src_dir

./20_create_lmdb.sh
./21_gen_fp32umodel.sh
./22_modify_fp32umodel.sh
./23_gen_int8umodel.sh
./24_gen_int8bmodel.sh