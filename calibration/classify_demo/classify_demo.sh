#!/bin/bash
 
 
 
INFE_NUM=10
CALI_NUM=10
 
 
# You need database for calibration
function convert_to_int8_demo()
{
 
 
  calibration_use_pb quantize \
    -model=./models/resnet18.prototxt   \
    -weights=./models/resnet18.fp32umodel  \
    -iterations=${CALI_NUM} \
    -bitwidth=TO_INT8 \
    -save_test_proto=true
 
  ret=$?; if [ $ret -ne 0 ]; then echo "#ERROR: Resnet18 Fp32ToInt8"; popd; return $ret;fi
 
 
  echo "#INFO: Run Example (Resnet18 Fp32ToInt8) Done"
  return $ret;
}
 
function test_fp32_demo()
{
 
 
  ufw test_fp32 \
    -model=./models/resnet18.prototxt \
    -weights=./models/resnet18.fp32umodel \
    -iterations=${INFE_NUM}
 
  ret=$?; if [ $ret -ne 0 ]; then echo "#ERROR: Test Resnet-FP32"; popd; return $ret;fi
  
 
  echo "#INFO: Test Resnet-FP32 Done";
  return $ret;
}
 
function test_int8_demo()
{
 
 
  ufw test_int8 \
    -model=./models/resnet18_test_int8_unique_top.prototxt \
    -weights=./models/resnet18.int8umodel \
    -iterations=${INFE_NUM}
 
  ret=$?; if [ $ret -ne 0 ]; then echo "#ERROR: Test Resnet-INT8"; popd; return $ret;fi
 
  echo "#INFO: Test Resnet-INT8 Done";
 
 
  return $ret;
}
 
 
function dump_tensor_fp32_demo()
{
  echo "#INFO: DUMP Resnet-FP32...";
 
  dump_tensor_data_use_pb \
    ./models/resnet18_test_fp32_unique_top.prototxt \
    ./models/resnet18.fp32umodel \
    1 CPU
 
  ret=$?; if [ $ret -ne 0 ]; then echo "#ERROR: DUMP Resnet-FP32"; popd; return $ret;fi
 
 
  echo "#INFO: Test Dump Tensor fp32 Done";
  return $ret;
}
 
function dump_tensor_int8_demo()
{
 
  echo "#INFO: DUMP Resnet-INT8...";
  dump_tensor_data_use_pb \
    ./models/resnet18_test_int8_unique_top.prototxt \
    ./models/resnet18.int8umodel \
    1    INT8_NEURON
 
  ret=$?; if [ $ret -ne 0 ]; then echo "#ERROR: DUMP Resnet-INT8"; popd; return $ret;fi
 
  echo "#INFO: Test Dump Tensor int8 Done";
  return $ret;
}