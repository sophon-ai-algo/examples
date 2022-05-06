
function gettop
{
  local TOPFILE=envsetup.sh
  if [ -n "$TOP" -a -f "$TOP/$TOPFILE" ] ; then
    # The following circumlocution ensures we remove symlinks from TOP.
    (cd $TOP; PWD= /bin/pwd)
  else
    if [ -f $TOPFILE ] ; then
      # The following circumlocution (repeated below as well) ensures
      # that we record the true directory name and not one that is
      # faked up with symlink names.
      PWD= /bin/pwd
    else
      local HERE=$PWD
      T=
      while [ \( ! \( -f $TOPFILE \) \) -a \( $PWD != "/" \) ]; do
        \cd ..
        T=`PWD= /bin/pwd -P`
      done
      \cd $HERE
      if [ -f "$T/$TOPFILE" ]; then
        echo $T
      fi
    fi
  fi
}

function mtcnn_build()
{
    make clean 
    make -j4
}

function run_demo_float()
{
    ./bin/mtcnn_demo demo --model_list=data/model_file_list_orig.txt --image_list=data/image_file_list_demo.txt --mode="FLOAT"
}

function dump_fddb_lmdb()
{
    rm lmdb/* -rf
    ./bin/mtcnn_demo dump --model_list=data/model_file_list_orig.txt --image_list=data/image_file_list_fddb.txt --iterations=1000
}

function convert_mtcnn_demo_pnet_to_int8_pb()
{
    calibration_use_pb  \
    quantize \
    -model=models/pnet_lmdb.prototxt \
    -weights=models/pnet.caffemodel \
    -iterations=1000 \
    -bitwidth=TO_INT8 \
    -save_test_proto=true
}

function convert_mtcnn_demo_rnet_to_int8_pb()
{
    calibration_use_pb  \
    quantize \
    -model=models/rnet_lmdb.prototxt \
    -weights=models/rnet.caffemodel \
    -iterations=1000 \
    -bitwidth=TO_INT8 \
    -save_test_proto=true
}

function convert_mtcnn_demo_onet_to_int8_pb()
{
    calibration_use_pb  \
    quantize \
    -model=models/onet_lmdb.prototxt \
    -weights=models/onet.caffemodel \
    -iterations=1000 \
    -bitwidth=TO_INT8 \
    -save_test_proto=true
}

function run_demo_int8()
{
    ./bin/mtcnn_demo demo --model_list=data/model_file_list_int8.txt --image_list=data/image_file_list_demo.txt  --mode="INT8"
}


MTCNN_TOP=$(gettop)
export MTCNN_TOP
