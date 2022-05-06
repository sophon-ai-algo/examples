

function gettop
{
  local TOPFILE=face_demo.sh
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



function printf_job()
{
printf "========start===========\n"
printf "                        \n"
printf "=========end============\n"
}



function  build_face_demo_fp32()
{
 
  pushd $FACE_DEMO_TOP
  
  make clean;
  make INT8_MODE=0;
  
  popd
}

function  build_face_demo_int8()
{
 
  pushd $FACE_DEMO_TOP
  
  make clean;
  make INT8_MODE=1;
  
  popd
}

function detect_squeezenet_fp32()
{
 build_face_demo_fp32

  ./bin/demo_detect \
   models/squeezenet/squeezenet_21k_deploy.prototxt \
   models/squeezenet/squeezenet_21k.fp32umodel \
  0.7 sample/test_2.jpg $1
}


#generate int8umodel:  squeezenet_21k.int8umodel
function convert_squeezenet_to_int8()
{
  calibration_use_pb  \
    quantize \
    -model=models/squeezenet/squeezenet_21k_test.prototxt   \
    -weights=models/squeezenet/squeezenet_21k.fp32umodel  \
    -iterations=100 \
    -bitwidth=TO_INT8

}

#use  squeezenet_21k.int8umodel to detect face
function detect_squeezenet_int8()
{
  build_face_demo_int8

  ./bin/demo_detect \
   models/squeezenet/squeezenet_21k_deploy_int8_unique_top.prototxt \
   models/squeezenet/squeezenet_21k.int8umodel \
  0.7 sample/test_2.jpg $1

}

FACE_DEMO_TOP=$(gettop)
export FACE_DEMO_TOP
