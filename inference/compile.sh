builddir=cmake-build-debug

set -e
rm -fr $builddir

supported_arch="x86 arm64 mips64 soc"
function build_app() 
{
    if [ $# != 1 ]; then
        echo "usage: bulid_app [x86|arm64|mips64|soc]"
        return 1
    fi

    local verify=false
    for ia in ${supported_arch[@]} 
    do 
        if [ "$ia" == "$1" ];then
            verify=true
            break
        fi
    done
    
    if [ "$verify" == "false" ];then
        echo "usage: build_app x86|arm64|mips64|soc"
        return 1
    fi

    local target_arch=$1

    rm -fr $builddir
    mkdir $builddir
    cd $builddir
    
    cmake_params="-DTARGET_ARCH=$target_arch -DUSE_QTGUI=OFF"
    
    if [ "$target_arch" == "arm64" -o "$target_arch" == "soc" ]; then
        cmake_params="$cmake_params -DCMAKE_TOOLCHAIN_FILE=toolchain-aarch64-linux.cmake"
    elif [ "$target_arch" == "mips64" ];then
        cmake_params="$cmake_params -DCMAKE_TOOLCHAIN_FILE=toolchain-mips64-linux.cmake"
    fi
    
    if [ "$1" == "client" ]; then
       cmake -DTARGET_ARCH=x86 -DUSE_BM_FFMPEG=OFF -DUSE_BM_OPENCV=OFF ..
    else
       cmake $cmake_params ..
    fi
    
    make -j4
    cd ..
    
}

function release_example_apps() {
    local arch=$1
    local all_app_list="openpose_demo yolov5s_demo facedetect_demo retinaface_demo safe_hat_detect_demo video_stitch_demo"
    if [[ $arch == "soc" ]];then
        all_app_list="openpose_demo yolov5s_demo facedetect_demo safe_hat_detect_demo video_stitch_demo"
    fi

    for app in ${all_app_list[@]}
    do
         # copy bin files
         if [ "$app" != "client" ]; then
             mkdir -p release/$app/$arch
             cp $builddir/bin/$app release/$app/$arch/
             cp ./examples/cameras.json release/$app/
         fi
    done
}

function release_others() {
  local arch=$1
  local all_app_list="cvs10 dr_demo"
  for app in ${all_app_list[@]}
  do
     mkdir -p release/$app/$arch
     cp $builddir/bin/$app release/$app/$arch/
     cp ./examples/cameras.json release/$app/
  done
}


function build_all() {
    local target_arch_list="x86 soc"
	if [ -n "$1" ];then
		target_arch_list=$1
	fi	
	for arch in ${target_arch_list[@]}
	do
		build_app $arch
        if [ "$?" == "1" ];then
            break
        fi
		release_example_apps $arch
		release_others $arch
	done
}

build_all $1 

