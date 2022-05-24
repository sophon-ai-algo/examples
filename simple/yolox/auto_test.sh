SDK_PATH=$1

function judge_ret() {
  if [[ $1 == 0 ]]; then
    echo "Passed: $2"
    echo ""
  else
    echo "Failed: $2"
    exit 1
  fi
  sleep 3
}

function judge_error_ret() {
  if [[ $1 != 0 ]]; then
    echo "Passed: $2"
    echo ""
  else
    echo "Failed: $2"
    exit 1
  fi
  sleep 3
}

function download_nas()
{
    if [ $# -eq 0 ];then
        judge_ret 1 "download_nas"
    fi
    command="grep"

    case "$OSTYPE" in
    solaris*) echo -e " OSType: SOLARIS! Aborting.\n"; exit 1 ;;
    darwin*)  echo -e " OSType: MACOSX!\n"; command="ggrep" ;;
    linux*)   echo -e " OSType: LINUX!\n" ;;
    bsd*)     echo -e " OSType: BSD! Aborting.\n"; exit 1 ;;
    msys*)    echo -e " OSType: WINDOWS! Aborting.\n"; exit 1 ;;
    *)        echo -e " OSType: unknown: $OSTYPE\n"; exit 1 ;;
    esac
    # judge platform, on macOS we need use ggrep command

    type $command >/dev/null 2>&1 || { echo >&2 "Using brew to install GUN grep first.  Aborting."; exit 1; }
    
    web_prefix="http://219.142.246.77:65000/fsdownload/"
    file_url=$1
    
    # id=echo $file_url | ggrep -Po '(?<=sharing/).*(?=/)'
    id=`echo $file_url | cut -d "/" -f 5`
    sid=`curl -i $file_url | $command  -Po '(?<=sid=).*(?=;path)'`
    v=`curl -i $file_url | $command -Po '(?<=none"&v=).*(?=">)'`
    file_name=`curl -b "sharing_sid=${sid}" -i "http://219.142.246.77:65000/sharing/webapi/entry.cgi?api=SYNO.Core.Sharing.Session&version=1&method=get&sharing_id=%22${id}%22&sharing_status=%22none%22&v=${v}" | $command -Po '(?<="filename" : ").*(?=")'`
    
    if [ $# -eq 2 ];then
        save_path=$2
    else
        save_path=$file_name
    fi
    curl -o $save_path -b "sharing_sid=${sid}" "${web_prefix}${id}/${file_name}"
    
    line_0=`cat $save_path|awk -F "\"" '{print $1}'`
    line_1=`cat $save_path|awk -F "\"" '{print $2}'`
    if [ $line_0 = "{" ];then
        if [ $line_1 = "error" ];then
            judge_ret 1 "download "$file_url
        fi
    fi
}

function download_files()
{
    curr_dir=$(pwd)
    data_dir=$(pwd)/data
    if [ ! -d $data_dir ]; then
        mkdir $data_dir
        echo "create dirctory: "$data_dir
    fi
    
    file_name=$data_dir/yolox_s_int8_bs4.bmodel
    if [ ! -f $file_name ]; then
        echo "Start download file: "$file_name
        file_url=http://219.142.246.77:65000/sharing/OZ7RjrjNB
        download_nas $file_url $file_name
        echo "Downloaded"$file_name
    else
        echo "File already existed: "$file_name", need not to download!"
    fi

    file_name=$data_dir/yolox_s_fp32_bs1.bmodel
    if [ ! -f $file_name ]; then
        echo "Start download file: "$file_name
        file_url=http://219.142.246.77:65000/sharing/Zao2JR3AR
        download_nas $file_url $file_name
        echo "Downloaded"$file_name
    else
        echo "File already existed: "$file_name", need not to download!"
    fi

    file_name=$data_dir/test_data.tar.gz
    if [ ! -f $file_name ]; then
        echo "Start download file: "$file_name
        file_url=http://219.142.246.77:65000/sharing/7z2YjOMag
        download_nas $file_url $file_name
        echo "Downloaded"$file_name
    else
        echo "File already existed: "$file_name", need not to download!"
    fi

    pushd $data_dir
    tar -vxf test_data.tar.gz
    popd
}

function run_make(){
  make -f Makefile.pcie clean
  make -f Makefile.pcie 
}

function run_make_sdkpath(){
  make -f Makefile.pcie clean
  make -f Makefile.pcie sdk_dir=$SDK_PATH
}

function build_cpp(){
    pushd ./cpp
    if [ -e ./Makefile.pcie ]; then
      if [ $# -eq 1 ]; then
        OUT_BUILD=$(run_make_sdkpath)
        if [[ $OUT_BUILD =~ "failed" ]]
        then
          judge_ret 1 "build_cpp"
        else
          judge_ret 0 "build_cpp"
        fi
      else
        OUT_BUILD=$(run_make)
        if [[ $OUT_BUILD =~ "failed" ]]
        then
          judge_ret 1 "build_cpp"
        else
          judge_ret 0 "build_cpp"
        fi
      fi
    else
      judge_ret 1 "build_cpp"
    fi
    popd
}

function run_example_cpp(){
    ./cpp/yolox_sail.pcie pic ./data/test_data ./data/yolox_s_int8_bs4.bmodel 0 0.25 0.45 save_result $1
    judge_ret $? "run_example_cpp [yolox_s_int8_bs4.bmodel]"
    ./cpp/yolox_sail.pcie pic ./data/test_data ./data/yolox_s_fp32_bs1.bmodel 0 0.25 0.45 save_result $1
    judge_ret $? "run_example_cpp [yolox_s_fp32_bs1.bmodel]"
}

function run_example_py(){
    python3 python/det_yolox_sail.py \
        --is_video=0 \
        --loops=0 \
        --file_name=./data/test_data/ \
        --bmodel_path=./data/yolox_s_int8_bs4.bmodel \
        --detect_threshold=0.25 \
        --nms_threshold=0.45 \
        --save_path=save_result \
        --device_id=$1
    judge_ret $? "run_example_py [yolox_s_int8_bs4.bmodel]"
    python3 python/det_yolox_sail.py \
        --is_video=0 \
        --loops=0 \
        --file_name=./data/test_data/ \
        --bmodel_path=./data/yolox_s_fp32_bs1.bmodel \
        --detect_threshold=0.25 \
        --nms_threshold=0.45 \
        --save_path=save_result \
        --device_id=$1
    judge_ret $? "run_example_py [yolox_s_fp32_bs1.bmodel]"
}

function get_tpu_num() {
  local tpu_num=0
  for id in {0..128}
  do
     if [ -c "/dev/bm-sophon$id" ];then
        tpu_num=$(($tpu_num+1))
     fi
  done
  echo $tpu_num
}

function get_tpu_ids() {
  tpu_num=$(get_tpu_num)
  declare -a tpus
  for i in $( seq 0 $(($tpu_num-1)) )
  do
    tpus+=($i)
  done
  echo ${tpus[*]}
}

function verify_result(){
    python3 python/calc_recall_accuracy.py \
        --ground_truths=./data/test_data/ground_truths.txt \
        --detections=./save_result/test_data_yolox_s_int8_bs4_py.txt \
        --iou_threshold=0.6
    judge_ret $? "Verify [python] [yolox_s_int8_bs4.bmodel]"

    python3 python/calc_recall_accuracy.py \
        --ground_truths=./data/test_data/ground_truths.txt \
        --detections=./save_result/test_data_yolox_s_int8_bs4_cpp.txt \
        --iou_threshold=0.6
    judge_ret $? "Verify [cpp] [yolox_s_int8_bs4.bmodel]"

    python3 python/calc_recall_accuracy.py \
        --ground_truths=./data/test_data/ground_truths.txt \
        --detections=./save_result/test_data_yolox_s_fp32_bs1_py.txt \
        --iou_threshold=0.6
    judge_ret $? "Verify [python] [yolox_s_fp32_bs1.bmodel]"

    python3 python/calc_recall_accuracy.py \
        --ground_truths=./data/test_data/ground_truths.txt \
        --detections=./save_result/test_data_yolox_s_fp32_bs1_cpp.txt \
        --iou_threshold=0.6
    judge_ret $? "Verify [cpp] [yolox_s_fp32_bs1.bmodel]"
    
}

download_files
if [ $# -eq 1 ];then
  build_cpp $1
  else
  build_cpp
fi

TPU_IDS=$(get_tpu_ids)

if [ ! -d save_result ]; then
mkdir save_result
fi
rm -rf save_result/*

for tpu_id in ${TPU_IDS[@]}
do
  run_example_cpp $tpu_id
  run_example_py $tpu_id
  verify_result
done
