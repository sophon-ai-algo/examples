function judge_ret() {
  if [[ $1 == 0 ]]; then
    echo "Passed: $2"
    echo ""
  else 
    if [[ $1 == 1000 ]]; then
        echo "$2"
        exit 1
    else
        echo "Failed: $2"
        exit 1
    fi
  fi
  sleep 2
}

if [ $# -eq 7 ];then
    sdk_dir=$1
    dst_image_path=$2"_enhance" 
    lmdbfolder=$dst_image_path"_lmbd"
    rm -rf lmdbfolder
    python3 image_resize.py --ost_path=$2 --dst_path=$dst_image_path --dst_width=$4 --dst_height=$5
    judge_ret $? "image_resize"
    if [ $7 -eq 1 ]; then
        python3 image_resize_sophgo.py --ost_path=$2 --dst_path=$dst_image_path --dst_width=$4 --dst_height=$5
        judge_ret $? "image_resize_sophgo"
    fi
    echo "Start convert_imageset..."
    python3 ${sdk_dir}/examples/calibration/create_lmdb_demo/convert_imageset.py \
        --imageset_rootfolder=$dst_image_path \
        --imageset_lmdbfolder=$lmdbfolder \
        --resize_height=$5 \
        --resize_width=$4 \
        --shuffle=True \
        --bgr2rgb=False \
        --gray=False
    judge_ret $? "convert_imageset"
    python3 gen_fp32_umodel.py \
        --trace_model=$3 \
        --data_path=$lmdbfolder \
        --dst_width=$4 \
        --dst_height=$5

    judge_ret $? "gen_fp32_umodel"
    python3 rewrite_fp32umodel.py --trace_model=$3
    judge_ret $? "rewrite_fp32umodel"

    ost=$3
    trace_name=${ost##*/}
    temp_name=${trace_name%.*}
    path=${ost%${temp_name}*}${temp_name%%.*}
    model=${path}"/"${trace_name%.*}"_bmnetp_test_fp32.prototxt"
    weights=${path}"/"${trace_name%.*}"_bmnetp.fp32umodel"

    echo "Start calibration..."
    calibration_use_pb \
        quantize \
        -model=$model \
        -weights=$weights \
        -iterations=100 \
        -bitwidth=TO_INT8
    judge_ret $? "calibration"

    batch_size=$6
    model_int8=${path}"/"${trace_name%.*}"_bmnetp_deploy_int8_unique_top.prototxt"
    weight_int8=${path}"/"${trace_name%.*}"_bmnetp.int8umodel"
    outdir=${path}"/int8model_bs"${batch_size}

    echo "Start convert to int8 bmodel..."
    bmnetu -model ${model_int8} \
        -weight ${weight_int8} \
        -max_n ${batch_size} \
        -prec=INT8 \
        -dyn=0 \
        -cmp=1 \
        -target=BM1684 \
        -outdir=${outdir}
    judge_ret $? "convert to int8 bmodel"


else if [ $# -eq 6 ];then
        dst_image_path=$1"_enhance" 
        lmdbfolder=$dst_image_path"_lmbd"
        rm -rf lmdbfolder
        python3 image_resize.py --ost_path=$2 --dst_path=$dst_image_path --dst_width=$4 --dst_height=$5
        judge_ret $? "image_resize"
        python3 ${sdk_dir}/examples/calibration/create_lmdb_demo/convert_imageset.py \
            --imageset_rootfolder=$dst_image_path \
            --imageset_lmdbfolder=$lmdbfolder \
            --resize_height=$5 \
            --resize_width=$4 \
            --shuffle=True \
            --bgr2rgb=False \
            --gray=False

    judge_ret $? "convert_imageset"
    python3 gen_fp32_umodel.py \
        --trace_model=$3 \
        --data_path=$lmdbfolder \
        --dst_width=$4 \
        --dst_height=$5

    judge_ret $? "gen_fp32_umodel"
    python3 rewrite_fp32umodel.py --trace_model=$3
    judge_ret $? "rewrite_fp32umodel"

    ost=$3
    trace_name=${ost##*/}
    temp_name=${trace_name%.*}
    path=${ost%${temp_name}*}${temp_name%%.*}
    model=${path}"/"${trace_name%.*}"_bmnetp_test_fp32.prototxt"
    weights=${path}"/"${trace_name%.*}"_bmnetp.fp32umodel"

    echo "Start calibration..."

    calibration_use_pb \
        quantize \
        -model=$model \
        -weights=$weights \
        -iterations=100 \
        -bitwidth=TO_INT8
    judge_ret $? "calibration"

    batch_size=$6
    model_int8=${path}"/"${trace_name%.*}"_bmnetp_deploy_int8_unique_top.prototxt"
    weight_int8=${path}"/"${trace_name%.*}"_bmnetp.int8umodel"
    outdir=${path}"/bmodel_int8_bs"${batch_size}

    echo "Start convert to int8 bmodel..."
    bmnetu -model ${model_int8} \
        -weight ${weight_int8} \
        -max_n ${batch_size} \
        -prec=INT8 \
        -dyn=0 \
        -cmp=1 \
        -target=BM1684 \
        -outdir=${outdir}
    judge_ret $? "convert to int8 bmodel"

    else 
        judge_ret 1000 "USAGE: $0 sdk_path image_path trace_model resize_w resize_h max_batch_size has_sc5"
    fi
fi