function test_auto_cali_demo()
{
    pushd ./auto_cali_demo
    ./extract_tar.sh
    python3 -m ufw.cali.cali_model --model pytorch/resnet18.pt \
        --cali_lmdb imagenet_preprocessed_by_pytorch_100/ --input_shapes '(1,3,224,224)' \
        --test_iterations 50 --net_name resnet18  --postprocess_and_calc_score_class topx_accuracy_for_classify \
        --cali_iterations=100
    popd
}

function test_caffemodel_to_fp32umodel_demo()
{
    pushd ./caffemodel_to_fp32umodel_demo
    python3 ./resnet50_to_umodel.py
    popd
}

function test_classify_demo()
{
    pushd ./classify_demo
    source ./classify_demo.sh
    convert_to_int8_demo
    test_fp32_demo
    test_int8_demo
    dump_tensor_fp32_demo
    dump_tensor_int8_demo
    popd
}

function test_create_lmdb_demo()
{
    pushd ./create_lmdb_demo
    bash download_coco128.sh
    python3 convert_imageset.py \
        --imageset_rootfolder=./coco128/images/train2017 \
        --imageset_lmdbfolder=./lmdb \
        --resize_height=640 \
        --resize_width=640 \
        --shuffle=True \
        --bgr2rgb=False \
        --gray=False
    popd
}

function test_dn_to_fp32umodel_demo()
{
    pushd ./dn_to_fp32umodel_demo
    bash ./get_model.sh
    python3 yolov3_to_umodel.py 
    popd
}

function test_face_demo()
{
    pushd ./face_demo
    source ./face_demo.sh
    detect_squeezenet_fp32
    convert_squeezenet_to_int8
    detect_squeezenet_int8
    popd
}

function test_mx_to_fp32umodel_demo()
{
    pushd ./mx_to_fp32umodel_demo
    python3 mobilenet0.25_to_umodel.py
    popd
}

function test_object_detection_python_demo()
{
    pushd ./object_detection_python_demo
    python3 ssd_vgg300_fp32_test.py
    python3 ssd_vgg300_int8_test.py
    popd
}

function test_on_to_fp32umodel_demo()
{
    pushd ./on_to_fp32umodel_demo
    python3 postnet_to_umodel.py
    popd
}

function test_pp_to_fp32umodel_demo()
{
    pushd ./pp_to_fp32umodel_demo
    python3 ppocr_rec_to_umodel.py
    popd
}

function test_pt_to_fp32umodel_demo()
{
    pushd ./pt_to_fp32umodel_demo
    python3 yolov5s_to_umodel.py
    popd
}


function test_tf_to_fp32umodel_demo()
{
    pushd ./tf_to_fp32umodel_demo
    python3 create_dummy_quant_lmdb.py
    python3 resnet50_v2_to_umodel.py 
    popd
}

# NOTE: DO create_lmdb_test FIRST!!!!!!
function test_yolov5s_demo()
{
    pushd ./yolov5s_demo/auto_cali_demo
    bash ./auto_cali.sh
    pip3 install pycocotools
    python3 -m dfn --url http://219.142.246.77:65000/sharing/ivVtP2yIg
    bash ./regression.sh
    popd
}

function test_all()
{
    test_auto_cali_demo
    test_caffemodel_to_fp32umodel_demo
    test_classify_demo
    test_create_lmdb_demo
    test_dn_to_fp32umodel_demo
    test_face_demo
    test_mx_to_fp32umodel_demo
    test_object_detection_python_demo
    test_on_to_fp32umodel_demo
    test_pp_to_fp32umodel_demo
    test_pt_to_fp32umodel_demo
    test_tf_to_fp32umodel_demo
    test_yolov5s_demo
}

test_all
