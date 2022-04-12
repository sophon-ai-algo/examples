# Example of YOLOX with Sophon Inference

  **this example can run in pcie with docker and soc**

## For pcie with docker

### Environment configuration 

```shell
# bmnnsdk2 should be download and uncompressed
cd bmnnsdk2-bm1684_vx.x.x/scripts
./install_lib.sh nntc
source envsetup_pcie.sh
```

### Python module named sophon is needed to install

```shell
# the wheel package is in the bmnnsdk2:
pip3 uninstall -y sophon
# get your python version
python3 -V
# choose the same verion of sophon wheel to install
# the following py3x maybe py35, py36, py37 or py38
# for x86
pip3 install ../lib/sail/python3/pcie/py3x/sophon-?.?.?-py3-none-any.whl --user
```

### run example

``` shell
    python3 det_yolox_sail.py \
        --bmodel_path=your-path-to-bmodel \
        --is_video=test-file-is-video-or-not \
        --file_name=your-video-name-or-picture-path \
        --loops=video-inference-count \
        --device_id=use-tpu-id \                # defaule 0
        --detect_threshold=detect-threshold \   # default 0.25
        --nms_threshold=nms-threshold \         # default 0.45
        --save_path=result-save-path
```

### Result
result in your save path

for picture,  picture save name is same as original name, save txt name is [ost picture name]_[bmodel name].txt

for video, save picture count is batch_size*loops, name is frame_[frame idx]_device_[device id].jpg, save txt name is [video name]_[bmodel name].txt


## For soc
### Environment configuration 

``` shell
    export PATH=$PATH:/system/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/system/lib/:/system/usr/lib/aarch64-linux-gnu
    export PYTHONPATH=$PYTHONPATH:/system/lib
```

### If not installed numpy, install numpy

``` shell
    sudo pip3 install numpy==1.17.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Run example

``` shell
    python3 det_yolox_sail.py \
        --bmodel_path=your-path-to-bmodel \
        --is_video=test-file-is-video-or-not \
        --file_name=your-video-name-or-picture-path \
        --loops=video-inference-count \
        --device_id=use-tpu-id \                # defaule 0
        --detect_threshold=detect-threshold \   # default 0.25
        --nms_threshold=nms-threshold \         # default 0.45
        --save_path=result-save-path
```

### Result
result in your save path

for picture,  picture save name is same as original name, save txt name is [ost picture name]_[bmodel name]_py.txt

for video, save picture count is batch_size*loops, name is frame_[frame idx]_device_[device id].jpg, save txt name is [video name]_[bmodel name]_py.txt


## calculate recall and accuracy
``` shell
    python3 py_sail/calc_recall_accuracy.py \
        --ground_truths=your-ground_truths-file \
        --detections=your-detections-file \
        --iou_threshold=your-iou-threshold  #default 0.6
```
