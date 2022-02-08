## Example of YOLOX decoded by bm-ffmpeg, preprocessed by bm-ffmpeg, inference by sail.

## Usage:

* environment configuration on PCIE mode.

```shell
# set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/workspace/lib/ffmpeg/x86: \
       /workspace/lib/decode/x86

# install sophon python whl
cd /workspace/lib/sail/python3/pcie/py35
pip3 install sophon-x.x.x-py3-none-any.whl --user
```

* environment configuration on SOC mode.

```shell
# set PYTHONPATH
export $PYTHONPATH=/workspace/lib/sail/python3/soc/py35/sophon:$PYTHONPATH
```

* A YOLOX example using bm-ffmpeg to decode and using bmcv to preprocess, with batch size is 1.

```shell
# bmodel: bmodel path, can be fp32 or int8 model
# input:  input path, can be video file or rtsp stream
# loops:  frames count to be detected, default: 1
# tpu_id:  tpu use, default: 0
# detect_threshold:  detection threshold
# nms_threshold:  nms threshold
python3 ./det_yolox_bmcv.py \
    --bmodel your-path-to-bmodel \
    --input your-path-to-input \
    --loops frames_count_to_detect \
    --tpu_id use-tpu-id \
    --detect_threshold detect-threshold \
    --nms_threshold nms-threshold 
```


* A YOLOX example using bm-ffmpeg to decode and using bmcv to preprocess, with batch size is 4.

```shell
# bmodel: bmodel path, can be fp32 or int8 model
# input:  input path, can be video file or rtsp stream
# loops:  frames count to be detected, default: 1
# tpu_id:  tpu use, default: 0
# detect_threshold:  detection threshold
# nms_threshold:  nms threshold
python3 ./det_yolox_bmcv_4b.py \
    --bmodel your-path-to-bmodel \
    --input your-path-to-input \
    --loops frames_count_to_detect \
    --tpu_id use-tpu-id \
    --detect_threshold detect-threshold \
    --nms_threshold nms-threshold 
```