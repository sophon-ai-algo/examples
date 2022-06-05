## Example of SSD300 decoded by bm-ffmpeg, preprocessed by bmcv, inference by sail.

## Usage:

* environment configuration on PCIE mode.

```shell
# set LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$REL_TOP/lib/ffmpeg/pcie:$REL_TOP/lib/decode/pcie

# install sophon python whl
cd $REL_TOP/lib/sail/python3/pcie/py37
pip3 install sophon-x.x.x-py3-none-any.whl --user
```

* environment configuration on SOC mode.

```shell
# set PYTHONPATH
# export PYTHONPATH=/workspace/lib/sail/python3/soc/py35/sophon:$PYTHONPATH
```

* A SSD example using bm-ffmpeg to decode and using bmcv to preprocess, with batch size is 1.

```shell
# bmodel: bmodel path, can be fp32 or int8 model
# input:  input path, can be image/video file or rtsp stream
# loops:  frames count to be detected, default: 1
# compare: conpare file path  (default is false)     

python3 ./det_ssd_bmcv.py \
    --bmodel your-path-to-bmodel \
    --input your-path-to-input \
    --loops frames_count_to_detect \
    --compare your-compare-file-path \
    --tpu_id
```

* A SSD example with batch size is 4 for acceleration of int8 model, using bm-ffmpeg to decode and using bmcv to preprocess.

```shell
# bmodel: bmodel path of int8 model
# input:  input path, video file or rtsp stream
# loops:  loop number of inference, each loop processes 4 frames. default: 1
# compare: conpare file path  (default is false)     
python3 ./det_ssd_bmcv_4b.py \
    --bmodel your-path-to-bmodel \
    --input your-path-to-input \
    --loops frames_count_to_detect \
    --compare your-compare-file-path \
    --tpu_id tpu_id