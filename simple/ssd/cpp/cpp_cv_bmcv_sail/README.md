## Example of SSD300 decoded by bm-opencv, preprocessed by bmcv, inference by sail.

## Usage:

* Deploy on SOC mode.

> make on x86 docker, but run on SE5

```shell
make -f Makefile.arm

# bmodel: bmodel path, only support batch_size=1, can be fp32 or int8 model
# input:  input path, can be image/video file or rtsp stream
# loops:  frames count to be detected, default: 1
./ssd300_cv_bmcv_sail.arm \
    --bmodel your-path-to-bmodel \
    --input your-path-to-input \
    --loops frames_count_to_detect
```
