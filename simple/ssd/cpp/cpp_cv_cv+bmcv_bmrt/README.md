# demo for bmopencv decode + bmcv preprocess

* SOC
> make on x86 docker, but run on SOC

```shell
make -f Makefile.arm

# for video
./ssd300_cv_cv+bmcv_bmrt.arm video your_path/*.avi  your_path/*.bmodel loops

# the output is in results with filename such as out-batch-t_id_video.jpg, you can use any picture tool to open it
# only support bmodel of batch_size=1

# for image
./ssd300_cv_cv+bmcv_bmrt.arm image your_path/*.jpg  your_path/*.bmodel loops

# the output is in results with filename such as out-batch-t_id_name.jpg, you can use any picture tool to open it
# only support bmodel of batch_size=1
```
