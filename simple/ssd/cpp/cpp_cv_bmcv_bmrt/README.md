# demo for bmopencv decode + bmcv preprocess
## usage:
* x86 pcie

```shell
make -f Makefile.pcie
./ssd300_cv_bmcv_bmrt.pcie video your_path/*.avi  your_path/*.bmodel loops devid

# the output is in results with filename such as out-batch-int8|fp32-t_id_dev_id_video.jpg, you can use any picture tool to open it
# only support bmodel of batch_size=1

./ssd300_cv_bmcv_bmrt.pcie image your_path/*.jpg  your_path/*.bmodel loops dev_id

# the output is in results with filename such as out-int8|fp32-id_imagename.jpg, you can use any picture tool to open it
# only support bmodel of batch_size=1
```
```
* arm pcie

> build in arm pcie host,  run on arm pcie host.

```shell
make -f Makefile.arm_pcie
./ssd300_cv_bmcv_bmrt.arm_pcie video your_path/*.avi  your_path/*.bmodel loops devid

# the output is in results with filename such as out-batch-int8|fp32-t_id_dev_id_video.jpg, you can use any picture tool to open it


./ssd300_cv_bmcv_bmrt.arm_pcie image your_path/*.jpg  your_path/*.bmodel loops devid

# the output is in results with filename such as out-int8|fp32-id_imagename.jpg, you can use any picture tool to open it
```
* SOC

> make on x86 docker, but run on SE5

```shell
make -f Makefile.arm
./ssd300_cv_bmcv_bmrt.arm video your_path/*.avi  your_path/*.bmodel loops devid

# the output is in results with filename such as out-batch-int8|fp32-t_id_dev_id_video.jpg, you can use any picture tool to open it
# only support bmodel of batch_size=1

./ssd300_cv_bmcv_bmrt.arm image your_path/*.jpg  your_path/*.bmodel loops devid

# the output is in results with filename such as out-int8|fp32-id_imagename.jpg, you can use any picture tool to open it
# only support bmodel of batch_size=1
```
