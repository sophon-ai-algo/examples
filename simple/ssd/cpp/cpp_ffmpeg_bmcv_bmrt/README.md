# demo for ffmpeg+bmcv
## usage:
* x86 pcie

> make on x86 docker, and run on x86 docker.

```shell
make -f Makefile.pcie
./ssd300_ffmpeg_bmcv_bmrt.pcie your_path/*.avi  your_path/*.bmodel loops devid

# ffmpeg can support all common video codec
# the output is in results with filename such as int8|fp32_dev_id_frame_id.jpg, you can use any picture tool to open it
```
* arm pcie
> make on aarch64 docker, but run on aarch64 PCIE host.

```shell
make -f Makefile.arm_pcie
./ssd300_ffmpeg_bmcv_bmrt.arm_pcie your_path/*.avi  your_path/*.bmodel loops devid

# ffmpeg can support all common video codec
# the output is in results with filename such as int8|fp32_dev_id_frame_id.jpg, you can use any picture tool to open it
```

* SOC
> make on x86 docker, but run on SOC.

```shell
make -f Makefile.arm
./ssd300_ffmpeg_bmcv_bmrt.arm your_path/*.avi  your_path/*.bmodel loops devid

# ffmpeg can support all common video codec
# the output is in results with filename such as int8|fp32_dev_id_frame_id.jpg, you can use any picture tool to open it
```


