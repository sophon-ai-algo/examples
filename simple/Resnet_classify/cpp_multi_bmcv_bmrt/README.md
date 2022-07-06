# A Multi-Thread Demo

## Description

- The demo program takes a folder of pictures as inputs.
- The input pictures are supposed to be processed by 4 decode threads and N inference threads, where N is specified by user.
- The given image is decoded by OpenCV API, while the pre-processing is done by BMCV API.


## How-To

To run the test, please follow the steps below.
``` shell
cd $sdk_dir/scripts
./install_lib.sh nntc

#for x86 pcie
source envsetup_pcie.sh

#for arm_pcie
source envsetup_arm_pcie.sh
```

### for X86 platform:
``` shell
$ make -f Makefile.pcie
$ ./resnet_multi_bmcv_bmrt.pcie <image path> ../data/models/resnet50.int8.bmodel <threads num> <device id>
```
## for arm64 pcie platform (build in arm pcie, run in arm pcie)
``` shell
$ make -f Makefile.arm_pcie
$ ./resnet_multi_bmcv_bmrt.arm_pcie <image path> ../data/models/resnet50.int8.bmodel <threads num> <device id>
```

## for SOC platform(build in x86, run in soc)
``` shell
$ make -f Makefile.soc
$ ./resnet_multi_bmcv_bmrt.soc <image path> ../data/models/resnet50.int8.bmodel <threads num>
```

Note: <image path> MUST contains 4*N pictures as the test is now hardcoded to batch-size=4.
