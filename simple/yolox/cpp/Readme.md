# Example of YOLOX with Sophon Inference

**this example can run in pcie with docker and soc**

## For pcie with docker(all steps in pcie with docker)

### Environment configuration 

```shell
# bmnnsdk2 should be download and uncompressed
cd bmnnsdk2-bm1684_vx.x.x/scripts
./install_lib.sh nntc
source envsetup_pcie.sh
```

### Build example
``` shell
make -f Makefile.pcie
```

### Run example

``` shell
./yolox_sail.pcie video <video url> <bmodel path> <test count> <detect threshold> <nms threshold> <save path> <device id>
```
- video           :test file is video, otherwise is picture
- video url       :video name or picture path
- bmodel path     : bmodel file name
- test count      : video inference count, does not take effect in picture mode
- detect threshold: detect threshold(0.25)
- nms threshold   : nms threshold(0.45)
- save path       : result save path
- device id       : device id

### Result
result in your save path

for picture,  picture save name is same as original name, save txt name is [ost picture name]_[bmodel name].txt

for video, save picture count is batch_size*loops, name is frame_[frame idx]_device_[device id].jpg, save txt name is [video name]_[bmodel name].txt


## For soc

### Environment configuration (in docker with pcie)

```shell [with SC5]
# bmnnsdk2 should be download and uncompressed
cd bmnnsdk2-bm1684_vx.x.x/scripts
./install_lib.sh nntc
source envsetup_pcie.sh 
```

```shell [not SC5]
# bmnnsdk2 should be download and uncompressed
cd bmnnsdk2-bm1684_vx.x.x/scripts
./install_lib.sh nntc
source envsetup_cmodel.sh 
```

### Build example(in docker with pcie)

``` shell
    make -f Makefile.arm
```
### Copy build result to soc

### Run example(in soc)

``` shell
    ./yolox_sail.arm video <video url> <bmodel path> <test count> <detect threshold> <nms threshold> <save path> <device id>
```
- video           :test file is video, otherwise is picture
- video url       :video name or picture path
- bmodel path     : bmodel file name
- test count      : video inference count, does not take effect in picture mode
- detect threshold: detect threshold(0.25)
- nms threshold   : nms threshold(0.45)
- save path       : result save path
- device id       : device id

### result

result in your save path

for picture,  picture save name is same as original name, save txt name is [ost picture name]_[bmodel name]_cpp.txt

for video, save picture count is batch_size*loops, name is frame_[frame idx]_device_[device id].jpg, save txt name is [video name]_[bmodel name]_cpp.txt