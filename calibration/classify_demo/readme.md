#create_lmdb_demo
-----
请先进入docker环境，再执行下述操作
## 配置运行环境
```bash
cd bmnnsdk2-<version>/scripts  #进入对应版本的sdk脚本文件夹
./install_lib.sh nntc #安装nntoolchain
source ./envsetup_cmodel.sh  #配置不同平台的运行环境
```

## 运行命令
```bash
source classify_demo.sh
convert_to_int8_demo
test_fp32_demo
test_int8_demo
dump_tensor_fp32_demo
dump_tensor_int8_demo
```

## 期望结果
`convert_to_int8_demo` 完成之后打印 `#INFO: Run Example (Resnet18 Fp32ToInt8) Done`
`test_fp32_demo` 完成之后打印 `#INFO: Test Resnet-FP32 Done`
`test_int8_demo` 完成之后打印 `#INFO: Test Resnet-INT8 Done`
`dump_tensor_fp32_demo` 完成之后打印 `#INFO: Test Dump Tensor fp32 Done`
`dump_tensor_int8_demo` 完成之后打印 `#INFO: Test Dump Tensor int8 Done`
并在当前文件夹下生成
`log`, `log_resnet18_test_fp32_unique_top_*`, `log_resnet18_test_int8_unique_top_*`文件夹

`log`文件夹下生成`resnet18*float32.md`和`resnet18*int8.md`
`log_resnet18_test_fp32_unique_top_*`下生成一系列txt文件
`log_resnet18_test_int8_unique_top_*`下生成一系列txt文件