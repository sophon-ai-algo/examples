## How to build ?

 * modify the PRODUCTFORM and top_dir at the Makefile accoring to the actual situation
 * PRODUCTFORM: can be pcie,soc, arm_pcie
 * top_dir: the root directory of sdks

For example, on x86 with pcie cards:
``` bash
make PRODUCTFORM=x86 -j4
```
## How to run ?

For more information, please run help command.
``` bash 
[test command] --help
```


