## How to build
 * modify the PRODUCTFORM and top_dir at the Mafefile accoring to the actual situation
 * PRODUCTFORM: can be x86,soc, arm_pcie
 * top_dir: the root directory of sdks

For example, on x86:
``` bash
make PRODUCTFORM=x86 -j4
```
## How to run ?

For more information, please run help command.
``` bash 
[test command] --help
```