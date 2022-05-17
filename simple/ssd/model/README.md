#for SSD300 model:

## about this model

The demo uses a pre-trained model by Caffe. The original project can be found:

	https://github.com/weiliu89/caffe/tree/ssd

## download orgin caffemodel
```
   $ ./download_ssd_model.sh
```
   MD5:
   94587f4ddfa49a23b6dc038aef66e2bb  VGG_VOC0712_SSD_300x300_iter_120000.caffemodel
   c3065f5e17e0e41c949e107ef46e50f8  deploy.prototxt

   Note: if can't download the SSD model by the above method, pls download it from 
         BaiduCloudDisk: https://pan.baidu.com/s/1pLxeLaVoisqN7IVyfrNhag Password: i4x9
## convert caffemodel to float32 bmodel
```
   $ ./gen_bmodel.sh

```

## convert caffemodel to int8 bmodel
```
    1. fetch ssd300 demo dataset from BaiduCloudDisk:https://pan.baidu.com/s/1o9e7uqKBFx0MODssm4JdiQ Password：nl7v
    2. calibration caffe model to umodel and convert umodel to bmodel
    $ copy data.mdb and lock.mdb to SSD_object/model/data/VOC0712
    $ ./gen_umodel_int8bmodel.sh
    $ ls ssd300.int8umodel
you can find umodel on current directory
    $ ls int8model/
you can find bmodel named compilation_1.bmodel and compilation_4.bmodel.
compilation_1.bmodel for 1 batch bmodel .
compilation_4.bmodel for 4 batch bmodel .
```

