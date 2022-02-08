#coding=utf-8

'''
This file is to convert pytorch model to umodel.
'''
import os
import sys
#os.environ['BMNETP_LOG_LEVEL'] = '3'
import ufw.tools as tools

# pt_centernet = [
#     '-m', '/workspace/examples/centernet_test/ctdet_coco_dlav0_1x.torchscript.pt',
#     '-s', '(1,3,512,512)',
#     '-d', 'compilation',
#     '-D', '/workspace/examples/centernet_test/CenterNet_object/data/img_lmdb',
#     '--cmp'
# ]

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('{} usage: {} <pt_model> <lmdb_path> <batch_size> <img_resolution>')
        exit(1)
        
    pt_centernet  = []
    pt_centernet += ['-m', sys.argv[1]]
    pt_centernet += ['-s', '({}, 3, {}, {})'.format(sys.argv[3], sys.argv[4], sys.argv[4])]
    pt_centernet += ['-d', 'compilation']
    pt_centernet += ['-D', sys.argv[2]]
    pt_centernet += ['--cmp']

    print('pt_centernet: {}'.format(pt_centernet))
    
    tools.pt_to_umodel(pt_centernet)
