import os
os.environ['GLOG_minloglevel'] = '2'
import ufw
import cv2
from ssd_vgg300 import SSDNet
import visualization

net_shape = (300, 300)
data_format = 'NHWC'

ufw.set_mode_cpu()
# Build SSD network
model = './models/ssd_vgg300/ssd_vgg300_deploy_fp32.prototxt'
weight = './models/ssd_vgg300/ssd_vgg300.fp32umodel'
ssd_net = ufw.Net(model, weight)
ssd = SSDNet(ssd_net)

path = './sample/'
image_names = sorted(os.listdir(path))

for ind in range(len(image_names)):
    img_name = image_names[ind]
    print('detect image: {}'.format(img_name))
    img = cv2.imread(path + img_name, 1)
    rclasses, rscores, rbboxes = ssd.process_image(img[:, :, ::-1])
    visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes,
                                     visualization.colors_plasma)
    image_names_ = img_name.split('.')[0]
    cv2.imwrite(image_names_ + '_fp32_detected.jpg', img)
