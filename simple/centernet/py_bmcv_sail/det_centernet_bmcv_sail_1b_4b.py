import os
import cv2
import torch
import time
from torch import nn
import logging
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import sophon.sail as sail
import colorsys
from datetime import datetime
import argparse
from numpy.lib.stride_tricks import as_strided

BASE_DIR = os.path.dirname(os.path.join(os.getcwd(), __file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data/')

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

class PreProcessor:
  """ Preprocessing class.
  """
  def __init__(self, bmcv, size_w, size_h, scale):
    """ Constructor.
    """
    self.bmcv = bmcv
    self.size_w = size_w
    self.size_h = size_h
    # bgr normalization
    self.ab = [x * scale for x in [0.01358, -1.4131, 0.0143, -1.6316, 0.0141, -1.69103]]
    #self.ab = [x * scale for x in [1, 0, 1, 0, 1, 0]]

  def process(self, input_img, output_img, is_letterbox):
    """ Execution function of preprocessing.
    Args:
    cv_input: sail.BMImage, input image
    bmcv_output: sail.BMImage, output data

    Returns:
    None
    """
    if is_letterbox:
        assert(self.size_w == self.size_h)
        
        # letterbox padding
        if input_img.width() > input_img.height():
            resize_ratio = self.size_w / input_img.width()
            target_w     = int(self.size_w)
            target_h     = int(input_img.height() * resize_ratio)
        else:
            resize_ratio = self.size_h / input_img.height()
            target_w     = int(input_img.width() * resize_ratio)
            target_h     = int(self.size_h)
            
        pad = sail.PaddingAtrr()
        offset_x = 0 if target_w >= target_h  else int((self.size_w  - target_w) / 2)
        offset_y = 0 if target_w <= target_h  else int((self.size_h  - target_h) / 2)
        pad.set_stx(offset_x)
        pad.set_sty(offset_y)
        pad.set_w(target_w)
        pad.set_h(target_h)
        # padding color grey
        pad.set_r(128)
        pad.set_g(128)
        pad.set_b(128)
        
        
        #tmp = self.bmcv.vpp_resize_padding(input_img, self.size_w, self.size_h, pad)
        tmp = self.bmcv.crop_and_resize_padding(input_img, 0, 0, 
                                                input_img.width(), input_img.height(),
                                                self.size_w, self.size_h,
                                                pad)
    else:
        tmp = self.bmcv.vpp_resize(input_img, self.size_w, self.size_h)
    self.bmcv.convert_to(tmp, output_img, ((self.ab[0], self.ab[1]), \
                                        (self.ab[2], self.ab[3]), \
                                        (self.ab[4], self.ab[5])))

class Detector(object):
    """
    This is CenterNet detector class
    """
    _defaults = {
        
        # bmodel模型文件
        "bmodel_path"       : os.path.join(DATA_DIR, 'models/ctdet_coco_dlav0_1x_int8_b1.bmodel'),
        #"bmodel_path"       : os.path.join(DATA_DIR, 'models/ctdet_coco_dlav0_1x_fp32.bmodel'),
        # 类标文件
        "classes_path"      : os.path.join(DATA_DIR, 'coco_classes.txt'),
        # 字体文件
        "font_path"         : os.path.join(DATA_DIR, 'simhei.ttf'),
        #--------------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #--------------------------------------------------------------------------#
        "confidence"        : 0.35,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #--------------------------------------------------------------------------#
        #   是否进行非极大抑制，可以根据检测效果自行选择
        #   backbone为resnet50时建议设置为True、backbone为hourglass时建议设置为False
        #--------------------------------------------------------------------------#
        "nms"               : True,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False, # must be False
    }
    def __init__(self, tpu_id):
        # 加载默认属性值
        self.__dict__.update(self._defaults)
        #---------------------------------------------------#
        #   计算总的类的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = self.get_classes()

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))


        # 加载bmodel
        self.engine         = sail.Engine(self.bmodel_path, tpu_id, sail.IOMode.SYSO)
        self.graph_name     = self.engine.get_graph_names()[0]
        self.input_name     = self.engine.get_input_names(self.graph_name)[0]
        self.output_name    = self.engine.get_output_names(self.graph_name)[0]
        self.input_dtype    = self.engine.get_input_dtype(self.graph_name, self.input_name)
        self.output_dtype   = self.engine.get_output_dtype(self.graph_name, self.output_name)
        self.input_shape    = self.engine.get_input_shape(self.graph_name, self.input_name)
        self.input_w        = int(self.input_shape[-1])
        self.input_h        = int(self.input_shape[-2])
        self.output_shape   = self.engine.get_output_shape(self.graph_name, self.output_name)
        self.handle         = self.engine.get_handle()
        self.input          = sail.Tensor(self.handle, self.input_shape, self.input_dtype, 
                                          True, True)
        self.output         = sail.Tensor(self.handle, self.output_shape, self.output_dtype, 
                                          True, True)
        self.input_tensors  = { self.input_name  : self.input }
        self.output_tensors = { self.output_name : self.output}
        
        self.bmcv           = sail.Bmcv(self.handle)
        self.img_dtype      = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.input_scale    = self.engine.get_input_scale(self.graph_name, self.input_name)
        self.output_scale   = self.engine.get_output_scale(self.graph_name, self.output_name)

        # batch size 1
        if self.output_shape[0] == 1:
            self.input_bmimage  = sail.BMImage(self.handle, 
                                            self.input_w, self.input_h,
                                            sail.Format.FORMAT_BGR_PLANAR, 
                                            self.img_dtype)
        elif self.output_shape[0] == 4:
            self.input_bmimage  = sail.BMImageArray4D(self.handle, 
                                                     self.input_w, self.input_h,
                                                     sail.Format.FORMAT_BGR_PLANAR, 
                                                     self.img_dtype)
        else:
            raise NotImplementedError(
                'This demo not supports inference with batch size {}'.format(self.output_shape[0]))

        self.preprocessor = PreProcessor(self.bmcv, self.input_w, self.input_h, self.input_scale)
        
        logging.info("\n" + "*" * 50 + "\n"
                     "graph_name:    {}\n"
                     "input_name:    {}\n"
                     "output_name:   {}\n"
                     "input_dtype:   {}\n"
                     "output_dtype:  {}\n"
                     "input_shape:   {}\n"
                     "output_shape:  {}\n"
                     "img_dtype:     {}\n"
                     "input_scale:   {}\n"
                     "output_scale:  {}\n".format(self.graph_name, self.input_name, self.output_name,
                                                 self.input_dtype, self.output_dtype, self.input_shape, 
                                                 self.output_shape, self.img_dtype, self.input_scale, self.output_scale)
                                                 + "*" * 50)

    #---------------------------------------------------#
    #   获得类
    #---------------------------------------------------#
    def get_classes(self):
        with open(self.classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names, len(class_names)
    
    def get_net_size(self):
        return int(self.input_w), int(self.input_h)

    def get_handle(self):
        return self.handle

    def get_batchsize(self):
        return int(self.input_shape[0])

    def predict(self, bm_img):
        # resize
        self.preprocessor.process(bm_img, self.input_bmimage, self.letterbox_image)
        # copy to tensor
        self.bmcv.bm_image_to_tensor(self.input_bmimage, self.input)
        # raw image shape
        image_shape = []
        if self.output_shape[0] == 1:
            # batch size = 1
            image_shape.append((bm_img.height(), bm_img.width()))
        else:
            for i in range(4):
                image_shape.append((bm_img[i].height(), bm_img[i].width()))
        
        logging.info('tensor shape {}, input bmimg HxW {}'.format(self.input.shape(), image_shape))
        
        # inference
        start    = time.time()
        self.engine.process(self.graph_name, self.input_tensors, self.output_tensors)
        logging.info('inference time {}ms'.format((time.time() - start) * 1000))
        
        dets     = self.output.asnumpy().astype(np.float32)
        dets    *= self.output_scale
        
        logging.info('inference finish. dets shape -> {}'.format(dets.shape))
        pred_hms = dets[:, :self.output_shape[1] - 4, ...]
        pred_whs = dets[:, self.output_shape[1] - 4:self.output_shape[1] - 2, ...]
        pred_off = dets[:, self.output_shape[1] - 2:, ...]

        # sigmoid
        pred_hms = 1. / (1 + np.exp(-pred_hms))        
        
        # 解码
        outputs = self.decode_bbox(pred_hms, pred_whs, pred_off)

        # 实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大
        results = self.postprocess(outputs, image_shape)
        
        return results
        
        
    def rgb_norm(self, image):
        # input is BGR
        #image   = np.array(image,dtype = np.float32)[:, :, ::-1]
        mean    = [0.40789655, 0.44719303, 0.47026116]
        std     = [0.2886383, 0.27408165, 0.27809834]
        return (image / 255. - mean) / std

    def centernet_correct_boxes(self, box_xy, box_wh, image_shape):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(self.input_shape[-2:])
        image_shape = np.array(image_shape)

        if self.letterbox_image:
            #-----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            #-----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes
  
    def pool_nms_v2(self, A, kernel_size=3, stride=1, padding=1):
        '''
        Parameters:
            A: input 4D array
            kernel_size: int, the size of the window
            stride: int, the stride of the window
            padding: int, implicit zero paddings on both sides of the input
        '''
        assert(len(A.shape) == 4)
        # Padding
        A = np.pad(A, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

        # Window view of A
        output_shape = A.shape[:2] + ((A.shape[-2] - kernel_size) // stride + 1,
                                      (A.shape[-1] - kernel_size) // stride + 1)
        
        kernel_size = (kernel_size, kernel_size)
        element_size = A.strides[-1]
        A_w = as_strided(A, shape   = output_shape + kernel_size, 
                            strides = A.strides[:2] + (stride * A.strides[-2], stride * A.strides[-1]) + A.strides[-2:])
        A_w = A_w.reshape(-1, *kernel_size)

        # the result of pooling
        hmax = A_w.max(axis=(1, 2)).reshape(output_shape)
        mask = (hmax == A[..., padding:-padding, padding:-padding])
        return A[..., padding:-padding, padding:-padding] * mask
        
    
    def pool_nms(self, heat, kernel = 3):
        pad = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep
      
    def postprocess(self, prediction, image_shape):
        output = [None for _ in range(len(prediction))]
        
        #----------------------------------------------------------#
        #   预测只用一张图片，只会进行一次
        #----------------------------------------------------------#
        for i, image_pred in enumerate(prediction):
            detections      = prediction[i]
            if len(detections) == 0:
                continue
            #------------------------------------------#
            #   获得预测结果中包含的所有种类
            #------------------------------------------#
            unique_labels   = np.unique(detections[:, -1])

            for c in unique_labels:
                #------------------------------------------#
                #   获得某一类得分筛选后全部的预测结果
                #------------------------------------------#
                detections_class = detections[detections[:, -1] == c]
                if self.nms:
                    #------------------------------------------#
                    #   使用官方自带的非极大抑制会速度更快一些！
                    #------------------------------------------#
                    keep = nms(
                        detections_class[:, :4] * [self.output_shape[3], self.output_shape[2], self.output_shape[3], self.output_shape[2]],
                        detections_class[:, 4],
                        self.nms_iou
                    )
                    max_detections = detections_class[keep]
                else:
                    max_detections  = detections_class
                
                output[i] = max_detections if output[i] is None else np.concatenate((output[i], max_detections))

            if output[i] is not None:
                box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4]    = self.centernet_correct_boxes(box_xy, box_wh, image_shape[i])
        return output
    
    def decode_bbox_v1(self, pred_hms, pred_whs, pred_offsets):
        #-------------------------------------------------------------------------#
        #   当利用512x512x3图片进行coco数据集预测的时候
        #   h = w = 128 num_classes = 80
        #   Hot map热力图 -> b, 80, 128, 128, 
        #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
        #   找出一定区域内，得分最大的特征点。
        #-------------------------------------------------------------------------#
        pred_hms = self.pool_nms(pred_hms)

        b, c, output_h, output_w = pred_hms.shape
        logging.info('pred_hms shape {}'.format(pred_hms.shape))
        detects = []
        #-------------------------------------------------------------------------#
        #   只传入一张图片，循环只进行一次
        #-------------------------------------------------------------------------#
        for batch in range(b):
            #-------------------------------------------------------------------------#
            #   heat_map        128*128, num_classes    热力图
            #   pred_wh         128*128, 2              特征点的预测宽高
            #   pred_offset     128*128, 2              特征点的xy轴偏移情况
            #-------------------------------------------------------------------------#
            heat_map    = pred_hms[batch].permute(1, 2, 0).view([-1, c])
            pred_wh     = pred_whs[batch].permute(1, 2, 0).view([-1, 2])
            pred_offset = pred_offsets[batch].permute(1, 2, 0).view([-1, 2])

            logging.info('heat_map shape {}, {}'.format(heat_map.shape, heat_map[0]))
            logging.info('pred_wh shape {}, {}'.format(pred_wh.shape, pred_wh[0])) 
            logging.info('pred_offset shape {}, {}'.format(pred_offset.shape, pred_offset[0]))
            #exit(0)

            yv, xv      = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
            #-------------------------------------------------------------------------#
            #   xv              128*128,    特征点的x轴坐标
            #   yv              128*128,    特征点的y轴坐标
            #-------------------------------------------------------------------------#
            xv, yv      = xv.flatten().float(), yv.flatten().float()
            #-------------------------------------------------------------------------#
            #   class_conf      128*128,    特征点的种类置信度
            #   class_pred      128*128,    特征点的种类
            #-------------------------------------------------------------------------#
            class_conf, class_pred  = torch.max(heat_map, dim = -1)
            logging.info('class_conf shape {}, {}'.format(class_conf.shape, class_conf))
            logging.info('class_pred shape {}, {}'.format(class_pred.shape, class_pred))
            mask                    = class_conf > self.confidence

            #-----------------------------------------#
            #   取出得分筛选后对应的结果
            #-----------------------------------------#
            pred_wh_mask        = pred_wh[mask]
            pred_offset_mask    = pred_offset[mask]
            if len(pred_wh_mask) == 0:
                detects.append([])
                continue     

            #----------------------------------------#
            #   计算调整后预测框的中心
            #----------------------------------------#
            logging.info(xv)
            logging.info(pred_offset_mask)
            logging.info(pred_wh_mask)
            xv_mask = torch.unsqueeze(xv[mask] + pred_offset_mask[..., 0], -1)
            yv_mask = torch.unsqueeze(yv[mask] + pred_offset_mask[..., 1], -1)
            #----------------------------------------#
            #   计算预测框的宽高
            #----------------------------------------#
            half_w, half_h = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2
            #----------------------------------------#
            #   获得预测框的左上角和右下角
            #----------------------------------------#
            bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)
            bboxes[:, [0, 2]] /= output_w
            bboxes[:, [1, 3]] /= output_h
            detect = torch.cat([bboxes, torch.unsqueeze(class_conf[mask],-1), torch.unsqueeze(class_pred[mask],-1).float()], dim=-1)
            detects.append(detect)
        return detects
    
    
    def decode_bbox(self, pred_hms, pred_whs, pred_offsets):
        #-------------------------------------------------------------------------#
        #   当利用512x512x3图片进行coco数据集预测的时候
        #   h = w = 128 num_classes = 80
        #   Hot map热力图 -> b, 80, 128, 128, 
        #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
        #   找出一定区域内，得分最大的特征点。
        #-------------------------------------------------------------------------#

        # 这边用numpy实现的maxpool耗时>100ms, 所以先注释
        # pred_hms    = self.pool_nms_v2(pred_hms)
        pred_hms    = self.pool_nms(torch.from_numpy(pred_hms)).numpy()
        
        b, c, output_h, output_w = pred_hms.shape
        logging.info('pred_hms shape {}'.format(pred_hms.shape))
        detects = []
        #-------------------------------------------------------------------------#
        #   只传入一张图片，循环只进行一次
        #-------------------------------------------------------------------------#
        for batch in range(b):
            #-------------------------------------------------------------------------#
            #   heat_map        128*128, num_classes    热力图
            #   pred_wh         128*128, 2              特征点的预测宽高
            #   pred_offset     128*128, 2              特征点的xy轴偏移情况
            #-------------------------------------------------------------------------#
            heat_map    = np.transpose(pred_hms[batch],     (1, 2, 0)).reshape((-1, c))
            pred_wh     = np.transpose(pred_whs[batch],     (1, 2, 0)).reshape((-1, 2))
            pred_offset = np.transpose(pred_offsets[batch], (1, 2, 0)).reshape((-1, 2))

            logging.info('heat_map shape {}'.format(heat_map.shape))
            logging.info('pred_wh shape {}'.format(pred_wh.shape)) 
            logging.info('pred_offset shape {}'.format(pred_offset.shape))
            #exit(0)

            # np.meshgrid和torch.meshgrid结果的维度排列不同
            yv, xv      = np.meshgrid(np.arange(0, output_h), np.arange(0, output_w))
            yv          = np.transpose(yv, (1, 0))
            xv          = np.transpose(xv, (1, 0))
            #-------------------------------------------------------------------------#
            #   xv              128*128,    特征点的x轴坐标
            #   yv              128*128,    特征点的y轴坐标
            #-------------------------------------------------------------------------#
            xv, yv      = xv.flatten().astype(np.float32), yv.flatten().astype(np.float32)
            #-------------------------------------------------------------------------#
            #   class_conf      128*128,    特征点的种类置信度
            #   class_pred      128*128,    特征点的种类
            #-------------------------------------------------------------------------#
            class_conf  = np.max(heat_map,    axis=-1)
            class_pred  = np.argmax(heat_map, axis=-1)
            logging.info('class_conf shape {}'.format(class_conf.shape))
            logging.info('class_pred shape {}'.format(class_pred.shape))
            mask                    = class_conf > self.confidence

            #-----------------------------------------#
            #   取出得分筛选后对应的结果
            #-----------------------------------------#
            pred_wh_mask        = pred_wh[mask]
            pred_offset_mask    = pred_offset[mask]
            if len(pred_wh_mask) == 0:
                detects.append([])
                continue     

            #----------------------------------------#
            #   计算调整后预测框的中心
            #----------------------------------------#
            xv_mask = np.expand_dims(xv[mask] + pred_offset_mask[..., 0], axis=-1)
            yv_mask = np.expand_dims(yv[mask] + pred_offset_mask[..., 1], axis=-1)
            #----------------------------------------#
            #   计算预测框的宽高
            #----------------------------------------#
            half_w, half_h = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2
            #----------------------------------------#
            #   获得预测框的左上角和右下角
            #----------------------------------------#
            bboxes = np.concatenate([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], 1)
            bboxes[:, [0, 2]] /= output_w
            bboxes[:, [1, 3]] /= output_h
            detect = np.concatenate(
                [bboxes, np.expand_dims(class_conf[mask], axis=-1), np.expand_dims(class_pred[mask], axis=-1).astype(np.float32)], 
                -1
            )
            logging.info(detect[..., -2])
            detects.append(detect)
        return detects
          
    #---------------------------------------------------------#
    #   将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    #---------------------------------------------------------#
    def cvt_color(self, image):
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return image 
        else:
            image = image.convert('RGB')
            return image 
      
    #---------------------------------------------------#
    #   对输入图像进行resize
    #---------------------------------------------------#
    def resize_image(self, image, size):
        iw, ih  = image.size
        w, h    = size
        if self.letterbox_image:
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image   = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        else:
            new_image = image.resize((w, h), Image.BICUBIC)
        return new_image



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo for CenterNet")
    parser.add_argument('--input',  default=os.path.join(DATA_DIR, 'ctdet_test.jpg'), required=False)
    parser.add_argument('--loops',  default=1,  type=int, required=False)
    parser.add_argument('--tpu_id', default=0,  type=int, required=False)
    parser.add_argument('--bmodel', default='', type=str, required=False)

    opt = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s%(msecs)03d[%(levelname)s][%(module)s:%(lineno)s] - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S.')
    logging.info('Start centernet detector sail demo.')

    if opt.bmodel:
        Detector._defaults['bmodel_path'] = opt.bmodel
        
    # Initialize centernet detector instance
    cet_detector      = Detector(tpu_id=opt.tpu_id)
    input_w, input_h  = cet_detector.get_net_size()
    
    batch_size = cet_detector.get_batchsize()
    logging.info("Input model batch size is {}".format(batch_size))
        
    input_path        = opt.input
    decoder           = sail.Decoder(input_path, True, opt.tpu_id)
    process_handle    = cet_detector.get_handle()
    
    for idx in range(opt.loops):
        logging.info('loop start')
        image_ost_list = []

        if batch_size == 1:
            img = sail.BMImage()
            ret = decoder.read(process_handle, img)
            if ret != 0:
                logging.warning('decoder read image failed!')
                continue
            dst_img = cet_detector.bmcv.convert_format(img)
            image_ost_list.append(dst_img)         
            logging.info('input format {}'.format(dst_img.format()))
            results = cet_detector.predict(dst_img)
            
        elif batch_size == 4:
            img = sail.BMImageArray4D()
            for i in range(4):
                rgb_img = cet_detector.bmcv.convert_format(decoder.read(process_handle))
                image_ost_list.append(rgb_img)
                img[i] = rgb_img.data()
            results = cet_detector.predict(img)
            
        else:
            raise NotImplementedError(
                'This demo not supports inference with batch size {}'.format(cet_detector.get_batchsize()))
        
        for b in range(len(results)):
            if results[b] is None:
                logging.info('batch {} detect nothing'.format(b + 1))
                continue
            top_label   = np.array(results[b][:, 5], dtype='int32')
            top_conf    = results[b][:, 4]
            top_boxes   = results[b][:, :4]
            
            #---------------------------------------------------------#
            #   图像绘制
            #---------------------------------------------------------#
            for i, c in list(enumerate(top_label)):
                predicted_class = cet_detector.class_names[int(c)]
                box             = top_boxes[i]
                score           = top_conf[i]

                top, left, bottom, right = box

                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image_ost_list[b].height(), np.floor(bottom).astype('int32'))
                right   = min(image_ost_list[b].width(),  np.floor(right).astype('int32'))

                logging.info('[object]:{} -> label {}, top {}, left {}, bottom {}, right {}'.format(score, predicted_class, top, left, bottom, right))
                cet_detector.bmcv.rectangle(image_ost_list[b], left, top, right - left, bottom - top, (255, 0, 0), 3)

            # draw result
            det_filename = 'ctdet_result_{}_b_{}.jpg'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), b)
            cet_detector.bmcv.imwrite(det_filename, image_ost_list[b])
            logging.info('Prediction result: {}'.format(det_filename))
        
    # exit
    logging.info('Demo exit..')
