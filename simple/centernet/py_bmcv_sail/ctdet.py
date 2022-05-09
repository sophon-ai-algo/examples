from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import logging

from base_detector import BaseDetector
from numpy.lib.stride_tricks import as_strided

class CtdetDetector(BaseDetector):
  def __init__(self, **args):
    super(CtdetDetector, self).__init__(**args)
  
  def process(self):
    self.engine.process(self.graph_name, self.input_tensors, self.output_tensors)
        
    dets     = self.output.asnumpy().astype(np.float32)
    dets    *= self.output_scale
    
    logging.info('inference finish. dets shape -> {}'.format(dets.shape))
    pred_hms = dets[:, :self.output_shape[1] - 4, ...]
    pred_whs = dets[:, self.output_shape[1] - 4:self.output_shape[1] - 2, ...]
    pred_off = dets[:, self.output_shape[1] - 2:, ...]

    # sigmoid
    pred_hms = 1. / (1 + np.exp(-pred_hms))
    forward_time = time.time()
    
    return pred_hms, pred_whs, pred_off, forward_time
  
    # dets    = self.output.asnumpy().astype(np.float32)
    # dets   *= self.output_scale
    # output  = torch.from_numpy(dets)
    # hm      = output[:,:80,...].sigmoid_()
    # wh      = output[:,80:82,...]
    # reg     = output[:,82:84,...]
    
  def decode_bbox(self, pred_hms, pred_whs, pred_offsets):
      #-------------------------------------------------------------------------#
      #   当利用512x512x3图片进行coco数据集预测的时候
      #   h = w = 128 num_classes = 80
      #   Hot map热力图 -> b, 80, 128, 128, 
      #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
      #   找出一定区域内，得分最大的特征点。
      #-------------------------------------------------------------------------#

      # 这边用numpy实现的maxpool耗时>100ms, 所以先注释
      pred_hms    = self.pool_nms_v2(pred_hms)
      #pred_hms    = self.pool_nms(torch.from_numpy(pred_hms)).numpy()
      
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
          mask        = class_conf > self.confidence

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
          detects.append(detect)
      return detects
        
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
  
  def nms(self, boxes, scores, nms_thr):
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
  
  def post_process(self, prediction, image_shape):
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
                keep = self.nms(
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
  
  def merge_outputs(self, detections):
    pass

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 4] > self.opt.center_thresh:
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4], 
                                 img_id='out_pred_{:.1f}'.format(scale))

  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='ctdet')
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        if bbox[4] > self.opt.vis_thresh:
          debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
    #debugger.show_all_imgs(pause=self.pause)
    debugger.save_img(imgId='ctdet', path='./')
