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

class PreProcessor:
  """ Preprocessing class.
  """
  def __init__(self, bmcv, size_w, size_h, scale):
    """ Constructor.
    """
    self.bmcv = bmcv
    self.size_w = size_w
    self.size_h = size_h
    # self.ab = [x * scale for x in [1, -123, 1, -117, 1, -104]]
    self.ab = [x * scale for x in [1, 0, 1, 0, 1, 0]]

  def process(self, input, output):
    """ Execution function of preprocessing.
    Args:
      cv_input: sail.BMImage, input image
      bmcv_output: sail.BMImage, output data

    Returns:
      None
    """
    tmp = self.bmcv.vpp_resize(input, self.size_w, self.size_h)
    self.bmcv.convert_to(tmp, output, ((self.ab[0], self.ab[1]), \
                                       (self.ab[2], self.ab[3]), \
                                       (self.ab[4], self.ab[5])))

class Detector(object):
    """
    This is CenterNet detector class
    """
    _defaults = {
        # bmodel模型文件
        "bmodel_path"       : '/workspace/examples/centernet_test/ctdet_coco_dlav0_1x_fp32.bmodel',
        # 类标文件
        "classes_path"      : '/workspace/examples/centernet_test/CenterNet_object/data/coco_classes.txt',
        # 字体文件
        "font_path"         : '/workspace/examples/centernet_test/CenterNet_object/data/simhei.ttf',
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
        "nms"               : False,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
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
        self.engine         = sail.Engine(self.bmodel_path, tpu_id, sail.IOMode.SYSIO)
        self.graph_name     = self.engine.get_graph_names()[0]
        self.input_name     = self.engine.get_input_names(self.graph_name)[0]
        self.output_name    = self.engine.get_output_names(self.graph_name)[0]
        self.input_dtype    = self.engine.get_input_dtype(self.graph_name, self.input_name)
        self.output_dtype   = self.engine.get_output_dtype(self.graph_name, self.output_name)
        self.input_shape    = self.engine.get_input_shape(self.graph_name, self.input_name)
        self.input_w        = int(self.input_shape[-1])
        self.input_h        = int(self.input_shape[-2])
        self.output_shape   = self.engine.get_output_shape(self.graph_name, self.output_name)
        logging.info("\n" + "*" * 50 + "\n"
                     "graph_name:   {}\n"
                     "input_name:   {}\n"
                     "output_name:  {}\n"
                     "input_dtype:  {}\n"
                     "output_dtype: {}\n"
                     "input_shape:  {}\n"
                     "output_shape: {}\n".format(self.graph_name, self.input_name, self.output_name,
                                                 self.input_dtype, self.output_dtype, self.input_shape, self.output_shape)
                                                 + "*" * 50)
        # self.handle         = self.engine.get_handle()
        # self.input          = sail.Tensor(self.handle, self.input_shape, self.input_dtype, 
        #                                   False, False)
        # self.output         = sail.Tensor(self.handle, self.output_shape, self.output_dtype, 
        #                                   True, True)
        # self.input_tensors  = { self.input_name  : self.input }
        # self.output_tensors = { self.output_name : self.output}

        # self.bmcv = sail.Bmcv(self.handle)
        # self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        # self.scale = self.engine.get_input_scale(self.graph_name, self.input_name)

        # logging.info("self.img_dtype: {}".format(self.img_dtype))
        # self.input_bmimage = sail.BMImage(self.handle, 
        #                                   self.input_w, self.input_h,
        #                                   sail.Format.FORMAT_BGR_PLANAR, 
        #                                   self.img_dtype)

        # self.preprocessor = PreProcessor(self.bmcv, self.input_w, self.input_h, self.scale)

    #---------------------------------------------------#
    #   获得类
    #---------------------------------------------------#
    def get_classes(self):
        with open(self.classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names, len(class_names)
  
    def predict(self, cv_img):
        #---------------------------------------------------#
        #   计算输入图片的高和宽 HxW
        #---------------------------------------------------#
        image_shape = np.array(np.shape(cv_img)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        cv_img      = self.cvt_color(cv_img)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = self.resize_image(cv_img, (self.input_w, self.input_h))
        #-----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        #-----------------------------------------------------------#
        image_data  = np.expand_dims(
            np.transpose(self.preprocess(np.array(image_data, dtype='float32')), (2, 0, 1)), 
            axis=0
        )
        # Convert the image to row-major order, also known as "C order":
        image_data  = np.ascontiguousarray(image_data)

        input_data  = { self.input_name : np.array(image_data, dtype=np.float32)}
        logging.info('debug input {}'.format(input_data[self.input_name][0][0]))

        # 推理
        start       = time.time()
        output      = self.engine.process(self.graph_name, input_data)
        logging.info('inference time {}ms'.format((time.time() - start) * 1000))
        
        dets        = output[self.output_name]
        logging.info('inference finish. dets shape -> {}'.format(dets.shape))
        pred_hms = dets[:,:80,...]
        pred_whs = dets[:,80:82,...]
        pred_off = dets[:,82:84,...]
        logging.info('debug  dets -> {}'.format(dets[0][0]))

        
        # 解码
        outputs = self.decode_bbox(torch.from_numpy(pred_hms).sigmoid(), 
                                   torch.from_numpy(pred_whs), 
                                   torch.from_numpy(pred_off))

        #-------------------------------------------------------#
        #   对于centernet网络来讲，确立中心非常重要。
        #   对于大目标而言，会存在许多的局部信息。
        #   此时对于同一个大目标，中心点比较难以确定。
        #   使用最大池化的非极大抑制方法无法去除局部框
        #   所以我还是写了另外一段对框进行非极大抑制的代码
        #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大
        results = self.postprocess(outputs, image_shape)
        
        #--------------------------------------#
        #   如果没有检测到物体，则返回原图
        #--------------------------------------#
        if results[0] is None:
            return cv_img
        
        top_label   = np.array(results[0][:, 5], dtype='int32')
        top_conf    = results[0][:, 4]
        top_boxes   = results[0][:, :4]
        
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font = ImageFont.truetype(font=self.font_path, 
                                  size=np.floor(3e-2 * np.shape(cv_img)[1] + 0.5).astype('int32'))
        logging.info('np.shape(cv_img) {}, self.input_shape {}'.format(np.shape(cv_img), self.input_shape))
        thickness = max((np.shape(cv_img)[0] + np.shape(cv_img)[1]) // self.input_shape[-1], 1)

        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(cv_img.size[1], np.floor(bottom).astype('int32'))
            right   = min(cv_img.size[0], np.floor(right).astype('int32'))
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(cv_img)

            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            logging.info(
              '[object] -> label {}, top {}, left {}, bottom {}, right {}'.format(label, top, left, bottom, right)
            )
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            #draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            #draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw


        return cv_img
        
        
    def preprocess(self, image):
        image   = np.array(image, dtype = np.float32)[:, :, ::-1]
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
                logging.info(detections)
                continue
            #------------------------------------------#
            #   获得预测结果中包含的所有种类
            #------------------------------------------#
            unique_labels   = detections[:, -1].cpu().unique()
            logging.info(unique_labels)
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
                        detections_class[:, :4],
                        detections_class[:, 4],
                        self.nms_iou
                    )
                    max_detections = detections_class[keep]

                    # #------------------------------------------#
                    # #   按照存在物体的置信度排序
                    # #------------------------------------------#
                    # _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
                    # detections_class = detections_class[conf_sort_index]
                    # #------------------------------------------#
                    # #   进行非极大抑制
                    # #------------------------------------------#
                    # max_detections = []
                    # while detections_class.size(0):
                    #     #---------------------------------------------------#
                    #     #   取出这一类置信度最高的，一步一步往下判断。
                    #     #   判断重合程度是否大于nms_thres，如果是则去除掉
                    #     #---------------------------------------------------#
                    #     max_detections.append(detections_class[0].unsqueeze(0))
                    #     if len(detections_class) == 1:
                    #         break
                    #     ious = bbox_iou(max_detections[-1], detections_class[1:])
                    #     detections_class = detections_class[1:][ious < nms_thres]
                    # #------------------------------------------#
                    # #   堆叠
                    # #------------------------------------------#
                    # max_detections = torch.cat(max_detections).data
                else:
                    max_detections  = detections_class
                
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

            if output[i] is not None:
                output[i]           = output[i].cpu().numpy()
                box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4]    = self.centernet_correct_boxes(box_xy, box_wh, image_shape)
        return output
  
    def decode_bbox(self, pred_hms, pred_whs, pred_offsets):
        #-------------------------------------------------------------------------#
        #   当利用512x512x3图片进行coco数据集预测的时候
        #   h = w = 128 num_classes = 80
        #   Hot map热力图 -> b, 80, 128, 128, 
        #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
        #   找出一定区域内，得分最大的特征点。
        #-------------------------------------------------------------------------#
        logging.info('debug pred_hms {}'.format(pred_hms[0][0]))
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
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s[%(levelname)s][%(module)s:%(lineno)s] - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('start centernet detector sail demo')

    # Initialize centernet detector instance
    cet_detector = Detector(tpu_id=0)
    
    # open an image to predict
    image = Image.open('/workspace/examples/centernet_test/CenterNet_object/data/ctdet_test.jpg')
    
    # do prediction
    det_image = cet_detector.predict(image)
    
    # draw result
    det_filename = 'ctdet_result_{}.jpg'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    det_image.save(det_filename, quality=95)
    logging.info('prediction result: {}'.format(det_filename))
    
    # exit
    logging.info('demo exit..')
