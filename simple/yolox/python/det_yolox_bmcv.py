import cv2
import numpy as np
import sophon.sail as sail
import argparse
import os

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

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


def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)


def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets

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
  def __init__(self,bmodel_path,tpu_id):
      print("bmodel_path:{}".format(bmodel_path))
      print("tpu_id:{}".format(tpu_id))
      self.engine = sail.Engine(bmodel_path,tpu_id,sail.IOMode.SYSO)
      self.graph_name = self.engine.get_graph_names()[0]
      self.input_name = self.engine.get_input_names(self.graph_name)[0]
      self.output_name = self.engine.get_output_names(self.graph_name)[0]
      self.input_dtype= self.engine.get_input_dtype(self.graph_name, self.input_name)
      self.output_dtype = self.engine.get_output_dtype(self.graph_name, self.output_name)
      self.input_shape = self.engine.get_input_shape(self.graph_name, self.input_name)
      self.input_w = int(self.input_shape[-1])
      self.input_h = int(self.input_shape[-2])
      self.output_shape = self.engine.get_output_shape(self.graph_name, self.output_name)
      self.handle = self.engine.get_handle()
      self.input = sail.Tensor(self.handle,self.input_shape,self.input_dtype,False,False)
      self.output = sail.Tensor(self.handle,self.output_shape,self.output_dtype,True,True)
      self.input_tensors = {self.input_name:self.input}
      self.output_tensors = {self.output_name:self.output}

      self.bmcv = sail.Bmcv(self.handle)
      self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
      self.scale = self.engine.get_input_scale(self.graph_name, self.input_name)

      print("self.img_dtype: {}".format(self.img_dtype))
      self.input_bmimage = sail.BMImage(self.handle, self.input_w, self.input_h, \
                      sail.Format.FORMAT_BGR_PLANAR, self.img_dtype)

      self.preprocessor = PreProcessor(self.bmcv, self.input_w, self.input_h, self.scale)

      print("graph_name:{}".format(self.graph_name))
      print("input_name:{}".format(self.input_name))
      print("output_name:{}".format(self.output_name))
      print("input_dtype:{}".format(self.input_dtype))
      print("output_dtype:{}".format(self.output_dtype))
      print("input_shape:{}".format(self.input_shape))
      print("output_shape:{}".format(self.output_shape))

  def predict(self,bm_img):
      self.preprocessor.process(bm_img, self.input_bmimage)
      self.bmcv.bm_image_to_tensor(self.input_bmimage, self.input)
      # inference
      self.engine.process(self.graph_name, self.input_tensors, self.output_tensors)

      # postprocess
      out_temp = self.output.asnumpy()
      predictions = self.yolox_postprocess(out_temp, self.input_w, self.input_w)

      return predictions

  def get_detectresult(self,predictions,dete_threshold,nms_threshold):
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.

    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_threshold, score_thr=dete_threshold)
    return dets


  def get_handle(self):
    return self.handle

  def get_batchsize(self):
    return int(self.input_shape[0])

  def get_net_size(self):
    return int(self.input_w),int(self.input_h)

  def yolox_postprocess(self, outputs, input_w, input_h, p6=False):
    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [input_h // stride for stride in strides]
    wsizes = [input_w // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo for YOLOX")
    parser.add_argument('--bmodel',default="../data/models/yolox_s_1_int8.bmodel",required=False)
    parser.add_argument('--tpu_id',default=0,type=int,required=False)
    parser.add_argument('--input',default='../data/videos/zhuheqiao_crop.mp4',required=False)
    parser.add_argument('--loops',default=1,type=int,required=False)
    parser.add_argument('--detect_threshold',default=0.25,required=False)
    parser.add_argument('--nms_threshold',default=0.45,required=False)
    opt = parser.parse_args()

    yolox= Detector(bmodel_path=opt.bmodel, tpu_id=opt.tpu_id)
    input_w,input_h = yolox.get_net_size()

    if yolox.get_batchsize() != 1:
      print("Input model batch size in not 1!")
      exit(1)
    input_path = opt.input
    decoder = sail.Decoder(input_path, True, opt.tpu_id)
    process_handle = yolox.get_handle()
    mkdir("result")
    for idx in range(opt.loops):
      img0 = sail.BMImage()
      ret = decoder.read(process_handle, img0)
      ratio_w = float(img0.width())/float(input_w)
      ratio_h = float(img0.height())/float(input_h)

      numpy_out = yolox.predict(img0)
      dete_boxs = yolox.get_detectresult(numpy_out[0],opt.detect_threshold, opt.nms_threshold)
      dete_boxs[:,0] *= ratio_w
      dete_boxs[:,1] *= ratio_h
      dete_boxs[:,2] *= ratio_w
      dete_boxs[:,3] *= ratio_h
      for dete_box in dete_boxs:
        yolox.bmcv.rectangle(img0, int(dete_box[0]), int(dete_box[1]), 
          int(dete_box[2]-dete_box[0]), int(dete_box[3]-dete_box[1]), (255, 0, 0), 3)
      yolox.bmcv.imwrite('result/loop-{}-dev-{}-video.jpg'.format(idx,opt.tpu_id), img0)


