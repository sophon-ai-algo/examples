""" Copyright 2016-2022 by Bitmain Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import division
import sys
import os
import argparse
import json
import cv2
import numpy as np
import sophon.sail as sail
import time
import datetime


def preprocess(img, new_shape):
  """ Preprocessing of a frame.

  Args:
    image : np.array, input image
    detected_size : list, yolov3 detection input size

  Returns:
    Preprocessed data.
  """
  #img_h, img_w = image.shape[0], image.shape[1]
  #net_h, net_w = detected_size[0], detected_size[1]
  #new_w = int(img_w * min(net_w / img_w, net_h / img_h))
  #new_h = int(img_h * min(net_w / img_w, net_h / img_h))
  #resized_image = cv2.resize(image, (new_w, new_h), \
  #                           interpolation=cv2.INTER_LINEAR)
  
  # canvas
  #canvas = np.full((net_w, net_h, 3), 114)
  #canvas[(net_h - new_h) // 2:(net_h - new_h) // 2 + new_h,\
  #  (net_w - new_w) // 2:(net_w - new_w) // 2 + new_w, :]\
  #  = resized_image
  #return canvas[:, :, ::-1].transpose([2, 0, 1]) / 255.0
  shape = img.shape[:2]
  if isinstance(new_shape, int):
      new_shape = (new_shape, new_shape)

  ######
  #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
  ratio = r, r
  new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
  dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
  dw, dh = dw / 2, dh / 2
  if shape[::-1] != new_unpad:
      img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
  top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
  left, right = int(round(dw-0.1)), int(round(dw+0.1))
  img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
  #img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
  #return img[:, :, ::-1].transpose([2, 0, 1]) / 255, ratio[0], (top, left)
  return img[:, :, ::-1].transpose([2, 0, 1]) / 255 , ratio[0], (top, left)

CONF_THRESH = 0.0
def post_process(outputs, origin_h, origin_w):
  """
  description: postprocess the prediction
  param:
      output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
      origin_h:   height of original image
      origin_w:   width of original image
  return:
      result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
      result_scores: finally scores, a tensor, each element is the score correspoing to box
      result_classid: finally classid, a tensor, each element is the classid correspoing to box
  """
  
  for key, output in outputs.items():
    output = np.squeeze(output)
    # Get the boxes
    boxes = output[:, :, :, :4]
    # Get the scores
    scores = output[:, :, :, 4]
    # Get the classid
    classid = output[:, :, :, 5]


    # Choose those boxes that score > CONF_THRESH
    # si = scores >= CONF_THRESH
    # boxes = boxes[si, :]
    # scores = scores[si]
    # classid = classid[si]

    x1 = boxes[:, :, :, 0]
    y1 = boxes[:, :, :, 1]
    x2 = boxes[:, :, :, 2]
    y2 = boxes[:, :, :, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
    indices = 0
    # boxes = xywh2xyxy(origin_h, origin_w, boxes)
    # Do nms
    while order.size > 0:
      i = order[0]
      # temp.append(i)
      #计算当前概率最大矩形框与其他矩形框的相交框的坐标
      # 由于numpy的broadcast机制，得到的是向量
      xx1 = np.maximum(x1[i], x1[:,order[1:]])
      yy1 = np.minimum(y1[i], y1[:,order[1:]])
      xx2 = np.minimum(x2[i], x2[:,order[1:]])
      yy2 = np.maximum(y2[i], y2[:,order[1:]])

      #计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，需要用0代替
      w = np.maximum(0.0, xx2 - xx1 + 1)
      h = np.maximum(0.0, yy2 - yy1 + 1)
      inter = w * h
      #计算重叠度IoU
      ovr = inter / (areas[i] + areas[order[1:]] - inter)

      #找到重叠度不高于阈值的矩形框索引
      indices = np.where(ovr <= CONF_THRESH)[0]
      #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
      order = order[indices + 1]

    result_boxes = boxes[indices, :]
    result_scores = scores[indices]
    result_classid = classid[indices]
  return result_boxes, result_scores, result_classid

def get_reference(compare_path):
  """ Get correct result from given file.
  Args:
    compare_path: Path to correct result file

  Returns:
    Correct result.
  """
  if compare_path:
    with open(compare_path, 'r') as f:
      reference = json.load(f)
      return reference
  return None

def compare(reference, bboxes, classes, probs, loop_id):
  """ Compare result.
  Args:
    reference: Correct result
    result: Output result
    loop_id: Loop iterator number

  Returns:
    True for success and False for failure
  """
  if not reference:
    print("No verify_files file or verify_files err.")
    return True
  if loop_id > 0:
    return True
  detected_num = len(classes)
  reference_num = len(reference["category"])
  if (detected_num != reference_num):
    message = "Expected deteted number is {}, but detected {}!"
    print(message.format(reference_num, detected_num))
    return False
  ret = True
  scores = ["{:.8f}".format(p) for p in probs]
  message = "Category: {}, Score: {}, Box: {}"
  fail_info = "Compare failed! Expect: " + message
  ret_info = "Result Box: " + message
  for i in range(detected_num):
    if classes[i] != reference["category"][i] or \
        scores[i] != reference["score"][i] or \
        bboxes[i] != reference["box"][i]:
      print(fail_info.format(reference["category"][i], reference["score"][i], \
                             reference["box"][i]))
      print(ret_info.format(classes[i], scores[i], bboxes[i]))
      ret = False
  return ret




import time
import numpy as np


ANCHORS = np.array([
    [10,  13, 16,  30,  33,  23 ], 
    [30,  61, 62,  45,  59,  119], 
    [116, 90, 156, 198, 373, 326]
])
ANCHOR_GRID = ANCHORS.reshape(3, -1, 2).reshape(3, 1, -1, 1, 1, 2)
STRIDES = [8, 16, 32]
CONF_THR = 0.6
IOU_THR = 0.5


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def make_grid(nx, ny):
    z = np.stack(np.meshgrid(np.arange(nx), np.arange(ny)), 2)
    return z.reshape(1, 1, ny, nx, 2).astype(np.float32)


def predict_preprocess(x):
    for i in range(len(x)):
        bs, na, ny, nx, no = x[i].shape
        grid = make_grid(nx, ny)
        x[i] = sigmoid(x[i])
        x[i][..., 0:2] = (x[i][..., 0:2] * 2. - 0.5 + grid) * STRIDES[i]
        x[i][..., 2:4] = (x[i][..., 2:4] * 2) ** 2 * ANCHOR_GRID[i]
        x[i] = x[i].reshape(bs, -1, no)
    return np.concatenate(x, 1)

def _nms(dets, scores, prob_threshold):
    #import pdb; pdb.set_trace()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
 
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    score_index = np.argsort(scores)[::-1]
 
    keep = []
 
    while score_index.size > 0:
        max_index = score_index[0]
        # 最大的肯定是需要的框
        keep.append(max_index)
        xx1 = np.maximum(x1[max_index], x1[score_index[1:]])
        yy1 = np.maximum(y1[max_index], y1[score_index[1:]])
        xx2 = np.minimum(x2[max_index], x2[score_index[1:]])
        yy2 = np.minimum(y2[max_index], y2[score_index[1:]])
 
        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)
 
        union = width * height
 
        iou = union / (areas[max_index] + areas[score_index[1:]] - union)
        ids = np.where(iou < prob_threshold)[0]
        # 以为算iou的时候没把第一个参考框索引考虑进来，所以这里都要+1
        score_index = score_index[ids+1]
    return keep

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.3, classes=None):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    prediction = [prediction['147'], prediction['148'], prediction['149']]
    prediction = predict_preprocess(prediction)
    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            # i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            i, j = np.stack((x[:, 5:] > conf_thres).nonzero())
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].astype(np.float32)), 1)
        else:  # best class only
            # conf, j = x[:, 5:].max(1, keepdim=True)
            conf = x[:, 5:].max(1, keepdims=True)
            j = x[:, 5:].argmax(1).reshape(-1, 1)
            x = np.concatenate((box, conf, j.astype(np.float32)), 1)[conf.reshape(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * max_wh # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = _nms(boxes, scores, iou_thres)
        if len(i) > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output



def inference(net, input_path, loops, tpu_id):
  """ Load a bmodel and do inference.
  Args:
   bmodel_path: Path to bmodel
   input_path: Path to input file
   loops: Number of loops to run
   tpu_id: ID of TPU to use
   compare_path: Path to correct result file

  Returns:
    True for success and False for failure
  """
  # set configurations
  load_from_file = True
  detected_size = (640, 640)
  threshold = 0.001
  nms_threshold = 0.6
  num_classes = 2
  cap = cv2.VideoCapture(input_path)
  # init Engine and load bmodel
  # get model info
  
  graph_name = net.get_graph_names()[0]
  input_name = net.get_input_names(graph_name)[0]

  #reference = get_reference(compare_path)
  status = True
  # pipeline of inference
  for i in range(loops):
    # read an image
    # ret, img = cap.read()
    img = cv2.imread(input_path)
    # if not ret:
    #   print("Finished to read the video!");
    #   break
    # preprocess
    # cv2.imwrite("/home/linaro/tmp/pic/mask3.jpg", img)
    t1=time.time()
    data, ratio, (top, left) = preprocess(img, detected_size)
    t2=time.time()
    print("preprocess cost : %.3f second" % (t2 - t1))
    
    input_data = {input_name: np.array([data], dtype=np.float32)}
    output = net.process(graph_name, input_data)
    
    '''
    output={}
    net.fill_blob_data({net.inputs[0]:data})
    net.forward()
    output['147']=net.get_blob_data('147')
    output['148']=net.get_blob_data('148')
    output['149']=net.get_blob_data('149')
    '''
    t1=time.time()
    print("infer cost : %.3f second" % (t1 - t2))
    #print(output)
    #print('infer time is {}'.format(t2-t1))
    # postprocess
    # bboxes, classes, probs = postprocess(output, img, detected_size, threshold)
    # post_process(output, img.shape[0], img.shape[1])
    #import pdb; pdb.set_trace()
    prediction = non_max_suppression(output, conf_thres=threshold, iou_thres=nms_threshold, classes=None)
    t2=time.time()
    print("nms cost : %.3f second" % (t2 - t1))
    # print result
    #print(prediction)

    # if compare(reference, bboxes, classes, probs, i):
    #   for bbox, cls, prob in zip(bboxes, classes, probs):
    #     message = "[Frame {} on tpu {}] Category: {}, Score: {:.3f}, Box: {}"
    #     print(message.format(i + 1, tpu_id, cls, prob, bbox))
    # else:
    #   status = False
    #   break
    #  bbox = prediction[0][0][:4]
    # x2 = (bbox[2]-bbox[0])*2+bbox[0]
    # y2 = (bbox[3]-bbox[1])*2+bbox[1]
    # cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (0,255,0), 4)
    for i in prediction:
      if i is None:
        continue
      for j in i:
        bbox = j[:4]
        bbox[0::2] -= left
        bbox[1::2] -= top
        bbox /= ratio
        #import pdb; pdb.set_trace()
        #cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0,255,0), 4)
    # cv2.rectangle(img, (int(bbox[0]-bbox[2]/2), int(bbox[1]-bbox[3]/2)), (int(bbox[0]+bbox[2]/2),int(bbox[1]+bbox[3]/2)), (0,255,0), 4)
    #cv2.imwrite("./preds/{}".format(input_path.split('/')[-1]), img)
      
  cap.release()
  return prediction

if __name__ == '__main__':
  """ A YOLOv5 example.
  """
  PARSER = argparse.ArgumentParser(description='for sail det_yolov5 py test')
  PARSER.add_argument('--bmodel', default='', required=True)
  PARSER.add_argument('--imgdir', default='', required=True)
  PARSER.add_argument('--tpu_id', default=0, type=int, required=False)
  PARSER.add_argument('--input', default='', required=True)
  PARSER.add_argument('--result', default='', required=True)
  ARGS = PARSER.parse_args()
  if not os.path.isfile(ARGS.input):
    print("Error: {} not exists!".format(ARGS.input))
    sys.exit(-2)
  import os
  import json
  with open(ARGS.input) as g:
      js = json.load(g)
  preds = []
  processed=0
  net = sail.Engine(ARGS.bmodel, ARGS.tpu_id, sail.IOMode.SYSIO)
  if not os.path.isfile(ARGS.bmodel):
    print('please input bmodel')
    sys.exit(-2)
  #net=ufw.Net(ARGS.model, ARGS.weight,ufw.TEST)
  print("imgdir: ", ARGS.imgdir)

  for img in js['images']:
    img_p = img['file_name']
    #print(img_p)
    if not os.path.isfile('/'.join((ARGS.imgdir,img_p))):
       continue
    processed=processed+1
    print('processing {}'.format(processed))
    pred = inference(net, os.path.join(ARGS.imgdir, img_p), 1, ARGS.tpu_id)
    coco91class = coco80_to_coco91_class()
    for pp in pred:
      if pp is None:
        continue
      for p in pp:
        bbox = p[:4]
        prob = float(p[4])
        clse = int(p[5])
        preds.append(dict(
          image_id=img['id'],
          category_id=coco91class[clse],
          bbox=[float(bbox[0]), float(bbox[1]), float(bbox[2]-bbox[0]), float(bbox[3]-bbox[1])],
          score=prob))

  with open(ARGS.result, 'w') as f:
      json.dump(preds, f)
    #json.dump(preds, f, indent=4)
                  
  #sys.exit(0 if status else -1)
