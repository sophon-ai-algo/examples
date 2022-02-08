from queue import Empty
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
        self.input_4dbm = sail.BMImageArray4D(self.handle, self.input_h, self.input_w, \
                                 sail.Format.FORMAT_BGR_PLANAR, self.img_dtype)

    def get_batchsize(self):
        return int(self.input_shape[0])

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

    def predict(self,bm4d_img):
        self.bmcv.convert_to(bm4d_img,  self.input_4dbm, ((self.scale, 0),(self.scale, 0),(self.scale, 0)))
        self.bmcv.bm_image_to_tensor(self.input_4dbm, self.input)
        self.engine.process(self.graph_name, self.input_tensors, self.output_tensors)
        out_temp = self.output.asnumpy()
        predictions = self.yolox_postprocess(out_temp, self.input_w, self.input_w)
        return predictions

    def get_detectresult(self,predictions,dete_threshold,nms_threshold):
        # print(predictions.shape)
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        # print("boxes")
        # print(boxes.shape)
        # print(boxes)
        # print("scores")
        # print(scores.shape)
        # print(scores)

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.

        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_threshold, score_thr=dete_threshold)
        return dets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo for YOLOX")
    parser.add_argument('--bmodel',default="../data/models/yolox_s_4_int8.bmodel",required=False)
    parser.add_argument('--tpu_id',default=0,type=int,required=False)
    parser.add_argument('--input',default='/workspace/mytest/data/video/zhuheqiao.mp4',required=False)
    parser.add_argument('--loops',default=1,type=int,required=False)
    parser.add_argument('--detect_threshold',default=0.25,required=False)
    parser.add_argument('--nms_threshold',default=0.45,required=False)
    opt = parser.parse_args()

    yolox= Detector(bmodel_path=opt.bmodel, tpu_id=opt.tpu_id)
    input_w = yolox.input_w
    input_h = yolox.input_h

    if yolox.get_batchsize() != 4:
        print("Input model batch size in not 4!")
        exit(1)
    
    video_path = opt.input
    decoder =  sail.Decoder(video_path, True, opt.tpu_id)
    mkdir("result")
    for idx in range(opt.loops):
        tmp_img = sail.BMImageArray4D()
        image_ost_list = []
        image_resize_list = []

        img_ost = sail.BMImage()
        for i in range(4):
            img_ost = sail.BMImage()
            img_resize = sail.BMImage()
            for temp_idx in range(25):
                ret = decoder.read(yolox.handle, img_ost)
                if ret != 0:
                    exit("Read the end!")
            img_resize = yolox.bmcv.vpp_resize(img_ost, input_w, input_h)
            image_ost_list.append(img_ost)
            image_resize_list.append(img_resize)
            tmp_img[i] = img_resize.data()
            yolox.bmcv.imwrite('result/loop-{}-batch-{}-dev-{}-input.jpg'.format(idx,i,opt.tpu_id), img_ost)
        
        ratio_w = float(image_ost_list[0].width())/float(input_w)
        ratio_h = float(image_ost_list[0].height())/float(input_h)

        numpy_out = yolox.predict(tmp_img)

        for image_idx, image_ost in enumerate(image_ost_list):
            dete_boxs = yolox.get_detectresult(numpy_out[image_idx],opt.detect_threshold, opt.nms_threshold)
            if dete_boxs is not None:
                dete_boxs[:,0] *= ratio_w
                dete_boxs[:,1] *= ratio_h
                dete_boxs[:,2] *= ratio_w
                dete_boxs[:,3] *= ratio_h
                for dete_box in dete_boxs:
                    yolox.bmcv.rectangle(image_ost, int(dete_box[0]), int(dete_box[1]), 
                        int(dete_box[2]-dete_box[0]), int(dete_box[3]-dete_box[1]), (255, 0, 0), 3)
                yolox.bmcv.imwrite('result/loop-{}-batch-{}-dev-{}-video.jpg'.format(idx,image_idx,opt.tpu_id), image_ost)

    print("end.........")    
        