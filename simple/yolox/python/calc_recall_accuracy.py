from genericpath import exists
import numpy as np
import os
import argparse

def calc_overlap(box_0,box_1):
  if abs(box_0[-1] - box_1[-1]) > 0:
    return 0

  max_left   = max(box_0[0],box_1[0])
  max_top    = max(box_0[1],box_1[1])
  min_right  = min(box_0[2],box_1[2])
  min_bottom = min(box_0[3],box_1[3])

  if max_left >= min_right or max_top > min_bottom:
    return 0
  
  sa = (box_0[2] - box_0[0])*(box_0[3] - box_0[1])
  sb = (box_1[2] - box_1[0])*(box_1[3] - box_1[1])

  cross = (min_right - max_left)*(min_bottom - max_top)

  return cross/(sa+sb-cross)

def same_box_obj(box_0,box_1,threshold):
  if calc_overlap(box_0,box_1) > threshold:
    return True
  else:
    return False

def get_diff_num(detection_list, ground_truth_list, threshold):
  same_count = 0
  for det_box in detection_list:
    for verify_box in ground_truth_list:
      if same_box_obj(det_box,verify_box,threshold):
        same_count += 1
        break

  return same_count,len(detection_list),len(ground_truth_list)

def get_detect_from_file(file_name):
  result = {}
  if os.path.exists(file_name) is False:
    return result
  with open(file_name,"r") as fp:
    frame_name = ""
    while (fp.readable()):
      line = fp.readline()
      if line == "":
        break

      line_str = line.rstrip("\n")
      if len(line_str) <= 0:
        continue
      line_split = line_str.split("[")
      if len(line_split) <= 0:
        continue
      frame_name_temp = line_split[-1].split("]")[0]
      if frame_name_temp != frame_name:
        if frame_name != "":
          result.update({frame_name:freame_objs})
        #   result.append(freame_objs)
          # print(frame_name)
          # print(freame_objs)
        frame_name = frame_name_temp
        freame_objs = [] 

      category  = int(fp.readline().rstrip("\n").split("=")[-1])
      score  = float(fp.readline().rstrip("\n").split("=")[-1])
      left  = float(fp.readline().rstrip("\n").split("=")[-1])
      top  = float(fp.readline().rstrip("\n").split("=")[-1])
      right  = float(fp.readline().rstrip("\n").split("=")[-1])
      bottom  = float(fp.readline().rstrip("\n").split("=")[-1])
      freame_objs.append([left,top,right,bottom,score,category])
    result.update({frame_name:freame_objs})
  return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calc Precision and Recall")
    parser.add_argument("--ground_truths",default="/workspace/bmnnsdk2-bm1684_v2.7.0/examples/YOLOX_object/py_sail/yolox_save/zhuheqiao_compilation.txt", type=str)
    parser.add_argument("--detections",default="/workspace/bmnnsdk2-bm1684_v2.7.0/examples/YOLOX_object/cpp_sail/save/zhuheqiao_compilation.txt", type=str)
    parser.add_argument("--iou_threshold",default=0.5, type=float)
    opt=parser.parse_args()

    if os.path.exists(opt.ground_truths) is False:
        print("Can not found ground_truths file: {}".format(opt.ground_truths))
        exit(1)
    if os.path.exists(opt.detections) is False:
        print("Can not found detections file: {}".format(opt.detections))
        exit(1)

    ground_truths = get_detect_from_file(opt.ground_truths)
    detections = get_detect_from_file(opt.detections)
    total_true_count = 0
    total_detection_count = 0
    total_gt_count = 0
    for key, value_gt in ground_truths.items():
        if key in detections.keys():
            value_dete = detections[key]
            true_num, dete_num, gt_num = get_diff_num(value_dete, value_gt, opt.iou_threshold)
            print("{}: true detections:{}, detections:{}, ground truths:{}, Recall:{:.2f}%, Presicion:{:.2f}%".format(
                key, true_num, dete_num, gt_num,true_num/gt_num*100, true_num/dete_num*100))
        else:
            true_num, dete_num, gt_num = 0, 0, len(value_gt)
            print("{}: true detections:{}, detections:{}, ground truths:{}, Recall:0, Accuracy:0 ".format(
                key, true_num, dete_num, gt_num))            


        total_true_count = total_true_count + true_num
        total_detection_count = total_detection_count + dete_num
        total_gt_count = total_gt_count + gt_num

    print(total_detection_count)
    precision = total_true_count/total_detection_count
    recall = total_true_count/total_gt_count
    print("IOU Threshold:{}, True:{}, Dete:{}, GT:{}, Recall:{:.2f}%, Accuracy:{:.2f}%".format(
        opt.iou_threshold, total_true_count, total_detection_count, total_gt_count, recall*100, precision*100))

    if recall < 0.75:
        exit(1)
    if precision < 0.90:
        exit(1)
