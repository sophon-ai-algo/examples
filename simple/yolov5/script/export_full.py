"""Export a YOLOv5 *.pt model to TorchScript, ONNX, CoreML formats

Usage:
    $ python path/to/export.py --weights yolov5s.pt --img 640 --batch 1
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.common import Conv
from models.yolo import Detect, Model
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import colorstr, check_img_size, check_requirements, file_size, set_logging, non_max_suppression
from utils.torch_utils import select_device


class FullModel(nn.Module):
    def __init__(self, model, conf_thres=0.001, iou_thres=0.6):  # model, input channels, number of classes
        super(FullModel, self).__init__()
        self.model  = model
        self.model.eval()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def forward(self, x):
        x, _ = self.model(x, False, False)
        # x = non_max_suppression(x, self.conf_thres, self.iou_thres)
        return x

def run(weights='./yolov5s.pt',  # weights path
        img_size=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        conf_thres = 0.001,
        iou_thres = 0.6,
        optimize=False,  # TorchScript: optimize for mobile
        dynamic=False,  # ONNX: dynamic axes
        ):
    device = "cpu"
    half=False  # FP16 half-precision export
    inplace=False  # set YOLOv5 Detect() inplace=True
    t = time.time()
    img_size *= 2 if len(img_size) == 1 else 1  # expand

    # Load PyTorch model
    device = select_device(device)
    assert not (device.type == 'cpu' and half), '--half only compatible with GPU export, i.e. use --device 0'
    model = attempt_load(weights, map_location=device)  # load FP32 model
    labels = model.names

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    img_size = [check_img_size(x, gs) for x in img_size]  # verify img_size are gs-multiples
    img = torch.zeros(batch_size, 3, *img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic

    # TorchScript export -----------------------------------------------------------------------------------------------
    for _ in range(2):
        y = model(img)  # dry runs
    prefix = colorstr('TorchScript:')
    try:
        print(f'\n{prefix} starting export with torch {torch.__version__}...')
        full_model = FullModel(model, conf_thres, iou_thres)
        full_model.to(device)
        full_model.eval()
        f = weights.replace('.pt', '.torchscript.pt')  # filename
        ts = torch.jit.trace(full_model, img, strict=False)
        (optimize_for_mobile(ts) if optimize else ts).save(f)
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')

    # Finish
    print(f'\nExport complete ({time.time() - t:.2f}s). Visualize with https://github.com/lutzroeder/netron.')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image (height, width)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='iou threshold')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    parser.add_argument('--dynamic', action='store_true', help='ONNX: dynamic axes')
    opt = parser.parse_args()
    return opt


def main(opt):
    set_logging()
    print(colorstr('export: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
