from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
#load_model
device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
weights = ROOT / 'runs/train/exp/weights/best.pt'  # model path or triton URL
data = ROOT / 'data/cf.yaml'  # dataset.yaml path
imgsz = (640, 640)  # inference size (height, width)
half = False  # use FP16 half-precision inference

conf_thres = 0.25  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
max_det = 1000  # maximum detections per image
classes = None  # filter by class: --class 0, or --class 0 2 3

line_thickness = 3  # bounding box thickness (pixels)