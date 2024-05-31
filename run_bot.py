import os
import sys
import pynput.mouse
import torch

from pathlib import Path
from mouse.mouse import mouse_xy
from models.common import DetectMultiBackend
from utils.general import (check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.augmentations import letterbox
from utils.torch_utils import select_device
from get_window import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
debug=True
def loading_model():
    device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    weights = ROOT / 'runs/train/exp/weights/best.pt'  # model path or triton URL
    data = ROOT / 'data/cf.yaml'  # dataset.yaml path
    imgsz = (640, 640)  # inference size (height, width)
    half = False  # use FP16 half-precision inference
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))  # warmup
    return model, stride, names, pt, imgsz


def loderdata(img, stride, pt, imgsz):
    # dataLoder
    im0 = np.array(img)[:, :, :3]  # [:, :, :3] BGRA to BGR
    im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    return im, im0


def pre_img(im, model):
    conf_thres = 0.7  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    classes = 0  # filter by class: --class 0, or --class 0 2 3

    #predict
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred = model(im, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=max_det)

    return pred, im


def get_result(pred, im, im0s, names, monitor):
    line_thickness = 3  # bounding box thickness (pixels)
    im0 = im0s.copy()
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    annotator = Annotator(im0, line_width=line_thickness, example=str(names))

    det = pred[0]
    xy_list = []
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
        # Write results
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh)  # label format
            xy_list.append(xywh)
            #print(xywh)

            c = int(cls)  # integer class
            label = (f'{names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))
    result = annotator.result()
    return result, xy_list


def get_mouse_position(mouse):
    global scaling_factor
    # 调整坐标考虑缩放比例
    mouse_x, mouse_y = mouse.position
    mouse_x *= scaling_factor
    mouse_y *= scaling_factor
    return mouse_x, mouse_y


def aim_lock(xy_list, mouse, left, top, width, height):
    global factor
    mouse_x, mouse_y = get_mouse_position(mouse)
    best_xy = None
    for xywh in xy_list:
        x, y, _, _ = xywh
        dist = ((x * width + left - mouse_x) ** 2 + (y * height + top - mouse_y) ** 2) ** 0.5
        if not best_xy:
            best_xy = ((x, y), dist)
        else:
            _, old_dist = best_xy
            if dist < old_dist:
                best_xy = ((x, y), dist)

    x, y = best_xy[0]
    x = x * width + left
    y = y * height + top
    dx = (x - mouse_x) * factor
    dy = (y - mouse_y) * factor
    '''print('x:', x)
    print('y:', y)
    print('mouse_x:', mouse_x)
    print('mouse_y:', mouse_y)'''
    speed = 0.5
    mouse_xy(dx * speed, dy * speed, False)


def on_click(x, y, button, pressed):
    global LOCK_MOUSE
    if pressed and button == button.x2:
        LOCK_MOUSE = not LOCK_MOUSE
        print('LOCK_MOUSE:', LOCK_MOUSE)


if __name__ == '__main__':
    # 获取窗口名
    hwnds = get_hwnd_list()
    for i in range(len(hwnds)):
        print(i + 1, ' ' + hwnds[i])
    x = int(input('请选择需要监测的窗口'))
    window_name = hwnds[x - 1]
    # print(window_name)

    # 获取屏幕分辨率
    sct = mss.mss()
    screen_size = sct.monitors[0]
    print(screen_size)

    # 获取监测窗口大小
    monitor = get_screen_size(window_name, screen_number=0)
    print(monitor)

    # 监测窗口
    screen_width = screen_size['width']
    screen_height = screen_size['height']
    monitor_width = screen_width // 5
    monitor_height = screen_height // 5
    sct = mss.mss()
    window_name = 'GAME monitor'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, monitor_width, monitor_height)
    # 获取监测窗口句柄
    handle = win32gui.FindWindow(None, window_name)

    #yolo部分
    model, stride, names, pt, imgsz = loading_model()

    #鼠标部分
    LOCK_MOUSE = False
    factor = 100 / 381
    # 获取屏幕缩放比例（150%意味着比例是1.5）
    scaling_factor = 1
    mouse_controller = pynput.mouse.Controller()
    listener = pynput.mouse.Listener(on_click=on_click)
    listener.start()

    while True:
        #img:ScreenShot
        #im0:Orignal Image
        img = sct.grab(monitor)
        im, im0 = loderdata(img, stride, pt, imgsz)
        pred, im = pre_img(im, model)
        result, xy_list = get_result(pred, im, im0, names, monitor)
        if xy_list and LOCK_MOUSE:
            aim_lock(xy_list, mouse_controller, **monitor)
        cv2.imshow(window_name, result)
        # 设置窗口为置顶
        win32gui.SetWindowPos(handle, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            cv2.destroyAllWindows()
            exit('exit monitor...')
