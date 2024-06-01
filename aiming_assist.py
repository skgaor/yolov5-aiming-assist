import os
import sys
import time
import pynput.mouse
import torch
import pathlib
from mouse.mouse import mouse_xy, click_mouse_button
from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.augmentations import letterbox
from utils.torch_utils import select_device
from get_window import *
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
debug = True


def loading_model():
    device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    imgsz = (640, 640)  # inference size (height, width)
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
            xy_list.append(xywh)
            # 锚框
            c = int(cls)  # integer class
            label = (f'{names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))
    result = annotator.result()
    return result, xy_list


def get_mouse_position(mouse):
    mouse_x, mouse_y = mouse.position
    mouse_x *= scaling_factor
    mouse_y *= scaling_factor
    return mouse_x, mouse_y


def aim_lock(xy_list, mouse, left, top, width, height):
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
    mouse_xy(dx * speed, dy * speed, True)
    click_mouse_button(1)


def on_click(x, y, button, pressed):
    global LOCK_MOUSE
    if pressed and button == button.x2:
        LOCK_MOUSE = not LOCK_MOUSE
        print('LOCK_MOUSE:', LOCK_MOUSE)


if __name__ == '__main__':
    # 模型加载部分参数
    weights = ROOT / 'runs/train/exp8/weights/best.pt'  # model path or triton URL
    data = ROOT / 'data/cf.yaml'  # dataset.yaml path
    half = False  # use FP16 half-precision inference
    #预测部分参数
    conf_thres = 0.7  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    classes = 0  # filter by class: --class 0, or --class 0 2 3
    # 预测结果部分参数
    line_thickness = 3  # bounding box thickness (pixels)

    # 获取窗口名
    hwnds = get_hwnd_list()
    for i in range(len(hwnds)):
        print(i + 1, ' ' + hwnds[i])
    x = int(input('请选择需要监测的窗口'))
    window_name = hwnds[x - 1]  # 被监测的窗口名

    # 获取屏幕分辨率
    sct = mss.mss()
    screen_size = sct.monitors[0]
    print('当前屏幕分辨率：', screen_size['width'], 'x', screen_size['height'])

    # 获取监测窗口大小
    monitor = get_screen_size(window_name, screen_number=0)
    print('监测窗口大小：', monitor['width'],'x',monitor['height'])

    # 监测窗口
    show_monitor = True
    screen_width = screen_size['width']
    screen_height = screen_size['height']
    monitor_width = monitor['width'] // 3
    monitor_height = monitor['height'] // 3
    sct = mss.mss()
    if show_monitor:
        monitor_name = 'GAME monitor'
        cv2.namedWindow(monitor_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(monitor_name, monitor_width, monitor_height)
        # 获取监测窗口句柄
        monitor_handle = win32gui.FindWindow(None, monitor_name)

    # yolo部分
    model, stride, names, pt, imgsz = loading_model()

    # 鼠标部分
    LOCK_MOUSE = False  # 锁定开关
    factor = 100 / 381  # 移动距离系数
    speed = 1  # 鼠标移动速度，防止抖动，建议小于1
    # 获取屏幕缩放比例（150%意味着比例是1.5）
    scaling_factor = 1
    mouse_controller = pynput.mouse.Controller()
    listener = pynput.mouse.Listener(on_click=on_click)
    listener.start()

    # 初始化计时器和帧数计数器
    start_time = time.time()

    # 定义编码器和创建 VideoWriter 对象
    save_video = True
    video_name = time.strftime("%Y%m%d-%H%M%S", time.localtime(start_time))
    video_name = f"{video_name}.mp4"
    video_dir = ROOT / 'video' / video_name
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v 编码器
        out = cv2.VideoWriter(str(video_dir), fourcc, 20.0, (monitor["width"], monitor["height"]))

    while True:
        #img:ScreenShot
        #im0:Orignal Image
        img = sct.grab(monitor)
        im, im0 = loderdata(img, stride, pt, imgsz)
        pred, im = pre_img(im, model)
        result, xy_list = get_result(pred, im, im0, names, monitor)
        if xy_list and LOCK_MOUSE:
            aim_lock(xy_list, mouse_controller, **monitor)
        # 计算并显示FPS
        now = time.time()
        elapsed_time = now - start_time
        fps = 1 / elapsed_time
        start_time = now
        #print('FPS:', fps)
        # 设置窗口为置顶
        if show_monitor:
            cv2.putText(result, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow(monitor_name, result)
            win32gui.SetWindowPos(monitor_handle, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        # 写入帧到视频文件
        if save_video:
            out.write(result)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            if save_video:
                out.release()
            cv2.destroyAllWindows()
            exit('exit monitor...')
