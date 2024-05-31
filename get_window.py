import time
import mss
import pygetwindow as gw
import cv2
import win32gui
import win32con
import numpy as np

hwnd_title = dict()


def get_all_hwnd(hwnd, mouse):
    if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
        hwnd_title.update({hwnd: win32gui.GetWindowText(hwnd)})


def get_screen_size(window, screen_number=0):
    # 获取窗口的名称列表
    windows = gw.getWindowsWithTitle(window)  # 将"窗口标题"替换为目标窗口的标题
    if len(windows) == 0:
        print("未找到指定窗口")
        exit(1)

    # 获取目标窗口的第一个匹配项
    window = windows[0]

    # 获取窗口的坐标和大小
    left, top, right, bottom = window.left, window.top, window.right, window.bottom
    width = right - left
    height = bottom - top
    # 定义捕获区域
    monitor = {"top": top, "left": left, "width": width, "height": height}
    return monitor


def get_hwnd_list():
    win32gui.EnumWindows(get_all_hwnd, 0)
    hwnds = [value for value in hwnd_title.values() if value]
    return hwnds


if __name__ == '__main__':

    # 获取窗口名
    hwnd_title = dict()
    win32gui.EnumWindows(get_all_hwnd, 0)
    hwnds = [value for value in hwnd_title.values() if value]
    for i in range(len(hwnds)):
        print(i + 1, ' ' + hwnds[i])
    x = int(input('请选择需要监测的窗口'))
    window_name = hwnds[x - 1]
    #print(window_name)

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
    # 获取窗口句柄
    handle = win32gui.FindWindow(None, window_name)
    # 初始化计时器和帧数计数器
    start_time = time.time()
    # 定义编码器和创建 VideoWriter 对象
    save_video = False
    video_name = time.strftime("%Y%m%d-%H%M%S", time.localtime(start_time))
    video_name = f"{video_name}.mp4"
    video_dir = video_name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v 编码器
    out = cv2.VideoWriter(str(video_dir), fourcc, 20.0, (monitor["width"], monitor["height"]))

    while True:
        img = sct.grab(monitor)
        img = np.array(img)
        # 将 BGRA 转换为 BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # 计算并显示FPS
        now = time.time()
        elapsed_time = now - start_time
        fps = 1 / elapsed_time
        start_time = now
        #print('FPS:', fps)
        cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow(window_name, img)
        # 写入帧到视频文件
        if save_video:
            out.write(img)
        # 设置窗口为置顶
        win32gui.SetWindowPos(handle, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            out.release()
            cv2.destroyAllWindows()
            exit('exit monitor...')
