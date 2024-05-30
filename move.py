import os
import pyautogui
import time
import glob
import pygetwindow as gw
import msvcrt
from pywinauto import Application
from pathlib import Path

def get_screen_size(window,screen_number=0):
    # 获取窗口的名称列表
    windows = gw.getWindowsWithTitle(window)# 将"窗口标题"替换为目标窗口的标题
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
    #monitor = {"top": top, "left": left, "width": width, "height": height}
    return [screen_number, left, top, width, height]

def set_window(window_title,size):
    # 获取窗口
    window = gw.getWindowsWithTitle(window_title)[0]
    # 获取窗口句柄
    hwnd = window._hWnd

    # 使用 pywinauto 来设置窗口位置
    app = Application().connect(handle=hwnd)
    app_window = app.window(handle=hwnd)

    # 设置窗口位置 (left, top, width, height)
    left = size[1]
    top = size[2]
    width = size[3]
    height = size[4]
    app_window.move_window(x=left, y=top, width=width, height=height)

    print(f"窗口已移动到位置: ({left}, {top})，大小: ({width}, {height})")


def get_latest_label_file(label_dir):
    files = list(Path(label_dir).glob('*.txt'))
    if not files:
        return None

    # 按照文件的创建时间排序，获取最新文件
    latest_file = max(files, key=os.path.getctime)
    return latest_file


def read_yolo_labels(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
    labels = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        center_x = float(parts[1])
        center_y = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        labels.append((class_id, center_x, center_y, width, height))
    return labels


def move_mouse_to_labels(labels, window_size):
    window_left, window_top = window_size[1], window_size[2]
    window_width, window_height = window_size[3], window_size[4]
    # print('window_left:',window_left,'window_top:',window_top)
    # print('window_width',window_width,'window_height',window_height)

    for class_id, center_x, center_y, width, height in labels:
        abs_center_x = window_left + int(center_x * window_width)
        abs_center_y = window_top + int(center_y * window_height)
        pyautogui.moveTo(abs_center_x, abs_center_y)
        print(f"Moved mouse to ({abs_center_x}, {abs_center_y}) for class {class_id}")
        # 你可以根据需要添加延迟或其他操作
        # pyautogui.click()  # 如果需要点击
        pyautogui.sleep(0.25)  # 等待0.25秒
        break

def delete_old_label_files(label_dir, retention_period=10):
    # 计算删除的截止时间
    delete_cutoff = time.time() - retention_period

    # 获取需要删除的文件列表
    files_to_delete = [file for file in Path(label_dir).glob('*.txt') if os.path.getctime(file) < delete_cutoff]

    # 删除文件
    for file in files_to_delete:
        file.unlink()
        #print(f"已删除较旧的标签文件: {file}")

if __name__ == "__main__":
    selected = ('4K街拍广州城中村、小巷子天河棠下 棠东 街道变漂亮很多！！！_哔哩哔哩_bilibili')
    window_size = get_screen_size(selected)
    size = ' '.join(map(str, window_size))
    print(size)

    #window_size = [0, 0, 0, 1024, 768]
    #set_window(selected, window_size)

    try:
        while True:
            # 标签输出目录
            directory = 'runs\detect\exp\labels'
            # 获取目录中的所有子目录
            detected = get_latest_label_file(directory)
            # 示例用法
            if detected:
                label_path = detected  # 替换为你的标签文件路径
                window_title = selected  # 替换为你要获取的窗口标题

                labels = read_yolo_labels(label_path)
                move_mouse_to_labels(labels, window_size)

            # 删除前10秒产生的标签文件
            delete_old_label_files(directory, retention_period=10)
            time.sleep(1)  # 每秒检查一次
    except Exception as e:
        print(e)