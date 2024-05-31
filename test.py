import pyautogui
import pynput
import time
from config import *

# 获取屏幕缩放比例（150%意味着比例是1.5）
#scaling_factor = 1
from mouse.mouse import move_mouse, mouse_xy


def get_mouse_position(mouse):
    # 调整坐标考虑缩放比例
    scaling_factor = 1
    mouse_x, mouse_y = mouse.position
    mouse_x *= scaling_factor
    mouse_y *= scaling_factor
    return mouse_x, mouse_y


mouse_controller = pynput.mouse.Controller()
mouse_x, mouse_y = get_mouse_position(mouse_controller)

x = 584
y = 349
dx = x - mouse_x
dy = y - mouse_y
speed=0.5
mouse_xy(dx*speed, dy*speed, True)

mouse_x, mouse_y = get_mouse_position(mouse_controller)
print(mouse_x)
print(mouse_y)
#factor=1
factor=100/381
width, height = pyautogui.size()
print(width)
print(height)
'''while True:
    #lock
    #mouse_controller = pynput.mouse.Controller()
    mouse_x, mouse_y = get_mouse_position(mouse_controller)
    print('mouse_x:', mouse_x)
    print('mouse_y:', mouse_y)
    dx = (x - mouse_x)*factor
    dy = (y - mouse_y)*factor
    print('dx:',dx)
    print('dy:',dy)
    print()
    mouse_xy(dx*speed, dy*speed, False)
    time.sleep(0.05)'''


