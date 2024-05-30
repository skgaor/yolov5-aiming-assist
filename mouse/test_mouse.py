from ctypes import CDLL, c_int, c_int64
from os import path

basedir = path.dirname(path.abspath(__file__))
ghubdlldir = path.join(basedir, 'ghub_device.dll')
msdkdlldir = path.join(basedir, 'msdk.dll')

# ↓↓↓↓↓↓↓↓↓ 调用ghub/键鼠驱动 ↓↓↓↓↓↓↓↓↓

gm = CDLL(ghubdlldir)
gmok = gm.device_open() == 1
#gmok = gm.Agulll()
print(gmok)
import time
import pynput
mouse_controller = pynput.mouse.Controller()
while True:
    mouse_x, mouse_y = mouse_controller.position
    #mouse_x *= 1.5
    #mouse_y *= 1.5
    print()
    print('mouse_x:',mouse_x)
    print('mouse_y:',mouse_y)
    time.sleep(1)