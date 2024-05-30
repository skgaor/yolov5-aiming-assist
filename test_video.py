import cv2
import time
import numpy as np
from mss import mss
from collections import deque

# 定义要截取的屏幕区域
monitor = {"top": 0, "left": 0, "width": 800, "height": 600}

# 创建 mss 对象
sct = mss()

# 获取视频帧的宽度和高度
frame_width = monitor["width"]
frame_height = monitor["height"]

# 定义编码器和创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 编码器
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

# 初始化滑动窗口和时间
frame_times = deque(maxlen=60)  # 保留最近60帧的时间戳
start_time = time.time()
fps = 0
fps_display_time = time.time()

while True:
    # 获取屏幕截图
    screenshot = sct.grab(monitor)

    # 将截图转换为 numpy 数组
    frame = np.array(screenshot)

    # 将 BGRA 转换为 BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # 获取当前时间戳
    current_time = time.time()

    # 将时间戳添加到滑动窗口
    frame_times.append(current_time)

    # 每秒计算并更新一次FPS
    if current_time - fps_display_time >= 1.0:
        elapsed_time = frame_times[-1] - frame_times[0]
        if elapsed_time > 0 and len(frame_times) > 1:
            fps = len(frame_times) / elapsed_time
        fps_display_time = current_time

    # 显示FPS
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 写入帧到视频文件
    out.write(frame)

    # 显示帧
    cv2.imshow('Frame', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放 VideoWriter 对象，并关闭所有窗口
out.release()
cv2.destroyAllWindows()
