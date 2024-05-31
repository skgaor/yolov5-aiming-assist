import multiprocessing
import time
import random
import keyboard
from multiprocessing import Manager


def screen_capture(screen_queue, running):
    while running.value:
        screen_image = f"ScreenImage_{random.randint(1, 100)}"
        screen_queue.put(screen_image)
        print(f"Captured: {screen_image}")
        time.sleep(1)


def model_predict(screen_queue, prediction_queue, running):
    while running.value:
        screen_image = screen_queue.get()
        prediction_image = f"PredictionImage_{screen_image}"
        coordinates = (random.randint(0, 1920), random.randint(0, 1080))
        prediction_queue.put((prediction_image, coordinates))
        print(f"Predicted: {prediction_image} with coordinates {coordinates}")
        screen_queue.task_done()
        time.sleep(1)


def mouse_control(prediction_queue, running):
    while running.value:
        _, coordinates = prediction_queue.get()
        print(f"Mouse moved to: {coordinates}")
        prediction_queue.task_done()
        time.sleep(1)


def result_display(prediction_queue, running):
    while running.value:
        prediction_image, _ = prediction_queue.get()
        print(f"Displaying: {prediction_image}")
        prediction_queue.task_done()
        time.sleep(1)


def listen_for_exit(running):
    while running.value:
        if keyboard.is_pressed('esc'):
            print("ESC pressed. Exiting...")
            running.value = False


if __name__ == "__main__":
    manager = Manager()
    screen_queue = manager.Queue()
    prediction_queue = manager.Queue()
    running = manager.Value('b', True)  # 使用Manager.Value创建一个共享的布尔值

    # 创建进程
    screen_process = multiprocessing.Process(target=screen_capture, args=(screen_queue, running))
    predict_process = multiprocessing.Process(target=model_predict, args=(screen_queue, prediction_queue, running))
    mouse_process = multiprocessing.Process(target=mouse_control, args=(prediction_queue, running))
    display_process = multiprocessing.Process(target=result_display, args=(prediction_queue, running))
    keyboard_process = multiprocessing.Process(target=listen_for_exit, args=(running,))

    # 启动进程
    screen_process.start()
    predict_process.start()
    mouse_process.start()
    display_process.start()
    keyboard_process.start()

    # 等待进程完成
    screen_process.join()
    predict_process.join()
    mouse_process.join()
    display_process.join()
    keyboard_process.join()

    print("所有进程完成")
