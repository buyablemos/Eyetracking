import pyautogui
import numpy as np
import cv2
import time

def capture_screenshot():
    screenshot = pyautogui.screenshot()
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def generate_heatmap(frame):
    heatmap = np.zeros_like(frame, dtype=np.uint8)
    height, width, _ = frame.shape

    # Example heatmap data (random points)
    for _ in range(1000):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        heatmap[y:y+10, x:x+10] = [0, 0, 255]  # Red points

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

def overlay_heatmap(frame, heatmap):
    return cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

def create_video_with_heatmap(filename, frame_rate=10, duration=10, heatmap_interval=5):
    frame_width, frame_height = pyautogui.size()
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (frame_width, frame_height))

    num_frames = frame_rate * duration
    heatmap = None
    for i in range(num_frames):
        frame = capture_screenshot()
        if i % heatmap_interval == 0 or heatmap is None:
            heatmap = generate_heatmap(frame)
        overlayed_frame = overlay_heatmap(frame, heatmap)
        out.write(overlayed_frame)
        time.sleep(1/frame_rate)

    out.release()

# Użyj funkcji do stworzenia nagrania
create_video_with_heatmap('screen_recording_with_heatmap.avi')
