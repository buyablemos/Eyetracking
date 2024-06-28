import cv2
from dlib import points
from gaze_tracking import GazeTracking
import pygame
import sys
import numpy as np
import time
import tkinter as tk
from ctypes import windll

pygame.init()

#--------!!!----------
# Set your screen size
#--------!!!----------
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 800
BG_COLOR = (0, 0, 0)
POINT_COLOR = (255, 0, 0)
POINT_RADIUS = 10

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Eye Tracking Calibration")
clock = pygame.time.Clock()

CALIBRATION_POINTS = [
    (int(0.05 * SCREEN_WIDTH), int(0.05 * SCREEN_HEIGHT)),
    (int(0.3 * SCREEN_WIDTH), int(0.05 * SCREEN_HEIGHT)),
    (int(0.6 * SCREEN_WIDTH), int(0.05 * SCREEN_HEIGHT)),
    (int(0.95 * SCREEN_WIDTH), int(0.05 * SCREEN_HEIGHT)),
    (int(0.05 * SCREEN_WIDTH), int(0.3 * SCREEN_HEIGHT)),
    (int(0.3 * SCREEN_WIDTH), int(0.3 * SCREEN_HEIGHT)),
    (int(0.6 * SCREEN_WIDTH), int(0.3 * SCREEN_HEIGHT)),
    (int(0.95 * SCREEN_WIDTH), int(0.3 * SCREEN_HEIGHT)),
    (int(0.05 * SCREEN_WIDTH), int(0.6 * SCREEN_HEIGHT)),
    (int(0.3 * SCREEN_WIDTH), int(0.6 * SCREEN_HEIGHT)),
    (int(0.6 * SCREEN_WIDTH), int(0.6 * SCREEN_HEIGHT)),
    (int(0.95 * SCREEN_WIDTH), int(0.6 * SCREEN_HEIGHT)),
    (int(0.05 * SCREEN_WIDTH), int(0.95 * SCREEN_HEIGHT)),
    (int(0.3 * SCREEN_WIDTH), int(0.95 * SCREEN_HEIGHT)),
    (int(0.6 * SCREEN_WIDTH), int(0.95 * SCREEN_HEIGHT)),
    (int(0.95 * SCREEN_WIDTH), int(0.95 * SCREEN_HEIGHT)),
]

max_val = 0

def draw_point(x, y, color):
    pygame.draw.circle(screen, color, (x, y), POINT_RADIUS)

def draw_calibration_points(point_index):
    screen.fill(BG_COLOR)
    draw_point(*CALIBRATION_POINTS[point_index], POINT_COLOR)
    pygame.display.flip()

def calibrate(gaze, webcam):
    calibrations = []
    for i, point in enumerate(CALIBRATION_POINTS):
        while True:
            draw_calibration_points(i)
            time.sleep(1)
            ret, frame = webcam.read()
            if not ret:
                continue

            #-------!!!-------
            # Rotate if needed
            #-------!!!-------            
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            gaze.refresh(frame)
            left_eye_coords = gaze.pupil_left_coords()
            right_eye_coords = gaze.pupil_right_coords()
            if left_eye_coords is not None and right_eye_coords is not None:
                calibrations.append((point, left_eye_coords, right_eye_coords))
                break
            screen.fill(BG_COLOR)
            pygame.display.flip()
    screen.fill(BG_COLOR)
    pygame.display.flip()
    return calibrations

def create_calibration_function(calibration_data):
    left_eye_x, left_eye_y, right_eye_x, right_eye_y, x_coords, y_coords = [], [], [], [], [], []
    for elem in calibration_data:
        left_eye_x.append(elem[1][0])
        left_eye_y.append(elem[1][1])
        right_eye_x.append(elem[2][0])
        right_eye_y.append(elem[2][1])
        x_coords.append(elem[0][0])
        y_coords.append(elem[0][1])

    degree = 2
    polyx_left = np.poly1d(np.polyfit(left_eye_x, x_coords, degree))
    polyy_left = np.poly1d(np.polyfit(left_eye_y, y_coords, degree))
    polyx_right = np.poly1d(np.polyfit(right_eye_x, x_coords, degree))
    polyy_right = np.poly1d(np.polyfit(right_eye_y, y_coords, degree))
    return polyx_left, polyy_left, polyx_right, polyy_right

def interpolate_calibration_data(left_eye_position, right_eye_position, polyx_left, polyy_left, polyx_right, polyy_right):
    x_lewy = polyx_left(left_eye_position[0])
    y_lewy = polyy_left(left_eye_position[1])
    x_prawy = polyx_right(right_eye_position[0])
    y_prawy = polyy_right(right_eye_position[1])

    x = int((x_prawy + x_lewy) / 2)
    y = int((y_prawy + y_lewy) / 2)

    return x, y

def create_dot_overlay():
    overlay = tk.Toplevel()
    overlay.title("Dot Overlay")
    
    size = 20
    overlay.geometry(f"{size}x{size}+0+0")

    overlay.attributes('-alpha', 0.5)
    
    overlay.attributes('-topmost', True)

    overlay.overrideredirect(1)

    hwnd = windll.user32.GetParent(overlay.winfo_id())
    styles = windll.user32.GetWindowLongPtrW(hwnd, -20)
    windll.user32.SetWindowLongPtrW(hwnd, -20, styles | 0x80000 | 0x20)

    canvas = tk.Canvas(overlay, width=size, height=size, bg='blue', highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    
    radius = size // 2
    dot = canvas.create_oval(0, 0, size, size, fill='red', outline='')

    def close_overlay(event=None):
        overlay.destroy()
        root.quit()

    overlay.bind("<Escape>", close_overlay)
    overlay.protocol("WM_DELETE_WINDOW", close_overlay)

    return overlay, canvas

def move_dot(overlay, canvas, x, y):
    size = 20
    a=screen_width/SCREEN_WIDTH
    b=screen_height/SCREEN_HEIGHT
    overlay.geometry(f"{size}x{size}+{int(x*a - size/2)}+{int(y*b - size/2)}")

gaze = GazeTracking()

#-------!!!--------
#Choose your camera
#-------!!!--------
webcam = cv2.VideoCapture(0)

calibration_data = calibrate(gaze, webcam)
polyx_left, polyy_left, polyx_right, polyy_right = create_calibration_function(calibration_data)

x_positions = []
y_positions = []

root = tk.Tk()
root.withdraw() 

overlay, canvas = create_dot_overlay()

screen_width=root.winfo_screenwidth()
screen_height=root.winfo_screenheight()
try:
    while True:
        for i in range(3):
            ret, frame = webcam.read()
            if not ret:
                continue

            #-------!!!-------
            # Rotate if needed
            #-------!!!-------   
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            gaze.refresh(frame)

            left_eye_coords = gaze.pupil_left_coords()
            right_eye_coords = gaze.pupil_right_coords()

            if left_eye_coords is not None and right_eye_coords is not None:
                calibrated_eye_x, calibrated_eye_y = interpolate_calibration_data(
                    left_eye_coords, right_eye_coords,
                    polyx_left, polyy_left, polyx_right, polyy_right
                )
                x_positions.append(calibrated_eye_x)
                y_positions.append(calibrated_eye_y)

        if x_positions and y_positions:
            x_average = sum(x_positions) / len(x_positions)
            y_average = sum(y_positions) / len(y_positions)

            if x_average > screen_width:
                x_average = screen_width
            if x_average < 0:
                x_average = 0

            if y_average > screen_height:
                y_average = screen_height
            if y_average < 0:
                y_average = 0

            x_positions.clear()
            y_positions.clear()

            move_dot(overlay, canvas, x_average, y_average)

        root.update_idletasks()
        root.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                webcam.release()
                cv2.destroyAllWindows()
                sys.exit()

except Exception as e:
    print(f"An error occurred: {e}")
    pygame.quit()
    webcam.release()
    cv2.destroyAllWindows()
    sys.exit()
