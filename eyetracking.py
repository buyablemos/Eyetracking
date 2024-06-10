import cv2
from gaze_tracking import GazeTracking
import pygame
import numpy as np
import time
from scipy.ndimage import gaussian_filter
from scroll import *
from image import *
from image2 import *
import subprocess
import pyautogui


def capture_screenshot():
    screenshot = pyautogui.screenshot()
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# Ustawienia ekranu
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 760
BG_COLOR = (0, 0, 0)
BG_HEATMAP_COLOR = (255,255, 255)
POINT_COLOR = (255, 0, 0)
POINT_RADIUS = 10

# Lista punktów do kalibracji
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


def draw_calibration_points(point_index, screen):
    screen.fill(BG_COLOR)
    draw_point(*CALIBRATION_POINTS[point_index], POINT_COLOR, screen)
    pygame.display.flip()


def overlay_heatmap(frame, heatmap):
    heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(frame, 0.6, heatmap_resized, 0.4, 0)

def draw_point(x, y, color, screen):
    pygame.draw.circle(screen, color, (x, y), POINT_RADIUS)


def get_color(value, min_val, max_val):
    normalized_value = (value - min_val) / (max_val - min_val)
    if normalized_value < 0.1:
        return 255, 255, int(255 * (1 - (normalized_value / 0.1)))
    elif normalized_value < 0.5:
        return 255, int(255 - (90 * ((normalized_value - 0.1) / 0.4))), 0
    else:
        return 255, int(165 * (1 - ((normalized_value - 0.5) / 0.5))), 0


def draw_heatmap_from_points(points, page_width=SCREEN_WIDTH, page_height=SCREEN_HEIGHT):
    global max_val
    global heatmap


    heatmap_color=np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT,3), dtype=np.uint8)

    for x, y in np.ndindex(heatmap.shape):
        heatmap[x, y] /= 2

    for x_average, y_average in points:
        x_average = int(x_average)
        y_average = int(y_average)

        if 0 <= x_average < page_width and 0 <= y_average < page_height:
            heatmap[x_average - 10: x_average + 10, y_average - 10: y_average + 10] += 100
            if np.any(heatmap[x_average - 10: x_average + 10, y_average - 10: y_average + 10] > max_val):
                max_val = np.max(heatmap[x_average - 10: x_average + 10, y_average - 10: y_average + 10])

    heatmap_smooth = gaussian_filter(heatmap, sigma=12)
    min_val = np.min(heatmap_smooth)
    max_val = np.max(heatmap_smooth)

    for x in range(0, page_width, 10):
        for y in range(0, page_height, 10):
            color = get_color(heatmap_smooth[x, y], min_val, max_val)
            for i in range(x, min(x + 10, page_width)):
                for j in range(y, min(y + 10, page_height)):
                    heatmap_color[i][j]=color

    return heatmap_color


def calibrate():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Calibration")
    calibrations = []
    last_time_checked = 1000
    for i, point in enumerate(CALIBRATION_POINTS):
        draw_calibration_points(i, screen)
        if i == 0:
            time.sleep(1)
        while True:
            current_time = pygame.time.get_ticks()
            if current_time - last_time_checked > 1000:  # Check every second (1000 milliseconds)
                last_time_checked = current_time
                _, frame = webcam.read()
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                gaze.refresh(frame)  # Pobieramy nową klatkę z eyetrackera
                left_eye_coords = gaze.pupil_left_coords()
                right_eye_coords = gaze.pupil_right_coords()
                if left_eye_coords is not None and right_eye_coords is not None:
                    left_eye_x, left_eye_y = left_eye_coords
                    right_eye_x, right_eye_y = right_eye_coords
                    calibrations.append((point, (left_eye_x, left_eye_y), (right_eye_x, right_eye_y)))
                    screen.fill(BG_COLOR)
                    pygame.display.flip()
                    break
            else:
                time.sleep(0.1)
        screen.fill(BG_COLOR)
        pygame.display.flip()
    pygame.quit()
    return calibrations


def create_calibration_function(calibration_data):
    left_eye_x = list()
    left_eye_y = list()
    right_eye_x = list()
    right_eye_y = list()
    x_coords = list()
    y_coords = list()
    for elem in calibration_data:
        left_eye_x.append(elem[1][0])
        left_eye_y.append(elem[1][1])
        right_eye_x.append(elem[2][0])
        right_eye_y.append(elem[2][1])
        x_coords.append(elem[0][0])
        y_coords.append(elem[0][1])

    degree = 1
    coefx = np.polyfit(left_eye_x, x_coords, degree)
    coefy = np.polyfit(left_eye_y, y_coords, degree)
    coefx2 = np.polyfit(right_eye_x, x_coords, degree)
    coefy2 = np.polyfit(right_eye_y, y_coords, degree)
    polyx_left = np.poly1d(coefx)
    polyy_left = np.poly1d(coefy)
    polyx_right = np.poly1d(coefx2)
    polyy_right = np.poly1d(coefy2)
    return polyx_left, polyy_left, polyx_right, polyy_right


def interpolate_calibration_data(left_eye_position, right_eye_position, *wielomiany):
    x_lewy = wielomiany[0](left_eye_position[0])
    y_lewy = wielomiany[1](left_eye_position[1])
    x_prawy = wielomiany[2](right_eye_position[0])
    y_prawy = wielomiany[3](right_eye_position[1])

    x = int((x_prawy + x_lewy) / 2)
    y = int((y_prawy + y_lewy) / 2)

    return x, y


gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

points = []

# Kalibracja
calibration_data = calibrate()

wielomianx_lewy, wielomiany_lewy, wielomianx_prawy, wielomiany_prawy = create_calibration_function(calibration_data)

x_positions = []
y_positions = []
filename = 'screen_recording_with_heatmap.avi'


points_for_heatmap = []
heatmap = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT))
heatmap_smooth = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT))
max_val = 0

x_average = 0
y_average = 0

running = True

out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 10, (SCREEN_WIDTH, SCREEN_HEIGHT))

while running:
    i = 0
    while i < 4:
        # Pobieramy nową klatkę z kamery
        _, frame = webcam.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        gaze.refresh(frame)

        # Odczytujemy współrzędne gałek ocznych
        left_eye_coords = gaze.pupil_left_coords()
        right_eye_coords = gaze.pupil_right_coords()

        if left_eye_coords is not None and right_eye_coords is not None:
            left_eye_x, left_eye_y = left_eye_coords
            right_eye_x, right_eye_y = right_eye_coords

            # Interpolacja danych kalibracyjnych dla lewego oka
            calibrated_eye_x, calibrated_eye_y = interpolate_calibration_data(
                (left_eye_x, left_eye_y),
                (right_eye_x, right_eye_y),
                wielomianx_lewy, wielomiany_lewy, wielomianx_prawy, wielomiany_prawy
            )
            x_positions.append(calibrated_eye_x)
            y_positions.append(calibrated_eye_y)
            i += 1

    x_average = sum(x_positions) / len(x_positions)
    y_average = sum(y_positions) / len(y_positions)

    if x_average > SCREEN_WIDTH:
        x_average = SCREEN_WIDTH
    if x_average < 0:
        x_average = 0

    if y_average > SCREEN_HEIGHT:
        y_average = SCREEN_HEIGHT
    if y_average < 0:
        y_average = 0

    points_for_heatmap.append((x_average, y_average))

    screenshot = capture_screenshot()
    heatmap_image = draw_heatmap_from_points(points_for_heatmap)
    overlayed_frame = overlay_heatmap(screenshot, heatmap_image)
    out.write(overlayed_frame)


    x_positions.clear()
    y_positions.clear()
    print(x_average, y_average)

out.release()
webcam.release()
cv2.destroyAllWindows()
