import cv2
from gaze_tracking import GazeTracking
import pygame
import sys
import numpy as np
import time
from scipy.ndimage import gaussian_filter
from mss import mss

# Initialize pygame
pygame.init()

# Screen settings
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 800
BG_COLOR = (0, 0, 0)
POINT_COLOR = (255, 0, 0)
POINT_RADIUS = 10

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Eye Tracking Calibration")
clock = pygame.time.Clock()

# Calibration points
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

heatmap = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT))
heatmap_smooth = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT))
heatmap_image = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=np.uint8)
max_val = 0

def draw_point(x, y, color):
    pygame.draw.circle(screen, color, (x, y), POINT_RADIUS)

def get_color(value, min_val, max_val):
    normalized_value = (value - min_val) / (max_val - min_val)
    if normalized_value < 0.1:
        return (0, 0, 0)
    elif normalized_value < 0.5:
        return (int(255 * normalized_value * 2), 0, 0)
    elif normalized_value < 0.7:
        return (255, int(255 * (normalized_value - 0.5) / 0.2), 0)
    else:
        return (255, int(165 + 90 * (normalized_value - 0.7) / 0.3), 0)

def draw_heatmap(x_average, y_average):
    global max_val, heatmap_smooth

    x_average = int(x_average)
    y_average = int(y_average)

    if 0 <= x_average < SCREEN_WIDTH and 0 <= y_average < SCREEN_HEIGHT:
        heatmap[max(0, x_average - 10): min(SCREEN_WIDTH, x_average + 10), 
                max(0, y_average - 10): min(SCREEN_HEIGHT, y_average + 10)] += 10

        local_max = np.max(heatmap[max(0, x_average - 10): min(SCREEN_WIDTH, x_average + 10), 
                                   max(0, y_average - 10): min(SCREEN_HEIGHT, y_average + 10)])
        if local_max > max_val:
            max_val = local_max

    heatmap_smooth = gaussian_filter(heatmap, sigma=12)
    min_val = np.min(heatmap_smooth)
    max_val = np.max(heatmap_smooth)
    
    for x in range(0, SCREEN_WIDTH, 10):
        for y in range(0, SCREEN_HEIGHT, 10):
            color = get_color(heatmap_smooth[x, y], min_val, max_val)
            heatmap_image[x:x + 10, y:y + 10] = color

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

            frame = frame[150:300, 250:400]
            frame = cv2.resize(frame, (1200, 1200))
            gaze.refresh(frame)
            left_eye_coords = gaze.pupil_left_coords()
            right_eye_coords = gaze.pupil_right_coords()
            if left_eye_coords is not None and right_eye_coords is not None:
                calibrations.append((point, left_eye_coords, right_eye_coords))
                break
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

# Initialize eyetracker
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

# Calibration
calibration_data = calibrate(gaze, webcam)
polyx_left, polyy_left, polyx_right, polyy_right = create_calibration_function(calibration_data)

x_positions = []
y_positions = []

sct = mss()
monitor = sct.monitors[0]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (monitor['width'], monitor['height']))

try:
    while True:
        for i in range(10):
            ret, frame = webcam.read()
            if not ret:
                continue

            frame = frame[150:300, 250:400]
            frame = cv2.resize(frame, (1200, 1200))
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

            x_positions.clear()
            y_positions.clear()
            print(x_average, y_average)
            screen_shot = sct.grab(monitor)
            screen_img = np.array(screen_shot)
            screen_img = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2BGR)

            draw_heatmap(x_average, y_average)
            heatmap_resized = cv2.resize(heatmap_image, (screen_img.shape[1], screen_img.shape[0]))
            
            blended_image = cv2.addWeighted(screen_img, 0.4, heatmap_resized, 0.6, 0)
            out.write(blended_image)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                webcam.release()
                cv2.destroyAllWindows()
                out.release()
                sys.exit()

        clock.tick(20)
except Exception as e:
    print(f"An error occurred: {e}")
    pygame.quit()
    webcam.release()
    cv2.destroyAllWindows()
    out.release()
    sys.exit()
