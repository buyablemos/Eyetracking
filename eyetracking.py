import cv2
from gaze_tracking import GazeTracking
import pygame
import numpy as np
import time
from scipy.ndimage import gaussian_filter
from scroll import *
from image import *

# Ustawienia ekranu
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 800
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


def draw_heatmap_from_points(points, page_width, page_height):
    global max_val

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
                    screen.set_at((i, j), color)


def calibrate():
    pygame.quit()
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


# Inicjalizacja eyetrackera
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

points = []

# Kalibracja
calibration_data = calibrate()

time.sleep(5)

wielomianx_lewy, wielomiany_lewy, wielomianx_prawy, wielomiany_prawy = create_calibration_function(calibration_data)

x_positions = []
y_positions = []

driver = init_driver()
app = BrowserTrackerApp(driver)
page_height, page_width, viewport_height, scroll_top = app.return_dimensions()
pygame.init()
pygame.display.set_caption("Eye Tracking")
screen = pygame.display.set_mode((page_width, page_height))

points_for_heatmap = []
heatmap = np.zeros((page_width, page_height))
heatmap_smooth = np.zeros((page_width, page_height))
max_val = 0

x_average = 0
y_average = 0

running = True
time.sleep(5)
while running:
    i = 0
    while i < 4:
        # Pobieramy nową klatkę z kamery
        _, frame = webcam.read()
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

    if x_average > page_width:
        x_average = page_width
    if x_average < 0:
        x_average = 0

    if y_average > viewport_height:
        y_average = viewport_height
    if y_average < 0:
        y_average = 0

    app = BrowserTrackerApp(driver)
    dimension = app.return_dimensions()
    scroll_top = dimension[3]


    y_average = y_average + scroll_top

    if y_average > page_height:
        y_average = page_height

    points_for_heatmap.append((x_average, y_average))

    x_positions.clear()
    y_positions.clear()
    print(x_average, y_average)

    # Odświeżenie ekran
    screen.fill(BG_COLOR)
    draw_point(x_average, y_average, (0, 0, 255), screen)
    pygame.display.flip()

    # Obsługa zdarzeń
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.display.flip()
            draw_heatmap_from_points(points_for_heatmap, page_width, page_height)
            pygame.image.save(screen, 'zapisana_heatmapa.png')
            create_heatmap_on_screen()
            pygame.quit()
            webcam.release()
            break
        if event.type == pygame.MOUSEBUTTONDOWN:
            draw_heatmap_from_points(points_for_heatmap, page_width, page_height)
            pygame.display.flip()
