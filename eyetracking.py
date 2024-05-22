import cv2
from gaze_tracking import GazeTracking
import pygame
import sys
import numpy as np
import time
import asyncio
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter
from matplotlib.image import NonUniformImage

# Inicjalizacja pygame
pygame.init()

# Ustawienia ekranu
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 800
BG_COLOR = (0, 0, 0)
POINT_COLOR = (255, 0, 0)
POINT_RADIUS = 10

# Inicjalizacja ekranu
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Eye Tracking Calibration")

points = []


def draw_point(x, y, color):
    pygame.draw.circle(screen, color, (x, y), POINT_RADIUS)


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
HEATMAP_COLORS = [
    (0, 0, 0),  # Czarny
    (0, 63, 0),  # Ciemnozielony
    (0, 127, 0),  # Zielony
    (0, 191, 0),  # Jasnozielony
    (0, 255, 0),  # Jasnozielony
    (0, 0, 63),  # Granatowoniebieski
    (0, 0, 127),  # Granatowoniebieski
    (0, 0, 191),  # Granatowoniebieski
    (0, 0, 255),  # Granatowoniebieski
    (63, 0, 0),  # Granatowy
    (127, 0, 0),  # Granatowy
    (191, 0, 0),  # Ciemnoczerwony
    (255, 0, 0),  # Czerwony
]
heatmap = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT))
heatmap_smooth = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT))
max_val = 0


def draw_calibration_points(point_index):
    screen.fill(BG_COLOR)
    draw_point(*CALIBRATION_POINTS[point_index], POINT_COLOR)
    pygame.display.flip()


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
    global max_val
    x_average = int(x_average)
    y_average = int(y_average)

    if 0 <= x_average < SCREEN_WIDTH and 0 <= y_average < SCREEN_HEIGHT:
        heatmap[x_average - 10: x_average + 10, y_average - 10: y_average + 10] += 10
        if np.any(heatmap[x_average - 10: x_average + 10, y_average - 10: y_average + 10] > max_val):
            max_val = np.max(heatmap[x_average - 10: x_average + 10, y_average - 10: y_average + 10])

    heatmap_smooth = gaussian_filter(heatmap, sigma=12)
    min_val = np.min(heatmap_smooth)
    max_val = np.max(heatmap_smooth)

    for x in range(0, SCREEN_WIDTH, 10):
        for y in range(0, SCREEN_HEIGHT, 10):
            color = get_color(heatmap_smooth[x, y], min_val, max_val)
            for i in range(x, min(x + 10, SCREEN_WIDTH)):
                for j in range(y, min(y + 10, SCREEN_HEIGHT)):
                    screen.set_at((i, j), color)

def draw_heatmap_from_points(points):
    global max_val

    for x, y in np.ndindex(heatmap.shape):
        heatmap[x, y] /= 2

    for x_average, y_average in points:
        x_average = int(x_average)
        y_average = int(y_average)

        if 0 <= x_average < SCREEN_WIDTH and 0 <= y_average < SCREEN_HEIGHT:
            heatmap[x_average - 10: x_average + 10, y_average - 10: y_average + 10] += 100
            if np.any(heatmap[x_average - 10: x_average + 10, y_average - 10: y_average + 10] > max_val):
                max_val = np.max(heatmap[x_average - 10: x_average + 10, y_average - 10: y_average + 10])

    heatmap_smooth = gaussian_filter(heatmap, sigma=12)
    min_val = np.min(heatmap_smooth)
    max_val = np.max(heatmap_smooth)

    for x in range(0, SCREEN_WIDTH, 10):
        for y in range(0, SCREEN_HEIGHT, 10):
            color = get_color(heatmap_smooth[x, y], min_val, max_val)
            for i in range(x, min(x + 10, SCREEN_WIDTH)):
                for j in range(y, min(y + 10, SCREEN_HEIGHT)):
                    screen.set_at((i, j), color)


# Procedura kalibracji
def calibrate(gaze, webcam):
    calibrations = []
    for i, point in enumerate(CALIBRATION_POINTS):
        while True:
            draw_calibration_points(i)
            time.sleep(1)
            _, frame = webcam.read()
            #frame = frame[150:300, 250:400]
            #frame = cv2.resize(frame, (1200, 1200))
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            gaze.refresh(frame)  # Pobieramy nową klatkę z eyetrackera
            left_eye_coords = gaze.pupil_left_coords()
            right_eye_coords = gaze.pupil_right_coords()
            if left_eye_coords is not None and right_eye_coords is not None:
                left_eye_x, left_eye_y = left_eye_coords
                right_eye_x, right_eye_y = right_eye_coords
                calibrations.append((point, (left_eye_x, left_eye_y), (right_eye_x, right_eye_y)))
                break

            # Wyczyszczenie ekranu
            screen.fill(BG_COLOR)
            pygame.display.flip()
        screen.fill(BG_COLOR)
        pygame.display.flip()

    return calibrations


def create_calibration_function(calibration_data):
    left_eye_x = list()
    left_eye_y = list()
    right_eye_x = list()
    right_eye_y = list()
    x_cords = list()
    y_cords = list()
    for elem in calibration_data:
        left_eye_x.append(elem[1][0])
        left_eye_y.append(elem[1][1])
        right_eye_x.append(elem[2][0])
        right_eye_y.append(elem[2][1])
        x_cords.append(elem[0][0])
        y_cords.append(elem[0][1])

    degree = 1
    coefx = np.polyfit(left_eye_x, x_cords, degree)
    coefy = np.polyfit(left_eye_y, y_cords, degree)
    coefx2 = np.polyfit(right_eye_x, x_cords, degree)
    coefy2 = np.polyfit(right_eye_y, y_cords, degree)
    polyx_left = np.poly1d(coefx)
    polyy_left = np.poly1d(coefy)
    polyx_right = np.poly1d(coefx2)
    polyy_right = np.poly1d(coefy2)
    return polyx_left, polyy_left, polyx_right, polyy_right


def interpolate_calibration_data(calibration_data, left_eye_position, right_eye_position, *wielomiany):
    x_lewy = wielomiany[0](left_eye_position[0])
    y_lewy = wielomiany[1](left_eye_position[1])
    x_prawy = wielomiany[2](right_eye_position[0])
    y_prawy = wielomiany[3](right_eye_position[1])

    # Obróć o 180 stopni, jeśli to konieczne
    # x = SCREEN_WIDTH - x

    x = int((x_prawy + x_lewy) / 2)
    y = int((y_prawy + y_lewy) / 2)

    # print(x,y)

    return x, y


# Inicjalizacja eyetrackera
gaze: GazeTracking = GazeTracking()
webcam = cv2.VideoCapture(1)

# Kalibracja
calibration_data = calibrate(gaze, webcam)
wielomianx_lewy, wielomiany_lewy, wielomianx_prawy, wielomiany_prawy = create_calibration_function(calibration_data)

x_positions = []
y_positions = []

points_for_heatmap = []

x_average = 0
y_average = 0

while True:
    i = 0
    while i < 4:
        # Pobieramy nową klatkę z kamery
        _, frame = webcam.read()

        #frame = frame[150:300, 250:400]
        #frame = cv2.resize(frame, (1200, 1200))
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        gaze.refresh(frame)

        # Odczytujemy współrzędne gałek ocznych
        left_eye_coords = gaze.pupil_left_coords()
        right_eye_coords = gaze.pupil_right_coords()

        if left_eye_coords is not None and right_eye_coords is not None:
            left_eye_x, left_eye_y = left_eye_coords
            right_eye_x, right_eye_y = right_eye_coords
            # cv2.putText(frame, "Left pupil:  " + str(left_eye_coords), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            # cv2.putText(frame, "Right pupil: " + str(right_eye_coords), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31),)

            # Interpolacja danych kalibracyjnych dla lewego oka
            calibrated_eye_x, calibrated_eye_y = interpolate_calibration_data(calibration_data,
                                                                              (left_eye_x, left_eye_y),
                                                                              (right_eye_x, right_eye_y),
                                                                              wielomianx_lewy, wielomiany_lewy,
                                                                              wielomianx_prawy, wielomiany_prawy)
            x_positions.append(calibrated_eye_x)
            y_positions.append(calibrated_eye_y)
            i += 1

    x_average = sum(x_positions) / len(x_positions)
    y_average = sum(y_positions) / len(y_positions)

    if x_average>SCREEN_WIDTH:
        x_average = SCREEN_WIDTH
    if x_average<0:
        x_average=0

    if y_average>SCREEN_HEIGHT:
        y_average = SCREEN_HEIGHT
    if y_average<0:
        y_average=0

    points_for_heatmap.append((x_average, y_average))

    x_positions.clear()
    y_positions.clear()
    print(x_average, y_average)
    # Odświeżenie ekran
    # screen.fill(BG_COLOR)
    #draw_heatmap(x_average, y_average)
    draw_point(x_average, y_average, (0, 0, 255))
    pygame.display.flip()

    # cv2.imshow("Demo", frame)

    # Obsługa zdarzeń
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            draw_heatmap_from_points(points_for_heatmap)
            pygame.image.save(screen, 'zapisana_heatmapa.png')
            pygame.quit()
            webcam.release()
            cv2.destroyAllWindows()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            draw_heatmap_from_points(points_for_heatmap)