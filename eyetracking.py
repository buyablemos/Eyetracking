import cv2
from gaze_tracking import GazeTracking
import pygame
import sys
import numpy as np
import time
from scipy.interpolate import CubicSpline

# Inicjalizacja pygame
pygame.init()

# Ustawienia ekranu
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 800
BG_COLOR = (0, 0, 0)
POINT_COLOR = (255, 0, 0)
POINT_RADIUS = 10

# Inicjalizacja ekranu
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Eye Tracking Calibration")


# Funkcja do rysowania punktu na ekranie
def draw_point(x, y):
    pygame.draw.circle(screen, POINT_COLOR, (x, y), POINT_RADIUS)


# Lista punktów do kalibracji
CALIBRATION_POINTS = [
    (int(0.05 * SCREEN_WIDTH), int(0.05 * SCREEN_HEIGHT)),
    (int(0.5 * SCREEN_WIDTH), int(0.05 * SCREEN_HEIGHT)),
    (int(0.95 * SCREEN_WIDTH), int(0.05 * SCREEN_HEIGHT)),
    (int(0.05 * SCREEN_WIDTH), int(0.5 * SCREEN_HEIGHT)),
    (int(0.5 * SCREEN_WIDTH), int(0.5 * SCREEN_HEIGHT)),
    (int(0.95 * SCREEN_WIDTH), int(0.5 * SCREEN_HEIGHT)),
    (int(0.05 * SCREEN_WIDTH), int(0.95 * SCREEN_HEIGHT)),
    (int(0.5 * SCREEN_WIDTH), int(0.95 * SCREEN_HEIGHT)),
    (int(0.95 * SCREEN_WIDTH), int(0.95 * SCREEN_HEIGHT)),
]


# Funkcja do rysowania punktów kalibracyjnych
def draw_calibration_points(point_index):
    screen.fill(BG_COLOR)
    draw_point(*CALIBRATION_POINTS[point_index])
    pygame.display.flip()


# Procedura kalibracji
def calibrate(gaze, webcam):
    calibration_data = []
    for i, point in enumerate(CALIBRATION_POINTS):
        while True:
            draw_calibration_points(i)
            time.sleep(2)
            _, frame = webcam.read()
            #frame = frame[000:400, 200:500]
            frame = cv2.resize(frame, (1200, 1200))
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            gaze.refresh(frame)  # Pobieramy nową klatkę z eyetrackera
            left_eye_coords = gaze.pupil_left_coords()
            right_eye_coords = gaze.pupil_right_coords()
            if left_eye_coords is not None and right_eye_coords is not None:
                left_eye_x, left_eye_y = left_eye_coords
                right_eye_x, right_eye_y = right_eye_coords
                calibration_data.append((point, (left_eye_x, left_eye_y), (right_eye_x, right_eye_y)))
                break

            # Wyczyszczenie ekranu
            screen.fill(BG_COLOR)
            pygame.display.flip()
        screen.fill(BG_COLOR)
        pygame.display.flip()

    return calibration_data

def create_calibration_function(calibration_data):
    left_eye_x = list()
    left_eye_y= list()
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


    degree = 2
    coefx = np.polyfit(left_eye_x,x_cords,degree)
    coefy = np.polyfit(left_eye_y,y_cords,degree)
    coefx2 = np.polyfit(right_eye_x, x_cords, degree)
    coefy2 = np.polyfit(right_eye_y, y_cords, degree)
    polyx_left = np.poly1d(coefx)
    polyy_left = np.poly1d(coefy)
    polyx_right = np.poly1d(coefx2)
    polyy_right = np.poly1d(coefy2)
    return polyx_left, polyy_left, polyx_right, polyy_right


def interpolate_calibration_data(calibration_data, left_eye_position, right_eye_position,*wielomiany):
    # Oblicz granice kwadratu na podstawie punktów kalibracyjnych
    left_x_values, left_y_values = zip(*[calibration[1] for calibration in calibration_data])
    right_x_values, right_y_values = zip(*[calibration[2] for calibration in calibration_data])

    # Oblicz granice kwadratu dla lewego i prawego oka
    left_min_x, left_max_x = min(left_x_values), max(left_x_values)
    left_min_y, left_max_y = min(left_y_values), max(left_y_values)
    right_min_x, right_max_x = min(right_x_values), max(right_x_values)
    right_min_y, right_max_y = min(right_y_values), max(right_y_values)

    min_x = (left_min_x + right_min_x) / 2
    min_y = (left_min_y + right_min_y) / 2
    max_x = (left_max_x + right_max_x) / 2
    max_y = (left_max_y + right_max_y) / 2

    # Interpolacja liniowa na podstawie granic kwadratu
    x_lewy = wielomiany[0](left_eye_position[0])
    y_lewy = wielomiany[1](left_eye_position[1])
    x_prawy = wielomiany[2](right_eye_position[0])
    y_prawy = wielomiany[3](right_eye_position[1])

    # Obróć o 180 stopni, jeśli to konieczne
    #x = SCREEN_WIDTH - x

    x=int((x_prawy + x_lewy)/2)
    y=int((y_prawy + y_lewy) / 2)

    #x = np.interp(x, [min_x, max_x], [0, SCREEN_WIDTH])
    #y = np.interp(y, [min_y, max_y], [0, SCREEN_HEIGHT])

    #print(x,y)

    return x, y





# Inicjalizacja eyetrackera
gaze = GazeTracking()
webcam = cv2.VideoCapture(1)

# Kalibracja
calibration_data = calibrate(gaze, webcam)
wielomianx_lewy,wielomiany_lewy,wielomianx_prawy,wielomiany_prawy=create_calibration_function(calibration_data)



x_positions=[]
y_positions=[]

x_average=0
y_average=0

while True:
    i = 0
    while i < 3:
        # Pobieramy nową klatkę z kamery
        _, frame = webcam.read()

        #frame = frame[000:400, 200:500]
        frame = cv2.resize(frame, (1200, 1200))
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
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
            calibrated_eye_x, calibrated_eye_y = interpolate_calibration_data(calibration_data, (left_eye_x, left_eye_y),
                                                                              (right_eye_x, right_eye_y),wielomianx_lewy,wielomiany_lewy,wielomianx_prawy,wielomiany_prawy)
            x_positions.append(calibrated_eye_x)
            y_positions.append(calibrated_eye_y)
            i += 1

    x_average=sum(x_positions)/len(x_positions)
    y_average=sum(y_positions)/len(y_positions)
    x_positions.clear()
    y_positions.clear()
    print(x_average, y_average)
    # Odświeżenie ekran
    screen.fill(BG_COLOR)
    draw_point(x_average, y_average)
    pygame.display.flip()

    # cv2.imshow("Demo", frame)

    # Obsługa zdarzeń
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

# Zwalniamy zasoby
webcam.release()
cv2.destroyAllWindows()
