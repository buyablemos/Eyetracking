import cv2
from gaze_tracking import GazeTracking
import pygame
import sys
import numpy as np
import time

# Inicjalizacja pygame
pygame.init()

# Ustawienia ekranu
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BG_COLOR = (255, 255, 255)
POINT_COLOR = (255, 0, 0)
POINT_RADIUS = 10

frame_width = 800
frame_height = 600
desired_width = 800
desired_height = 600
x = (frame_width - desired_width) // 2
y = (frame_height - desired_height) // 2

# Lista punktów do kalibracji
CALIBRATION_POINTS = [(100, 100), (700, 100), (400, 300), (100, 500), (700, 500)]

# Inicjalizacja ekranu
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Eye Tracking Calibration")

# Funkcja do rysowania punktu na ekranie
def draw_point(x, y):
    pygame.draw.circle(screen, POINT_COLOR, (x, y), POINT_RADIUS)

# Funkcja do rysowania punktów kalibracyjnych
def draw_calibration_points(point_index):
    screen.fill(BG_COLOR)
    draw_point(*CALIBRATION_POINTS[point_index])
    pygame.display.flip()

# Procedura kalibracji
def calibrate(gaze, webcam):
    calibration_data = []
    for i, point in enumerate(CALIBRATION_POINTS):
        draw_calibration_points(i)
        while True:
            _, frame = webcam.read()  # Pobieramy nową klatkę z kamery
            frame = frame[y:y + desired_height, x:x + desired_width]
            gaze.refresh(frame)  # Pobieramy nową klatkę z eyetrackera
            eye_coords = gaze.pupil_left_coords()
            if eye_coords is not None:
                eye_x, eye_y = eye_coords
                calibration_data.append((point, (eye_x, eye_y)))
                break


            # Wyczyszczenie ekranu
            screen.fill(BG_COLOR)
            pygame.display.flip()
        time.sleep(5)
    return calibration_data

# Funkcja do interpolacji liniowej na podstawie danych kalibracyjnych
import numpy as np


import numpy as np

def interpolate_calibration_data(calibration_data, eye_position):
    # Oblicz granice kwadratu na podstawie punktów kalibracyjnych
    x_values, y_values = zip(*[calibration[0] for calibration in calibration_data])
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    # Interpolacja liniowa na podstawie granic kwadratu
    x = np.interp(eye_position[0], [min_x, max_x], [0, 1])
    y = np.interp(eye_position[1], [min_y, max_y], [0, 1])

    # Skalowanie interpolowanych wartości do rozmiarów ekranu
    x_screen = int(x * SCREEN_WIDTH)
    y_screen = int(y * SCREEN_HEIGHT)

    return x_screen, y_screen



# Inicjalizacja eyetrackera
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

# Kalibracja
calibration_data = calibrate(gaze, webcam)

while True:
    # Pobieramy nową klatkę z kamery
    _, frame = webcam.read()
    frame = frame[y:y + desired_height, x:x + desired_width]
    gaze.refresh(frame)
    frame = gaze.annotated_frame()

    # Przesyłamy klatkę do eyetrackera w celu analizy


    # Odczytujemy współrzędne lewego oka
    eye_coords = gaze.pupil_left_coords()
    if eye_coords is not None:
        eye_x, eye_y = eye_coords
        # Tutaj możemy kontynuować działania na współrzędnych oka
    else:
        print("Nie wykryto źrenicy dla lewego oka.")

    # Wyczyszczenie ekranu
    screen.fill(BG_COLOR)

    # Interpolacja danych kalibracyjnych
    calibrated_eye_x, calibrated_eye_y = interpolate_calibration_data(calibration_data, (eye_x, eye_y))

    # Rysowanie punktu tam, gdzie patrzy użytkownik
    draw_point(calibrated_eye_x, calibrated_eye_y)
    cv2.imshow("Demo", frame)
    # Odświeżenie ekranu
    pygame.display.flip()

    # Obsługa zdarzeń
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

# Zwalniamy zasoby
webcam.release()
cv2.destroyAllWindows()
