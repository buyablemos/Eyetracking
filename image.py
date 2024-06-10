import cv2
import numpy as np


def create_heatmap_on_screen():
    screenshot = cv2.imread('full_page_screenshot.png')
    heatmap = cv2.imread('zapisana_heatmapa.png')

    heatmap_resized = cv2.resize(heatmap, (screenshot.shape[1], screenshot.shape[0]))

    alpha = 0.5

    blended_image = cv2.addWeighted(screenshot, 1 - alpha, heatmap_resized, alpha, 0)

    cv2.imwrite('heatmap_screen.png', blended_image)
