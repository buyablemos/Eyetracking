import cv2
import numpy as np


def create_heatmap_on_screen():
    # Load the screenshot and heatmap images
    screenshot = cv2.imread('full_page_screenshot.png')
    heatmap = cv2.imread('zapisana_heatmapa.png')

    # Resize the heatmap to match the dimensions of the screenshot
    heatmap_resized = cv2.resize(heatmap, (screenshot.shape[1], screenshot.shape[0]))

    # Set the alpha value for blending (adjust as needed)
    alpha = 0.5

    # Blend the images
    blended_image = cv2.addWeighted(screenshot, 1 - alpha, heatmap_resized, alpha, 0)

    # Save or display the result
    cv2.imwrite('heatmap_screen.png', blended_image)
    # Or display the result
    #cv2.imshow('Blended Image', blended_image)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
