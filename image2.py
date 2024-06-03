import cv2
import numpy as np
def create_opacity_map_on_screen():
    # Load heatmap and screen images
    heatmap = cv2.imread('zapisana_heatmapa.png')
    screen = cv2.imread('full_page_screenshot.png')

    # Resize heatmap to match screen size if necessary
    heatmap = cv2.resize(heatmap, (screen.shape[1], screen.shape[0]))

    # Convert heatmap to grayscale
    heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)

    # Threshold the heatmap to create a binary mask
    _, mask = cv2.threshold(heatmap_gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)

    # Apply the mask to the screen image to retain only non-white pixels from the heatmap
    result = cv2.bitwise_and(screen, screen, mask=mask)

    # Set the rest of the pixels to white
    result[mask == 0] = [255, 255, 255]

    # Save the resulting image to a file
    cv2.imwrite('opacity_map.jpg', result)

    # Display the result
    #cv2.imshow('Result', result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


