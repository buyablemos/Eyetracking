"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking

frame_width = 1280
frame_height = 720
desired_width = 640
desired_height = 720
x = (frame_width - desired_width) // 2
y = (frame_height - desired_height) // 2

gaze = GazeTracking()

#-------!!!--------
#Choose your camera
#-------!!!--------
webcam = cv2.VideoCapture(0)

while True:
    _, frame = webcam.read()

    #-------!!!-------
    # Rotate if needed
    #-------!!!------- 
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
   
webcam.release()
cv2.destroyAllWindows()
