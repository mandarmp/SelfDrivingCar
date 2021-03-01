import time
import cv2
from captureScreen import grab_screen

while True:
    start_time = time.time() # start time of the loop

    ########################
    # your fancy code here #
    ########################
    screen = grab_screen(region=(0, 40, 800, 600))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen = cv2.resize(screen, (128, 128))

    print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
    cv2.imshow('window', screen)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

