import numpy as np
import cv2
import time
#import pyautogui
from directKeys import PressKey, ReleaseKey, W, A, S, D
from captureScreen import grab_screen
from getKeys import key_check
import os

def presskeys_to_keystrokes(keys):
    #[A, W, D]

    keystrokes = [0, 0, 0]

    if 'A' in keys:
        keystrokes[0] = 1
    elif 'D' in keys:
        keystrokes[2] = 1
    else:
        keystrokes[1] = 1

    return keystrokes

count = 1
data_file = "training_data.npy"
print(len(data_file))

if os.path.isfile(data_file):
    print("file exists, data generation started from previous data")
    training_data = list(np.load(data_file))
else:
    print("file doesnt exist, data generation started from scratch")
    training_data = []




def screen_record():


    count = 0

    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

        # 800x600 windowed mode

    while(True):
        last_time = time.time()
        screen = grab_screen(region=(0, 40, 800, 600))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (128, 128))
        keys = key_check()
        keystrokes = presskeys_to_keystrokes(keys)
        training_data.append([screen, keystrokes])
        #print("Frames capturing at {} fps".format(1.0 / time.time()-last_time))
        last_time = time.time()
        path = "C:/Users/ajeya/PycharmProjects/Self Driving Car/images"
        name = 'image'+str(count)+'.png'
        count += 1
        #new_img = cv2.resize(screen,(500,400))
        #backtorgb = cv2.cvtColor(new_screen, cv2.COLOR_GRAY2RGB)

        #cv2.imshow('Window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        cv2.imshow('window', screen)
        #cv2.imwrite(os.path.join(path,name), new_screen)
        if len(training_data) % 400 == 0:
            print(len(training_data))
            np.save(data_file, training_data)


        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


screen_record()