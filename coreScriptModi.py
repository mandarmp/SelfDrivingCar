import numpy as np
import cv2
import time
#import pyautogui
from directKeys import PressKey, ReleaseKey, W, A, S, D
from captureScreen import grab_screen
from getKeys import key_check
import os
from PIL import Image

from keras import backend as K

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


path = os.getcwd()
data_path = path + '/data'
data_dir_list = os.listdir(data_path)




def screen_record():
    lc = 0
    rc = 0
    sc = 0

    count = 0

    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

        # 800x600 windowed mode

    while(True):
        last_time = time.time()
        screen = grab_screen(region=(0, 400, 800, 600))
        #screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (128, 128))
        keys = key_check()
        keystrokes = presskeys_to_keystrokes(keys)
        #print("Frames capturing at {} fps".format(1.0 / time.time()-last_time))
        last_time = time.time()
        path = "C:/Users/ajeya/PycharmProjects/Self Driving Car/data"
        name = 'image'+str(count)+'.png'
        if 'A' in keys:
            lc += 1
            name = 'image'+str(lc)+'.png'
            print(keys)
            path = path + "/left"
            cv2.imwrite(os.path.join(path, name), screen)
        elif 'W' in keys:
            sc += 1
            name = 'image'+str(sc)+'.png'
            print(keys)
            path = path + "/straight"
            cv2.imwrite(os.path.join(path, name), screen)
        elif 'D' in keys:
            rc += 1
            name = 'image'+str(rc)+'.png'
            print(keys)
            path = path + "/right"
            cv2.imwrite(os.path.join(path, name), screen)
        else:
            print(keys)
            path = path + "/slow"
            pass
            #cv2.imwrite(os.path.join(path, name), screen)


        count += 1
        #new_img = cv2.resize(screen,(500,400))
        #backtorgb = cv2.cvtColor(new_screen, cv2.COLOR_GRAY2RGB)

        #cv2.imshow('Window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        cv2.imshow('window', screen)
        #cv2.imwrite(os.path.join(path,name), new_screen)


        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


screen_record()