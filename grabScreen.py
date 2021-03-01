import numpy as np
from PIL import ImageGrab
import cv2
import time
import os
import pyautogui
from directKeys import ReleaseKey, PressKey, W, A, S, D
from numpy import ones,vstack
from numpy.linalg import lstsq
from statistics import mean
from sklearn.cluster import KMeans
import threading

keytime = 0.1
STRAIGHT = 0x11
RIGHT = 0x20
LEFT = 0x1E

def t_key(key_a, key_b, key_c):
    PressKey(key_a)
    ReleaseKey(key_b)
    ReleaseKey(key_c)
    time.sleep(keytime)


def straight():
    thread_straight = threading.Thread(target=t_key,
                                    args=(STRAIGHT, RIGHT, LEFT))
    thread_straight.start()


def right():
    thread_right = threading.Thread(target=t_key,
                                    args=(RIGHT, STRAIGHT, LEFT))
    thread_right.start()


def left():
    thread_left = threading.Thread(target=t_key,
                                   args=(LEFT, STRAIGHT, RIGHT))
    thread_left.start()


def slope(line):
    try:
        y = line[1] - line[3]
        x = line[0] - line[2]
        slope = np.divide(y, x)
    except ZeroDivisionError:
        slope = 100000
    finally:
        return slope

def drive(m=None):
    sign = np.sum(np.sign(m))
    if sign == -2:
        right()
    elif sign == 2:
        left()
    else:
        straight()


def draw_lines(img, lines):
    try:
        m = []
        for coords in lines:
            m.append(slope(coords))
            coords = np.array(coords, dtype='uint32')
            cv2.line(img,
                     (coords[0], coords[1]),
                     (coords[2], coords[3]),
                     [255, 255, 255], 20)
    except TypeError as e:
        print('draw lines error: {}'.format(e))
    else:
        pass
        drive(m)



def region_of_intrest(img, coords):
    mask = np.zeros_like(img)
    #print(img.shape)
    cv2.fillPoly(mask, coords, 255)
    roi = cv2.bitwise_and(img, mask)
    return roi

def process_image(orig_img):
    coords = np.array([[0, 600], [0, 400], [200, 200], [600, 200], [800, 400], [800, 600]])
    filter_1 = np.ones((3, 3), np.uint8)
    filter_2 = np.ones((5, 5), np.uint8)
    #process_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    process_img = cv2.Canny(orig_img,
                              threshold1=100, threshold2=300)

    process_img = region_of_intrest(process_img, [coords])

    process_img = cv2.GaussianBlur(process_img, (5, 5), 0)
    lines = cv2.HoughLinesP(process_img, 1,
                            np.pi / 180, 180, np.array([]), 100, 50)
    # nlines = np.array([l[0] for l in lines])
    # draw_lines(process_img, nlines)

    #process_img = process_img & 0b01111111

    #process_img = cv2.GaussianBlur(process_img, (5, 5), 0)
    # process_img = cv2.Canny(process_img, threshold1=200, threshold2=210)
    # process_img = cv2.dilate(process_img, filter_1)
    #coords = np.array([[[10, 500], [10, 300], [300, 300], [500, 300], [800, 300], [800, 500]]])
    #coords = np.array([[[10, 600], [200, 300], [700, 300], [700, 600]]])
    #  working set of coordinates 1
    #coords = np.array([[0,600], [0, 400], [200, 200], [600, 200], [800, 400], [800, 600]])

    # working set of coordinates 2
    #coords = np.array([[0, 600], [200, 300], [700, 300], [800, 600]])
    #coords = np.array([[130, 700], [170, 500], [715, 500], [700, 750]])

    # best coords so far
    #coords = np.array([[0, 600], [0, 400], [800, 400], [800, 600]])

    #coords = np.array([[250, 780], [400, 430], [550, 430], [730, 780]])

    #process_img = region_of_intrest(process_img, [coords])


    #lines = cv2.HoughLinesP(process_img, 1, np.pi/180, 180,np.array([]), 20, 15)
    # #draw_lines(process_img, lines)
    try:
        nlines = np.array([l[0] for l in lines])
        kmeans = KMeans(n_clusters=2, random_state=0).fit(nlines)
        draw_lines(process_img, kmeans.cluster_centers_)
    except (ValueError, TypeError) as e:
        print('Kmeans error: {}'.format(e))


    return process_img

def screen_record():
    last_time = time.time()
    count = 0

    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

        # 800x600 windowed mode

    while(True):

        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
        #new_screen, original_image = process_image(screen)
        # cv2.imshow('window', new_screen)
        # cv2.imshow('window2', cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        new_screen = process_image(screen)

        #new_screen = compute_binary_image(screen)

        #PressKey(W)
        path = "C:/Users/ajeya/PycharmProjects/Self Driving Car/images"
        name = 'image'+str(count)+'.png'
        count += 1
        #new_img = cv2.resize(screen,(500,400))
        #backtorgb = cv2.cvtColor(new_screen, cv2.COLOR_GRAY2RGB)

        #cv2.imshow('Window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        cv2.imshow('window', new_screen)
        #cv2.imwrite(os.path.join(path,name), new_screen)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break












screen_record()