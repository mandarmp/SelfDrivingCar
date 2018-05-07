# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 03:22:37 2018

@author: mandar
"""

import numpy as np
from PIL import ImageGrab
import cv2
import time
#import numpy as np
#from PIL import ImageGrabww
#import cv2
#import time
import random
import os
#wwwwwwwimport pyautogui
from directKeys import ReleaseKey, PressKey, W, A, S, D
from numpy import ones,vstack
from numpy.linalg import lstsq
from statistics import mean
from sklearn.cluster import KMeans
import threading
#import os

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
        straight()
        right()
    elif sign == 2:
        left()
        straight()
        left()
    else:
        rand=random.randint(0,7)
        if rand == 1 :
            straight()
        else :
            pass
    
def draw_lines(img,lines):
    #if lines is not None:
        #for line in lines:
            #coords = line[0]
            #cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)    
            #ajeya
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
    
def roi(img, vertices):
    #blank mask:
    mask = np.zeros_like(img)
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked

def proc_img(image):
    #kernel for morphological dilation
    kernel1 = np.ones((5,5),np.uint8)
    kernel2 = np.ones((1,1),np.uint8)
    kernel3 = np.ones((3,3),np.uint8)
    org_img=image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #to remove shadows
    processed_img = processed_img & 0b01111111
    #processed_img=cv2.blur(processed_img,(5,5))
    #processed_img = cv2.bilateralFilter(processed_img,9,75,75)
    processed_img = cv2.GaussianBlur(processed_img,(3,3),0)
    #vertices = np.array([[65,425],[65,365],[745,365],[745,425],], np.int32)
    #processed_img = roi(processed_img, [vertices])
     # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=210)
    #processed_img=cv2.dilate(processed_img,kernel1,iterations=1)
    #processed_img=cv2.erode(processed_img,kernel2,iterations=1)
    processed_img=cv2.dilate(processed_img,kernel1,iterations=1)
    processed_img=cv2.erode(processed_img,kernel3,iterations=1)
    #roi
    vertices = np.array([[183,420],[180,356],[627,356],[627,420],], np.int32)
    #processed_img = cv2.GaussianBlur(processed_img,(3,3),0)
    processed_img = roi(processed_img, [vertices])
    
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180,20,15)
    #if(lines != null)
    #print(lines)
    #draw_lines(processed_img,lines)
    try:
        if lines is not None :
            nlines = np.array([l[0] for l in lines])
            kmeans = KMeans(n_clusters=2, random_state=0).fit(nlines)
            draw_lines(processed_img, kmeans.cluster_centers_)
        else:
            rand=random.randint(0,3)
            if rand == 1 :
                straight()
            else :
                pass
    except (ValueError, TypeError) as e:
        print('Kmeans error: {}'.format(e))
    return processed_img
    
def main(): 
    last_time = time.time()
    #path = "D:/User/mandar/Documents/Python/images"
    while(True) :
        # 800x600 windowed mode
        #img=
        #time.sleep(2)
        screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        new_screen =proc_img(screen)
        #new_screen.save(os.getcwd()+"\\full_snap_" +str(int(time.time()))+".png","PNG")
        cv2.imshow('window',new_screen)
        #img.save('D:/User/mandar/Documents/Python/images','png')
        
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
main()