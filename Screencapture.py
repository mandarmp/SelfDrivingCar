# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 03:22:37 2018

@author: mandar
"""

import numpy as np
from PIL import ImageGrab
import cv2
import time
#import os



    
def draw_lines(img,lines):
    for line in lines:
        coords = line[0]
        cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)    
    
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
    kernel2 = np.ones((3,3),np.uint8)
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
    processed_img=cv2.dilate(processed_img,kernel1,iterations=1)
    processed_img=cv2.erode(processed_img,kernel2,iterations=1)
    #roi
    vertices = np.array([[65,425],[65,365],[745,365],[745,425],], np.int32)
    #processed_img = cv2.GaussianBlur(processed_img,(3,3),0)
    processed_img = roi(processed_img, [vertices])
    
    #lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180,      20,         15)
    #draw_lines(processed_img,lines)
    return processed_img
    
def main(): 
    last_time = time.time()
    #path = "D:/User/mandar/Documents/Python/images"
    while(True) :
        # 800x600 windowed mode
        #img=
        
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