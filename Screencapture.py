# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 03:22:37 2018

@author: mandar
"""

import numpy as np
from PIL import ImageGrab
import cv2
import time


def proc_img(image):
    org_img=image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
    return processed_img
    
    
    
    
def main(): 
    last_time = time.time()
    while(True) :
        # 800x600 windowed mode
        screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        new_screen =proc_img(screen)
        cv2.imshow('window',new_screen)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
main()