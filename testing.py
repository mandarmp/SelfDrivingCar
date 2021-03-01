# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 23:34:33 2018

@author: ajeya
"""
import numpy as np
import cv2
import time
from directKeys import PressKey, ReleaseKey, W, A, S, D
from captureScreen import grab_screen
from getKeys import key_check
#import os
#from convNet import build_model
from keras.models import load_model
import random

n_rows = 128
n_cols = 128
alpha = 1e-3
n_epochs = 8
d_time = 0.09

def go_straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

def go_left():
    i = 5
    while(i<0):
        PressKey(A)
        i = i-1
        
    ReleaseKey(S)
    ReleaseKey(D)
    time.sleep(d_time)
    #ReleaseKey(S)

def go_right():
    i = 5
    while(i<0):
        PressKey(D)
        i = i-1
    #PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    time.sleep(d_time)
    
def go_reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def go_forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    
    
def go_forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)

    
def go_reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

    
def go_reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)

def no_keys():

    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    
trained_model = load_model("best_model_2.h5")

def screen_record():
    last_time = time.time()
    count = 0

    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

        # 800x600 windowed mode
    stopped = False
    while(True):
        
        if not stopped:
        
            screen = grab_screen(region=(0, 400, 800, 600))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (128, 128))
            
            pred = trained_model.predict([screen.reshape(1,128, 128, 1)])[0]
            moves = list(np.around(pred))
#            moves = np.array(pred)*np.array([4.5, 0.1, 0.1])
#            moves = np.argmax(moves)
            print("Move {0} with prediction {1}".format(moves, pred))
            
            turn_prob = 0.75
            no_turn_prob = 0.70
#            if pred[1] > no_turn_prob:
#                go_straight()
#            elif pred[0] > turn_prob:
#                go_left()
#            elif pred[2] > turn_prob:
#                go_right()
#            else:
#                go_straight()
#            if moves == 0:
#                go_straight()
#                choice_picked = 'straight'
#                
#            elif moves == 1:
#                go_reverse()
#                choice_picked = 'reverse'
#                w
#            elif moves == 2:
#                go_left()
#                choice_picked = 'left'
#            elif moves == 3:
#                go_right()
#                choice_picked = 'right'
#            elif moves == 4:
#                go_forward_left()
#                choice_picked = 'forward+left'
#            elif moves == 5:
#                go_forward_right()
#       sss         choice_picked = 'forward+right'
#            elif moves == 6:
#                go_reverse_left()
#                choice_picked = 'reverse+left'
#            elif moves == 7:
#                go_reverse_right()
#                choice_picked = 'reverse+right'
#            elif moves == 8:
#                no_keys()
#                choice_picked = 'nokeys'
            
            if moves==[1, 0 ,0]:
                go_left()
            elif moves==[0, 1, 0]:
                go_straight()
            elif moves==[0, 0, 1]:
                go_right()
            else:
                go_straight()
                pass
            key = key_check()
            
            if 'T' in key:
                if stopped:
                    stopped = False
                    time.sleep(1)
                else:
                    stopped = True
                    ReleaseKey(A)
                    ReleaseKey(W)
                    ReleaseKey(D)
        #print("Frames capturing at {} fps".format(time.time()-last_time))
        last_time = time.time()
        

screen_record()