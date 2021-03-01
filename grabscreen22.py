# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 16:04:36 2018

@author: mandar
"""

import cv2
import numpy as np
import pywin32
#import pywin32, pywin32, win32con, pywin32


def grab_screen(region=None):
    hwin = pywin32.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = pywin32.GetSystemMetrics(78)
        height = pywin32.GetSystemMetrics(79)
        left = pywin32.GetSystemMetrics(76)
        top = pywin32.GetSystemMetrics(77)

    hwindc = pywin32.GetWindowDC(hwin)
    srcdc = pywin32.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = pywin32.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top),S)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    pywin32.ReleaseDC(hwin, hwindc)
    pywin32.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)