import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui


def process_img(image):
    original_image = image
    # convert to opencv
    processed_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    #processed_img = cv2.resize(processed_img, (0, 0), fx=0.5, fy=0.5)
    # convert to gray
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)

    return processed_img


def main():
    num=0
    while num < 500:
        screen = pyautogui.screenshot()
        new_screen = process_img(screen)
        cv2.imshow('img',new_screen)
        cv2.imwrite('./test_images/img' + str(num) + '.png', new_screen)
        num += 1


main()
