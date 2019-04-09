import numpy as np
import webbrowser
from PIL import ImageGrab #grabbing image
from PIL import Image
import cv2 #opencv
import io
import time
from pyautogui import press, typewrite, hotkey, click, keyDown, keyUp
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (30, 30)
#import seaborn as sns
import pandas as pd
from random import randint
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
#keras imports
#%matplotlib inline
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
from keras.callbacks import TensorBoard
from collections import deque
import random
import pickle
import json

url = 'chrome://dino'
chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
chrome_driver_path = "C:/webdrivers/chromedriver.exe"

class GameModule:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        self._driver = webdriver.Chrome(chrome_driver_path)
        self._driver.maximize_window()
        self._driver.get(url)
        press('space')
        self._driver.execute_script("Runner.config.ACCELERATION=0")
    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")

        time.sleep(0.25)# no actions are possible
                        # for 0.25 sec after game starts,
                        # skip learning at this time and make the model wait

    def press_up(self):
        press('up')
        # time.sleep(0.35)

    def press_down(self):
        keyDown('down')
        time.sleep(0.2)
        keyUp('down')

    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array) # the javascript object is of type array with score in the formate[1,0,0] which is 100.
        return int(score)

    def end(self):
        self._driver.close()

# game = GameModule()
# time.sleep(3)
