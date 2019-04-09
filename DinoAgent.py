import numpy as np
import webbrowser
from PIL import ImageGrab #grabbing image
from PIL import Image
import cv2 #opencv
import io
import time
from pyautogui import press, typewrite, hotkey, click
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (30, 30)
#import seaborn as sns
import pandas as pd
from random import randint
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
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

class DinoAgent:
    def __init__(self,game): #takes game as input for taking actions
        self._game = game;
        self.jump(); #to start the game, we need to jump once
        time.sleep(.5) # no action can be performed for the first time when game starts
    def is_running(self):
        return self._game.get_playing()
    def is_crashed(self):
        return self._game.get_crashed()
    def jump(self):
        self._game.press_up()
    def duck(self):
        self._game.press_down()
