import webbrowser
import time

import cv2
import numpy as np
from PIL import Image
from PIL import ImageGrab
# from mss import mss
from pyautogui import press, typewrite, hotkey, click
import webbrowser


def open_game():
    url = 'chrome://dino'
    # MacOS
    #chrome_path = 'open -a /Applications/Google\ Chrome.app %s'

    # Windows
    chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'

    webbrowser.get(chrome_path).open(url)
    typewrite(url, interval=0.05)
    press('enter')
    press('space')


open_game()
