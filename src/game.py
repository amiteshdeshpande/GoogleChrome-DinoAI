import webbrowser
import time

import cv2
import numpy as np
from PIL import Image
from PIL import ImageGrab
# from mss import mss
from pyautogui import press, typewrite, hotkey, click


def open_game():
    url = 'chrome://dino'
    # MacOS
    hotkey('command', 'space')
    typewrite('Google Chrome')
    hotkey('enter')
    time.sleep(0.5)
    hotkey('command', 'l')
    typewrite(url, interval=0.01)
    press('enter')
    press('space')


open_game()
