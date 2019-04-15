# The file to define a game module which provides method to
# perform during the game
import time
from pyautogui import press, typewrite, hotkey, click, keyDown, keyUp
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

# The url of the game
url = 'chrome://dino'
# Set the below path to the location of chromedriver in your system
chrome_driver_path = "C:/webdrivers/chromedriver.exe"

class GameModule:
    # The constructor to initialize the game by creating browser
    # module and start the game
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        self._driver = webdriver.Chrome(chrome_driver_path)
        self._driver.maximize_window()
        self._driver.get(url)
        press('space')
        self._driver.execute_script("Runner.config.ACCELERATION=0")

    # Function to get if the game has crashed (Game-Over)
    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    # Function to get if the game is still running (Game not over)
    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")

    # Function to restart the game
    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")
        time.sleep(0.25)# no actions are possible

    # Function to press the 'up' arrow for dino to jump
    def press_up(self):
        press('up')

    # Function to hold the 'down' arror for dino to duck
    def press_down(self):
        keyDown('down')
        time.sleep(0.2)
        keyUp('down')

    # Function to get the current score of the game
    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)
        return int(score)

    # Function to close the browser and end the game at any state
    def end(self):
        self._driver.close()
