import pyautogui
import time

x=444
print('Beginning counter')
time.sleep(3)
while x<1000:
	pyautogui.screenshot('Screenshots_Train/'+str(x)+'.png')
	x+=1
	time.sleep(0.5)