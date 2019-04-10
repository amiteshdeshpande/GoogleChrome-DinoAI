import time
from PIL import Image
import glob
import os
import pickle
import cv2 as cv
import tensorflow as tf
import numpy as np
from random import seed, randint, random
import GameModule
import DinoAgent
from pyautogui import press, typewrite, hotkey, click
import pyautogui
# from tempfile import TemporaryFile
# from src import LearningAgent

# Define ID for object detection
DINO_CLASS_ID = 1
BIRD_CLASS_ID = 2
CACTUS_1_CLASS_ID = 3
CACTUS_2_CLASS_ID = 4
CACTUS_3_CLASS_ID = 5
CACTUS_4_CLASS_ID = 6
CACTUS_SMALL_CLASS_ID = 7
GAME_OVER_CLASS_ID = 8

# Variables for POSSIBLE MOVES
STAY = 0
JUMP = 1
DUCK = 2
POSSIBLE_MOVES = 3
# Reward
reward = 0

# Steps to run for training
step = 250000
s = 0
# Hyperparameters for Reinforcement Learning (Q-Learning)
DISCOUNT_FACTOR = 0.3
LEARNING_RATE = 0.01

# Initialize environments
environment = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
new_environment = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

# Method to return the graph of trained model
def setup_detection_environment():
    # Read the graph.
    with tf.gfile.FastGFile('dino_chrome/frozen_inference_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.summary.FileWriter('logs' , graph_def)
        return graph_def

# Method to return a tensorflow session based on a given graph for detection
def start_session(graph_def):
    sess = tf.Session()
    sess.graph.as_default()
    tf.import_graph_def(g, name='')
    return sess

# Method to return an environment based on the object detection in a screenshot of game
def create_environment(obstacle):

    env = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    if 300 <= obstacle[1] < 900:
        if 200 <= obstacle[0] < 300:
            env[0][0] = 1
        elif 300 <= obstacle[0] < 400:
            env[0][1] = 1
        elif 400 <= obstacle[0] < 500:
            env[0][2] = 1
        elif 500 <= obstacle[0] < 600:
            env[0][3] = 1
        elif 600 <= obstacle[0] < 700:
            env[0][4] = 1
        elif 700 <= obstacle[0] < 800:
            env[0][5] = 1
        elif 800 <= obstacle[0] < 900:
            env[0][6] = 1
        elif 900 <= obstacle[0] < 1000:
            env[0][7] = 1
        elif 1000 <= obstacle[0] < 1100:
            env[0][8] = 1
        elif 1100 <= obstacle[0] < 1200:
            env[0][9] = 1
        elif 1200 <= obstacle[0] < 1300:
            env[0][10] = 1
        elif 1300 <= obstacle[0] < 1400:
            env[0][11] = 1
        elif 1400 <= obstacle[0] < 1500:
            env[0][12] = 1
        elif 1500 <= obstacle[0] < 1600:
            env[0][13] = 1
        elif 1600 <= obstacle[0] < 1700:
            env[0][14] = 1
        elif 1700 <= obstacle[0] < 1800:
            env[0][15] = 1

    return env

# Detector class to perform detection and return the coordinated of
# obstacle using the image and the tensorflow session
class Detector:

    def __init__(self):
        self.count = 0

    def run_detection(self, sess, frame):
        self.count = self.count + 1
        start_time = time.time()
        tensor_num_detections = sess.graph.get_tensor_by_name('num_detections:0')
        tensor_detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
        tensor_detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
        tensor_detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')
        # print('Initialize took {} seconds'.format(time.time() - start_time))

        # frame = cv.imread(frame)
        # start_time = time.time()
        frame = np.array(frame)
        img = frame
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (512, 512))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
        # print('Reading to np took {} seconds'.format(time.time() - start_time))
        # Run the model

        # start_time = time.time()
        out = sess.run([tensor_num_detections, tensor_detection_scores, tensor_detection_boxes, tensor_detection_classes],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
        # print('Run model took {} seconds'.format(time.time() - start_time))
        # Visualize detected bounding boxes.
        start_time = time.time()
        num_detections = int(out[0][0])
        dino_position = None
        obstacles = []
        is_game_over = False
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            if score > 0.3:
                bbox = [float(v) for v in out[2][0][i]]
                left = bbox[1] * cols
                top = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows

                rect_color = (255, 0, 0)
                if classId == DINO_CLASS_ID:
                    dino_position = (right, bottom)
                elif classId == CACTUS_SMALL_CLASS_ID or classId == CACTUS_1_CLASS_ID \
                        or classId == CACTUS_2_CLASS_ID or classId == CACTUS_3_CLASS_ID \
                        or classId == CACTUS_4_CLASS_ID or classId == BIRD_CLASS_ID:
                    obstacles.append((left, bottom))
                    rect_color = (0, 255, 0)
                elif classId == GAME_OVER_CLASS_ID:
                    is_game_over = True
                    rect_color = (0, 0, 255)
                    cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), rect_color, thickness=2)
                    # cv.imwrite('Screenshots/img' + str(self.count) + '.png', img)
                    break

                cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), rect_color, thickness=2)
                # cv.imwrite('Screenshots/img' + str(self.count) + '.png', img)

        closest_obstacle_position = [0,0]

        if not is_game_over:
            if dino_position is not None:
                min_dist = float("Inf")
                for obstacle_position in obstacles:
                    o_pos = obstacle_position[0]
                    d_pos = dino_position[0]
                    x_dist = o_pos - d_pos
                    if 10 <= x_dist < min_dist:
                        min_dist = x_dist
                        closest_obstacle_position = obstacle_position

        # print('Detection loop took {} seconds'.format(time.time() - start_time))
        # print('Dino position',dino_position)
        # print('Closest obstacle',closest_obstacle_position)
        # print('Game Over', is_game_over)
        return closest_obstacle_position

def generate_Q_using_widhrowhoff(environment,action):
    derivedQ = 0.0
    for j in range(1):
        for k in range(16):
            if environment[j][k]==1:
                env_value = 1
            else:
                env_value = -1
            derivedQ = derivedQ + (Weight[action][j][k]*env_value)
    return derivedQ


# Set-up detector, graph, and session instances
d = Detector()
g = setup_detection_environment()
sess = start_session(g)

# Set-up an instance of Game and DinoAgent
game = GameModule.GameModule()
# time.sleep(3)
dino = DinoAgent.DinoAgent(game=game)

Weight = np.load('Checkpoints/weight_matrix_noDuck_1x16.npy')
# Weight = np.load('Checkpoints/weight_test.npy')

# Q-Learning implementation
while s<step:
    image = pyautogui.screenshot()
    obstacle = d.run_detection(sess=sess,frame=image)
    new_environment = create_environment(obstacle)

    # print(environment)

    SELECT_MOVE = 0
    maxQ = -100
    for i in range(2):
        qUsingWidrowhoff = 0
        qUsingWidrowhoff = generate_Q_using_widhrowhoff(environment,i)
        print(qUsingWidrowhoff)
        if qUsingWidrowhoff > maxQ:
            maxQ = qUsingWidrowhoff
            SELECT_MOVE = i

    if SELECT_MOVE == JUMP:
        print("Jumped")
        dino.jump()

    if dino.is_crashed():
        score = game.get_score()
        #print('Score='+ (str)(score))
        f_in = open("Checkpoints/Highscore.pickle","rb")
        un = pickle.Unpickler(f_in)
        Highscore = un.load()
        f_in.close()
        if Highscore < score:
            f_out = open("Checkpoints/Highscore.pickle","wb")
            pickle.dump(score,f_out)
            f_out.close()
        game.restart()

    environment = new_environment
    # print('Run model took {} seconds'.format(time.time() - start_time))

game.end()
