# The main file to do the object detection from an already
# created detection model, use the detection to create
# environment of the game, and train a Q-learning model
# for setting up the weights of the model which then will
# be used to play the game as an AI.

import cv2 as cv
import tensorflow as tf
import numpy as np
from random import seed, randint, random
import GameModule
import DinoAgent
from pyautogui import press, typewrite, hotkey, click
import pyautogui

# Define ID for object detection (already trained in the object detection model)
DINO_CLASS_ID = 1
BIRD_CLASS_ID = 2
CACTUS_1_CLASS_ID = 3
CACTUS_2_CLASS_ID = 4
CACTUS_3_CLASS_ID = 5
CACTUS_4_CLASS_ID = 6
CACTUS_SMALL_CLASS_ID = 7
GAME_OVER_CLASS_ID = 8

# Variables for POSSIBLE MOVES (Duck not included
# as explained in the project document)
STAY = 0
JUMP = 1
POSSIBLE_MOVES = 2

# Steps to run for training
step = 500000
s = 0

# Hyperparameters for Reinforcement Learning (Q-Learning)
DISCOUNT_FACTOR = 0.1
LEARNING_RATE = 0.01

# Initialize environments
environment = [[0,0,0,0,0,0]]
new_environment = [[0,0,0,0,0,0]]

# Method to return the graph of trained model
def setup_detection_environment():
    # Read the graph.
    with tf.gfile.FastGFile('dino_chrome/frozen_inference_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.summary.FileWriter('logs' , graph_def)
        return graph_def

# Method to return a tensorflow session based
# on a given graph for detection
def start_session(graph_def):
    sess = tf.Session()
    sess.graph.as_default()
    tf.import_graph_def(g, name='')
    return sess

# Method to return an environment based on the object
# detected and it's location on the screen as parameter
def create_environment(obstacle):

    env = [[0,0,0,0,0,0]]
    if 300 <= obstacle[1] < 900:
        if 350 <= obstacle[0] < 400:
            env[0][0] = 1
        elif 400 <= obstacle[0] < 450:
            env[0][1] = 1
        elif 450 <= obstacle[0] < 500:
            env[0][2] = 1
        elif 500 <= obstacle[0] < 550:
            env[0][3] = 1
        elif 550 <= obstacle[0] < 600:
            env[0][4] = 1
        elif 600 <= obstacle[0] < 650:
            env[0][5] = 1

    return env

# Detector class to perform detection and return the coordinates of
# obstacle using the image and the tensorflow session
# This class used the already trained model for object detection
class Detector:

    def __init__(self):
        self.count = 0

    def run_detection(self, sess, frame):
        self.count = self.count + 1
        tensor_num_detections = sess.graph.get_tensor_by_name('num_detections:0')
        tensor_detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
        tensor_detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
        tensor_detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')

        frame = np.array(frame)
        img = frame
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (512, 512))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        out = sess.run([tensor_num_detections, tensor_detection_scores, tensor_detection_boxes, tensor_detection_classes],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

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
                    break

                cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), rect_color, thickness=2)

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
        return closest_obstacle_position

# Function for calculation of Q-value using Widrowhoff
def generate_Q_using_widrowhoff(environment,action):
    derivedQ = 0.0
    for j in range(1):
        for k in range(6):
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

# Set-up an instance of Game
game = GameModule.GameModule()

# Set-up an instance of DinoAgent and start the game
dino = DinoAgent.DinoAgent(game=game)

# Initialize the Weight matrix
Weight = np.zeros((2,1,6),dtype=float)

# Q-Learning implementation
while s < step:

    # Increment the counter after every step of training
    s+=1

    # Take screenshot of the game
    image = pyautogui.screenshot()
    # Pass image for obstacle detection
    obstacle = d.run_detection(sess=sess,frame=image)
    # Create the environment based on obstacle location
    new_environment = create_environment(obstacle)

    # Generate a random number to use for random moves during
    # initial stages of the training
    r = random()

    # Define a variable for the move selected, and maxQ for training
    SELECT_MOVE = 0
    maxQ = -100.0

    # Make a random move once every while to learn better
    if r < 0.5 and s < 300000:
        CURR_MOVE = randint(0, 1)
        SELECT_MOVE = CURR_MOVE

    # Select the move based on the environment
    else:
        for i in range(POSSIBLE_MOVES):
            qUsingWidrowhoff = generate_Q_using_widhrowhoff(environment,i)
            if qUsingWidrowhoff > maxQ:
                maxQ = qUsingWidrowhoff
                SELECT_MOVE = i

    # JUMP action
    if SELECT_MOVE == JUMP:
        dino.jump()

    # Reward based on game state
    reward = 0.0

    # Restarting a game if the dino crashes
    if dino.is_crashed():
        reward = -10.0      # Negative reward when Dino dies for an action
        game.restart()

    # Selecting maxQ corresponding to the environment and move
    bestmaxQ = -100.0
    for i in range(POSSIBLE_MOVES):
        bestNextQ = 0.0
        bestNextQ = generate_Q_using_widhrowhoff(new_environment,i)
        if bestNextQ > bestmaxQ:
            bestmaxQ = bestNextQ

    # Calculate the error
    idealQ = reward + (DISCOUNT_FACTOR * bestmaxQ)
    error = idealQ - generate_Q_using_widhrowhoff(environment,SELECT_MOVE)

    # Calculation to update the weight matrix
    for j in range(1):
        for k in range(6):
            if environment[j][k]==1:
                env_value = 1
            else:
                env_value = -1
            Weight[SELECT_MOVE][j][k] = Weight[SELECT_MOVE][j][k] + (LEARNING_RATE * error * env_value)

    # Saving the weight matrix after every 100 steps
    if s%100 == 0:
        np.save('Checkpoints/weight_matrix.npy', Weight)

    # Updating the environment with new environment
    environment = new_environment

# End the game
game.end()
