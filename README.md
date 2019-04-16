# GoogleChrome-DinoAI

As a part of the Graduate course Game Artificial Intelligence CS 5150 at Khoury College of Computer Sciences, Northeastern University, my group partner and I worked on the project to develop an AI using Q-Learning model to learn to play the Dino-Chrome game.

[![VIDEO](https://img.youtube.com/vi/oy5_0C4n98Q/0.jpg)](https://v637g.app.goo.gl/9eWb6WFFLxmiy36cA)

# Introduction
We started with the idea to train an AI that plays the T-Rex Run (Google Chrome Dino game) using Q-learning. <br />
The final state of the project consists of a two-fold implementation. First, we use an Object-detection model which does real-time screen capture and then gives out the coordinates of the obstacle on the screen. These coordinates are used to create an environment which is then used to train the Q-learning model. We have used the Widrow-Hoff Learning rule in the Q-learning model. 

# Team
The team members were: Himanshu Budhia (https://github.com/budhiahimanshu96/) and Amitesh Deshpande (https://github.com/amiteshdeshpande/). 

# Instructions
•	Set-up TensorFlow and object detection environment in the system (GPU preferred). You may follow this link: https://bit.ly/2UWmIOb <br />
•	Install PyAutoGUI and selenium libraries using pip install <br />
•	Once this is installed and tested, clone or download the project repository <br />
•	There is already an object-detection model trained for our game in the ‘dino_chrome’ folder with the file ‘frozen_inference_graph.pb’. This file will be now used while playing the game. <br />
•	Chrome driver also needs to be installed for the next steps. It can be found here: http://chromedriver.chromium.org/downloads <br />
•	After installing, you need to set the chrome driver executable path in the file: ‘GameModule.py’. <br />
•	Now, there is also a trained weights file for the Q-Learning model from our implementation available in the ‘Checkpoints’ directory with name ‘weight_matrix.npy’. <br />
•	To run the implementation, open terminal and run the ‘Test.py’ file (make sure the to use the TensorFlow environment installed in the step 1). 

# Systems
In the project, the main implementation is of the Q-Learning model using the Widrow-Hoff rule. To implement this model, we used the technique and steps taught in the class. <br />
Q learning update rule:<br />
Q (a, s) = (1-α) Q (a, s) + α (r + γ max<sub>a’</sub> Q (a’, s’))<br />
where,<br />
	α is the learning rate (0.01)<br />
	γ is the discount factor (0.1)<br />
	r is the reward (-10 for death)<br />
	max<sub>a’</sub> Q(a’s’) is the best action’s Q-value in the state reached from the previous state. <br />

Using Widrow-Hoff rule our q-functions are of form:
Q (a, s) = Σ w<sub>aijvij</sub> <br />

The most important design decisions to take were the creation of the environment for the Q-Learning model, and the reward system which will guide the AI towards learning how to play the game. The environment was created based on the screen capture and the location of the obstacle detected on the screen. For e.g. the object detection model provides with the coordinates of the bottom left corner of bounding box on the first obstacle detected in front of the Dino. The environment is just a matrix with the value of an element as 0 (if obstacle absent) or 1 (if obstacle present). The acceleration of dino was set to be 0 so that the game speed does not hinder with object detection process and to maintain a balance between them.

# Failures and Setbacks

One of the most difficult part of the project was the set-up of environment. There were a lot of dependencies to take care of for the installation of object detection library. We had to consult many different tutorials and videos for this to work. Even after the set-up, the training of the model took nearly 2 complete days. We were able to achieve good results for the object detection as shown in the document: object_detection_result.pdf. <br />

When this model was trained, our next goal was to detect object in real-time. Initially, when we implemented this real-time, the average time to detect objects in each image was nearly 4s which was not feasible for the further work. We worked on this by making some changes in the conversion of image to NumPy arrays. This helped us do the detection in 3s which was still not good enough. After scratching our heads, we realised that we were making a new TensorFlow session for each image, which was itself taking a lot of time. Finally, after fixing the issue, we were able to get the real-time object detection time for our game to as low as 10ms. <br />

Next, we faced some major setbacks when we started to implement Q-learning model. The implementation was straight forward but the training results were not as expected. Initially, we had a 4x4 environment and a reward system based on both positive and negative rewards. This did not give good results. The weight matrix was very random for every step trained. We also tried making the reward only negative (when the dino died) which did improve the results sometimes but overall, this was a failure. <br />

On further brainstorming, we thought the reason for poor results could be the environment size. The 4x4 environment had many states which were never going to affect the action of dino (like the region exactly above the dino). So, we tried a new environment of dimension 1x16. But still the results were poor (trying different permutations of rewards as well). We also found that we the variable ‘maxQ’ and ‘bestmaxQ’ should be set to a negative value. The reason being our q values of environment were being calculated as negative, so when we initially set ‘maxQ’ and ‘bestmaxQ’ to 0, the best move was always being selected as ‘do nothing’. <br />

We now decided to work with just the 1x1 environment which would be right in-front of the dino to make sure if it able to understand the environment and our understanding is correct or not. This approach gave very good results. We were able to achieve a high score of approx. 300, which gave us some hope to work towards refining the result further. <br />

In the end, we realised that the ‘ducking’ feature of the dino was unnecessary (the dino can jump instead of ducking). We changed the number of possible actions to only 2 now (jump or do nothing). We also finalised on a 1x6 environment with every 1x1 block maintaining the state of 50 pixels of the screen. <br />

# Success

After the final implementation, we trained the model for 500000 steps. We obtained a very good weight matrix. On testing the game using these weights, we observed a high score of 706 for the game. <br />

The game changes from day time to night time and the object detection was trained for both the scenarios. But we did not factor in the transition time between them. So, the object detection seemingly fails here because every time there is a transition, we can see the dino dying. <br />

As some future work, we can see this being taken forward to be trained over more sets of states in terms of object detection. There is also a possibility to make the model completely based on CNN. We could pass the entire image as the input and the model will learn based on the truth values of the labelled images.<br />


# Evaluation

Our experimentation with AI was based on trying different rewards and environments. We tried a combination of 4x4 environment, 1x16 environment, 1x1 environment, and 1x6 environment with negative reward (only on dino crashing), both positive-negative reward (whole numbers), and both positive-negative reward based on current game score. <br />

| Environment | Max Score |
| --- | --- |
| 4x4 | 175 |
| 1x16 | 230 |
| 1x1 | 320 |
| 1x6 | 706 |

# Lessons Learned

We were able to understand the object detection using TensorFlow and dependencies management in the installation of the same. We understood the importance of the environment in the Q-learning model and how it impacts the correctness of training. Also, we realised that the Q-learning is very much dependant on the game or application (we initially implemented model exactly as Frogger in assignment 5, which essentially failed). We were able to appreciate the role of reward in the learning of weights for q-learning model (the rewards need to be adjusted such that the agent gets prize only when necessary and not for all small/wrong actions).

# References

[1]"Build an AI to play Dino Run", Hello Paperspace, 2019. [Online]. Available: https://blog.paperspace.com/dino-run/. <br />
[2]"Installation — TensorFlow Object Detection API tutorial documentation", Tensorflow-object-detection-api-tutorial.readthedocs.io, 2019. [Online]. Available: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html. <br />
[3]"Training Custom Object Detector — TensorFlow Object Detection API tutorial documentation", Tensorflow-object-detection-api-tutorial.readthedocs.io, 2019. [Online]. Available: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html. 
