import time

import cv2 as cv
import tensorflow as tf

# from src import LearningAgent

DINO_CLASS_ID = 1
CACTUS_SMALL_CLASS_ID = 2
CACTUS_1_CLASS_ID = 3
CACTUS_2_CLASS_ID = 4
CACTUS_3_CLASS_ID = 5
CACTUS_4_CLASS_ID = 6
BIRD_CLASS_ID = 7
GAME_OVER_CLASS_ID = 8


def setup_detection_environment():
    # Read the graph.
    with tf.gfile.FastGFile('./src/edge_detection_inference_graph/frozen_inference_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.summary.FileWriter('logs' , graph_def)
        return graph_def


def start_session(graph_def):
    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        return sess


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

        img = frame
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (512, 512))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
        # Run the model

        out = sess.run([tensor_num_detections, tensor_detection_scores, tensor_detection_boxes, tensor_detection_classes],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        # Visualize detected bounding boxes.
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
                    cv.imwrite('screenshots/img' + str(self.count) + '.png', img)
                    break

                cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), rect_color, thickness=2)
                cv.imwrite('screenshots/img' + str(self.count) + '.png', img)

        closest_obstacle_position = None

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

        print('Detection loop took {} seconds'.format(time.time() - start_time))
        print('Dino position',dino_position)
        print('Closest obstacle',closest_obstacle_position)
        print('Game Over', is_game_over)
