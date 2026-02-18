from config import BOARD_SIZE, categories, image_size
from tensorflow.keras import models
import numpy as np
import tensorflow as tf
import os
import cv2
import random
from matplotlib import pyplot as plt
from matplotlib.image import imread

class TicTacToePlayer:
    def get_move(self, board_state):
        raise NotImplementedError()

class UserInputPlayer:
    def get_move(self, board_state):
        inp = input('Enter x y:')
        try:
            x, y = inp.split()
            x, y = int(x), int(y)
            return x, y
        except Exception:
            return None

class RandomPlayer:
    def get_move(self, board_state):
        positions = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board_state[i][j] is None:
                    positions.append((i, j))
        return random.choice(positions)

class UserWebcamPlayer:
    def __init__(self):
        # Load the most recent model from results directory once
        results_dir = 'results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        model_files = [f for f in os.listdir(results_dir) if f.endswith('.keras')]
        if not model_files:
            print("No model found in results directory. Please train the model first.")
            self.model = None
        else:
            # Sort by timestamp to get the latest model
            model_files.sort()
            latest_model_path = os.path.join(results_dir, model_files[-1])
            print(f"Loading model for controller: {latest_model_path}")
            self.model = models.load_model(latest_model_path)

    def _process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        width, height = frame.shape
        size = min(width, height)
        pad = int((width-size)/2), int((height-size)/2)
        frame = frame[pad[0]:pad[0]+size, pad[1]:pad[1]+size]
        return frame

    def _access_webcam(self):
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)
        frame = None
        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
            if rval:
                frame = self._process_frame(frame)
        else:
            rval = False
            
        while rval:
            cv2.imshow("preview", frame)
            rval, frame = vc.read()
            if not rval:
                break
            frame = self._process_frame(frame)
            key = cv2.waitKey(20)
            if key == 13: # exit on Enter
                break

        vc.release()
        cv2.destroyWindow("preview")
        return frame

    def _print_reference(self, row_or_col):
        print('reference:')
        for i, emotion in enumerate(categories):
            print('{} {} is {}.'.format(row_or_col, i, emotion))
    
    def _get_row_or_col_by_text(self):
        try:
            val = int(input())
            return val
        except Exception as e:
            print('Invalid position')
            return None
    
    def _get_row_or_col(self, is_row):
        try:
            row_or_col = 'row' if is_row else 'col'
            self._print_reference(row_or_col)
            img = self._access_webcam()
            if img is None:
                print("Failed to capture image from webcam.")
                return self._get_row_or_col_by_text()
                
            emotion = self._get_emotion(img)
            if type(emotion) is not int or emotion not in range(len(categories)):
                print('Invalid emotion number {}'.format(emotion))
                return None
            print('Emotion detected as {} ({} {}).'.format(categories[emotion], row_or_col, emotion))
            print('Press Enter to continue, or type the correct number (0, 1, or 2):')
            inp = input().strip()
            if inp == '':
                return emotion
            if inp in ['0', '1', '2']:
                return int(inp)
            print('Invalid input, please try again.')
            return None
        except Exception as e:
            raise e
    
    def _get_emotion(self, img) -> int:
        if self.model is None:
            print("Model not loaded. Defaulting to Neutral.")
            return 0

        # Resize image to match model input shape (150, 150)
        resized_img = cv2.resize(img, image_size)
        
        # Convert grayscale to RGB if model expects 3 channels
        if len(resized_img.shape) == 2:
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
            
        # Add batch dimension
        input_tensor = np.expand_dims(resized_img, axis=0)
        
        # Predict
        predictions = self.model.predict(input_tensor, verbose=0)
        emotion_index = np.argmax(predictions[0])
        
        return int(emotion_index)
    
    def get_move(self, board_state):
        row, col = None, None
        while row is None:
            row = self._get_row_or_col(True)
        while col is None:
            col = self._get_row_or_col(False)
        return row, col
