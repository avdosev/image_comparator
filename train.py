import numpy as np
import pandas as pd
import os
from cv2 import resize, imread
import efficientnet.tfkeras
from tensorflow import keras
import tensorflow as tf
from config import *
from networks import get_model
import math
from datetime import datetime


class ImagesSequence(keras.utils.Sequence):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_y = [np.zeros(images_count, dtype=np.int) for _ in range(len(batch_x))]
        for i, filename in enumerate(batch_x):
            f, _ = os.path.splitext(os.path.basename(filename))
            index = int(f[f.rfind('_'):])
            batch_y[i][index] = 1

        return np.array([
            resize(imread(file_name), input_shape)
            for file_name in batch_x]), np.array(batch_y)


train_data = ImagesSequence(os.listdir(images_folder))

model = get_model(input_shape, 64, images_count)

model.compile()
