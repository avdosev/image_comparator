import numpy as np
import pandas as pd
import os
import efficientnet.tfkeras
from tensorflow import keras
import tensorflow as tf
from config import *
import math
from datetime import datetime
from utils import *
from scipy.spatial.distance import cosine

model = keras.models.load_model(model_name)
model = keras.Model(inputs=[model.input], outputs=[model.layers[-2].output])


def similarity_images(images_path):
    x = np.array([resize_image(load_image(filename)) for filename in images_path])
    y = model.predict(x)
    return 1 - cosine(y[0], y[1])


if __name__ == '__main__':
    first_image_path = './dataset/images/image_171.jpg'
    second_image_path = './dataset/images/image_172.jpg'
    similarity = similarity_images((first_image_path, second_image_path))
    print("Похожесть:", similarity)
    first_image_path = './dataset/images/image_708.jpg'
    second_image_path = './dataset/images/image_709.jpg'
    similarity = similarity_images((first_image_path, second_image_path))
    print("Похожесть:", similarity)
    first_image_path = './dataset/images/image_190.jpg'
    second_image_path = './dataset/images/image_709.jpg'
    similarity = similarity_images((first_image_path, second_image_path))
    print("Похожесть:", similarity)
