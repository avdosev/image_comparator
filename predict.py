import numpy as np
import pandas as pd
import os
import efficientnet.tfkeras
from tensorflow import keras
import tensorflow as tf
from config import *
from datetime import datetime
from utils import *
from scipy.spatial.distance import cosine

model = keras.models.load_model(model_name)
model = keras.Model(inputs=[model.input], outputs=[model.layers[-2].output])
assert model.output_shape, (None, emb_size)

def similarity_images(images_paths1, images_paths2):
    assert len(images_paths1) == len(images_paths2)
    x1 = np.array([resize_image(load_image(filename)) for filename in images_paths1])
    x2 = np.array([resize_image(load_image(filename)) for filename in images_paths2])
    y1 = model.predict(x1)
    y2 = model.predict(x2)
    cosine_distances = np.array([cosine(image1, image2) for image1, image2 in zip(y1, y2)])
    similarity = 1 - cosine_distances
    return similarity


if __name__ == '__main__':
    data = pd.read_csv('./dataset/test.csv')
    for i in range(1, 3):
        data[f'image{i}'] = data[f'image{i}'].transform(lambda id: os.path.join(test_images_folder, f"image_{id}.jpg"))
    similarities = similarity_images(data['image1'], data['image2'])
    print("Похожесть:", similarities)
