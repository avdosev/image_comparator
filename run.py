import tensorflow.keras as keras
from config import *
import os
import pandas as pd
from predict import similarity_images

model = keras.models.load_model(os.path.join(models_path, 'model_best.hdf5'))
model = keras.Model(inputs=[model.input], outputs=[model.layers[-2].output])

data = pd.read_csv('./dataset/evaluation.csv')
for i in (1, 2):
    data[f'image{i}'] = data[f'image{i}'].transform(lambda id: f"./dataset/evaluation/{id}.jpg")
similarities = similarity_images(data['image1'], data['image2'], model)
print("Похожести:", similarities)