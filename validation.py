from sklearn.metrics import mean_absolute_error
from config import *
import pandas as pd
import os
from predict import similarity_images


def score_model(model, print_similarities=False):
    data = pd.read_csv('./dataset/test.csv')
    for i in (1, 2):
        data[f'image{i}'] = data[f'image{i}'].transform(lambda id: os.path.join(test_images_folder, f"image_{id}.jpg"))
    similarities = similarity_images(data['image1'], data['image2'], model)
    if print_similarities:
        print("Похожести:", similarities)
        print("Ожидаемые:", data['similarity'].to_numpy())
    return mean_absolute_error(data['similarity'], similarities)
