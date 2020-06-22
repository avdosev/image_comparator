from sklearn.metrics import mean_absolute_error, r2_score
from config import *
import pandas as pd
import numpy as np
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
        print('Разность:', np.absolute(similarities-data['similarity'].to_numpy()))
    return -r2_score(data['similarity'], similarities)
