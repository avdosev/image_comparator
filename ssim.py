from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import mean_absolute_error
import pandas as pd
from config import *
import os
import numpy as np
from utils import general_pipeline


def ssim_images(images_paths1, images_paths2):
    assert len(images_paths1) == len(images_paths2)
    x1 = np.array([general_pipeline(filename) for filename in images_paths1])
    x2 = np.array([general_pipeline(filename) for filename in images_paths2])
    similarity = np.array([compare_ssim(*img, multichannel=True) for img in zip(x1, x2)])
    return similarity


def main():
    data = pd.read_csv('./dataset/test.csv')
    for i in (1, 2):
        data[f'image{i}'] = data[f'image{i}'].transform(lambda id: os.path.join(test_images_folder, f"image_{id}.jpg"))
    similarities = ssim_images(data['image1'], data['image2'])
    print("Похожести:", similarities)
    print("Ожидаемые:", data['similarity'].to_numpy())
    print('Разность:', np.absolute(similarities - data['similarity'].to_numpy()))
    return mean_absolute_error(data['similarity'], similarities)


if __name__ == '__main__':
    print('Score:', main())
