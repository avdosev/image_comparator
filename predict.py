import numpy as np
from config import *
from utils import *
from scipy.spatial.distance import cosine


def similarity_images(images_paths1, images_paths2, model):
    assert model.output_shape, (None, emb_size)
    assert len(images_paths1) == len(images_paths2)
    x1 = np.array([resize_image(load_image(filename)) for filename in images_paths1])
    x2 = np.array([resize_image(load_image(filename)) for filename in images_paths2])
    y1 = model.predict(x1)
    y2 = model.predict(x2)
    cosine_distances = np.array(map(lambda img1, img2: cosine(img1, img2), zip(y1, y2)))
    similarity = 1 - cosine_distances
    return similarity
