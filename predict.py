import numpy as np
from config import *
from utils import *
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances


def cosine_similarity_pair(arr1, arr2):
    assert len(arr1) == len(arr2)
    c_distances = np.array([cosine_distances(y1.reshape(1, -1), y2.reshape(1, -1))[0][0] for y1, y2 in zip(arr1, arr2)])
    return 1 - np.clip(c_distances, 0, 1)


def similarity_images(images_paths1, images_paths2, model):
    assert model.output_shape, (None, emb_size)
    assert len(images_paths1) == len(images_paths2)

    x1 = [resize_image(load_image(filename)) for filename in images_paths1]
    x2 = [resize_image(load_image(filename)) for filename in images_paths2]
    y = model.predict(np.array(x1+x2))

    b = len(images_paths1)
    y1, y2 = y[:b], y[b:]
    return cosine_similarity_pair(y1, y2)
