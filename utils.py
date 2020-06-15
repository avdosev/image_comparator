from cv2 import imread, resize
from config import input_shape
from augment import augment_image


def load_image(filename):
    image = imread(filename)
    assert image is not None and image.size != 0, f"Какая то шняга с файлом: {filename}"
    return image


def resize_image(image):
    return resize(image, input_shape[:2])


def train_pipeline(filename):
    return resize_image(augment_image(load_image(filename)))


def general_pipeline(filename):
    return resize_image(load_image(filename))