from cv2 import imread, resize
from config import input_shape
from augment import augment_image


def load_image(filename):
    return imread(filename)


def resize_image(image):
    return resize(image, input_shape[:2])


def train_pipeline(filename):
    return resize_image(augment_image(load_image(filename)))


def general_pipeline(filename):
    return resize_image(load_image(filename))