from cv2 import imread, resize
from config import input_shape


def augment_image(image):
    return image


def load_image(filename):
    return imread(filename)


def resize_image(image):
    return resize(image, input_shape[:2])
