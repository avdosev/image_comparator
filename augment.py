from albumentations import *


def augment_image(image):
    aug = OneOf([
        CLAHE(),
        MotionBlur(blur_limit=3),
        MedianBlur(blur_limit=3),
        Blur(blur_limit=3),
        HorizontalFlip(),
        ShiftScaleRotate(rotate_limit=15),
        ImageCompression(),
        RandomGamma(),
        ToGray(),
        ToSepia()
    ])
    image = aug(image=image)['image']
    return image
