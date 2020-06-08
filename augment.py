from albumentations import Compose, OneOf,\
    Blur, MotionBlur, MedianBlur, HorizontalFlip, VerticalFlip, ShiftScaleRotate


def augment_image(image):
    aug = OneOf([
        MotionBlur(),
        MedianBlur(blur_limit=3),
        Blur(blur_limit=7),
        HorizontalFlip(),
        ShiftScaleRotate()
    ])
    image = aug(image=image)['image']
    return image
