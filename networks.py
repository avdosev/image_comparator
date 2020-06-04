from tensorflow import keras
import efficientnet.tfkeras as efn


def get_model(input_shape, emb_size, num_imgs):
    return keras.Sequential([
        keras.applications.MobileNetV2(input_shape=input_shape, weights='imagenet', include_top=False),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(emb_size, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(num_imgs, activation='relu'),
        keras.layers.Softmax(input_shape=(num_imgs,))
    ])
