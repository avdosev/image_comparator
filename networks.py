from tensorflow import keras
import efficientnet.tfkeras as efn


def get_model(input_shape, emb_size, num_imgs):
    base_model = keras.applications.MobileNetV2(input_shape=input_shape, weights='imagenet', include_top=False)
    x = base_model.output
    x1 = keras.layers.GlobalAveragePooling2D()(x)
    x2 = keras.layers.GlobalMaxPooling2D()(x)
    x = keras.layers.average([x1, x2])
    x = keras.layers.Dense(emb_size, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(num_imgs, activation='relu')(x)
    x = keras.layers.Softmax(input_shape=(num_imgs,))(x)
    return keras.Model(inputs=base_model.input, outputs=x)

