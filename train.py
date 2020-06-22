import numpy as np
import pandas as pd
import os
from utils import *
import efficientnet.tfkeras
import tensorflow as tf
from tensorflow import keras
from config import *
from networks import get_model
import math
from validation import score_model


class ImagesSequence(keras.utils.Sequence):
    def __init__(self, images, batch_size):
        def images_in_count_limit(filename):
            f, _ = os.path.splitext(os.path.basename(filename))
            index = int(f[f.rfind('_')+1:])
            return index < images_count
        self.images = list(filter(images_in_count_limit, images))
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.images) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size:
                              (idx + 1) * self.batch_size]
        batch_y = np.zeros((len(batch_x), images_count), dtype=np.int)
        for i, filename in enumerate(batch_x):
            f, _ = os.path.splitext(os.path.basename(filename))
            index = int(f[f.rfind('_')+1:])
            batch_y[i][index] = 1

        return np.array([
            train_pipeline(file_name)
            for file_name in batch_x]), np.array(batch_y)


class ValidationPrint(tf.keras.callbacks.Callback):
    def __init__(self, emb_index, details=False):
        super().__init__()
        self.emb_index = emb_index
        self.details = details

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            model = keras.Model(inputs=[self.model.input], outputs=[self.model.layers[self.emb_index].output])
            score = score_model(model, print_similarities=self.details)
            logs['val_score'] = score
            print('\n val_score: ', score)


def main():
    train_dataset = ImagesSequence([os.path.join(images_folder, filename) for filename in os.listdir(images_folder)],
                                   batch)

    model, emb_index = get_model(input_shape, emb_size, images_count)
    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    model.fit(
        train_dataset,
        epochs=epoch,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=2, verbose=0, mode="min"),
            ValidationPrint(emb_index),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(models_path, 'model_best.hdf5'),
                save_weights_only=False,
                monitor='val_score',
                mode='min',
                save_best_only=True
            )
        ]
    )

    model.save(model_name)
    model = keras.Model(inputs=[model.input], outputs=[model.layers[emb_index].output])
    print("Final validation")
    print('Score:', score_model(model, print_similarities=True))

    model = keras.models.load_model(os.path.join(models_path, 'model_best.hdf5'))
    model = keras.Model(inputs=[model.input], outputs=[model.layers[emb_index].output])
    print("Final best validation")
    print('Score:', score_model(model, print_similarities=True))


if __name__ == '__main__':
    main()
