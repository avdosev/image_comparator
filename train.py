import numpy as np
import pandas as pd
import os
from utils import *
import efficientnet.tfkeras
from tensorflow import keras
from config import *
from networks import get_model
import math
from datetime import datetime
from validation import score_model


class ImagesSequence(keras.utils.Sequence):
    def __init__(self, images, batch_size):
        self.images = images
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


train_dataset = ImagesSequence([os.path.join(images_folder, filename) for filename in os.listdir(images_folder)],
                               batch)

model, emb_index = get_model(input_shape, emb_size, images_count)
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_dataset,
    epochs=epoch,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=2, verbose=0, mode="min"),
    ]
)

if not os.path.exists(models_path):
    os.makedirs(models_path)

model.save(model_name)
model = keras.Model(inputs=[model.input], outputs=[model.layers[emb_index].output])

print('Score:', score_model(model, print_similarities=True))
