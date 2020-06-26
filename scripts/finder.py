import tensorflow.keras as keras
import os
import numpy as np
from utils import general_pipeline
from itertools import combinations
from predict import cosine_similarity_pair
from config import batch

print("finding images")
images = []
for root, _, files in os.walk('dataset/test'):
    images += [os.path.join(root, file) for file in files]

print("predict embedings")
model = keras.models.load_model('models/model_best.hdf5')
model = keras.Model(inputs=[model.input], outputs=[model.layers[-2].output])
embedings = model.predict(np.array(list(map(general_pipeline, images))), batch_size=batch)

print("combine embedings")
combination_embedings = list(combinations(embedings, 2))
comb_emb1 = [emb1 for emb1, emb2 in combination_embedings]
comb_emb2 = [emb2 for emb1, emb2 in combination_embedings]

print("compute cosine similarity")
distances = cosine_similarity_pair(comb_emb1, comb_emb2)

print("find min cosine distance")
min_index = np.argmin(distances)
min_val = distances[min_index]

combination_images = list(combinations(images, 2))
min_element = combination_images[min_index]

print(f'Min pair: {min_element[0]} | {min_element[1]}')
print(f'Min similarity: {min_val}')