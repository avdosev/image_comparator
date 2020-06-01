import numpy as np
import pandas as pd
import os
import cv2
import efficientnet.tfkeras
from tensorflow import keras
import tensorflow as tf
from config import *
import math
from datetime import datetime
from scipy.spatial.distance import cosine

model = keras.models.load_model(model_name)
model = keras.Model(inputs=[model.input], outputs=[model.layers[-2].output])

