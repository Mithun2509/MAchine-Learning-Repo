from tensorflow import keras
import tensorflow as tf
import numpy as np


new_model = tf.keras.models.load_model('number_map_model.h5')
new_model.summary()
