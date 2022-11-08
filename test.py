import pickle

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import os
from tensorflow import keras
import pandas as pd

lst = os.listdir("test")
batch_size = 32
img_height = 250
img_width = 250
class_names = ['enface', 'profile']
model = keras.models.load_model(os.path.abspath("model/model.hdf5"))
columns = {'filename': [], 'label': []}

for name in os.listdir("test"):
    img = tf.keras.utils.load_img(
        os.path.abspath(f"test/{name}"), target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    columns['filename'].append(f"{name.lstrip()}")
    columns["label"].append(np.argmax(score))
with open("dict.txt", "wb") as d:
    pickle.dump(columns, d)
df = pd.DataFrame(columns)
print(df)
df.to_csv("result.csv")
