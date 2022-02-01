import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation='relu',name='dense-128-relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax',name='dense-128-softmax'))


model.save("model.h5")
