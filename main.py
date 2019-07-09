#!/usr/bin/env python

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.optimizers as optimizers
import numpy as np

# TEMPORARY STUFF UNRELATED TO MIDIS
print(tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, 3)
x_test = np.expand_dims(x_test, 3)
print(x_train.shape)

model = tf.keras.models.Sequential([
    layers.Conv2D(16, (5, 5), input_shape=(28, 28, 1)),
    layers.Dropout(0.03),
    layers.Activation('relu'),
    layers.Flatten(),
    layers.Dense(10, activation=tf.nn.softmax),
    layers.Dropout(0.03),
])
model.compile(optimizer=optimizers.Adam(learning_rate=0.005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=128)
model.evaluate(x_test, y_test)
