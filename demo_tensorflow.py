#!/usr/bin/env python

# R-MAC Layer Demo for TensorFlow 2
# copyright (c) 2020 IMATAG
# imatag.com
#
# Author: Vedran Vukotic

import numpy as np
from rmac import RMAC
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Lambda
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

batch_size = 32
n_classes = 10
n_samples = 10000


# generate random data
X = np.random.rand(n_samples, 32, 32, 3)
y = np.eye(n_classes)[np.random.choice(n_classes, n_samples)]


# Not a sensible architecture, just a demo
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25, name='block2'))

# create and add an RMAC layer
rmac = RMAC(model.output_shape, norm_fm=True, sum_fm=True)
model.add(Lambda(rmac.rmac, name="rmac"))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation('softmax'))

opt = Adam(lr=0.0001)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# show model
print(model.summary())

# dummy training
model.fit(X, y,
          batch_size=batch_size,
          epochs=5,
          shuffle=True)
