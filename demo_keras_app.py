#!/usr/bin/env python

# Keras API R-MAC Layer Demo for TensorFlow 2
# copyright (c) 2020 IMATAG
# imatag.com
#
# Author: Vedran Vukotic

import numpy as np
import tensorflow as tf

from rmac import RMAC

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda

# load the pretinrained network from Keras Applications
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array



# load the base model
base_model = MobileNetV2()

# check the architecture and see where to attach our RMAC layer
# print(base_model.summary())

# create the new model consisting of the base model and a RMAC layer
layer = "out_relu"
base_out = base_model.get_layer(layer).output

rmac = RMAC(base_out.shape, levels=5, norm_fm=True, sum_fm=True)

# add RMAC layer on top
rmac_layer = Lambda(rmac.rmac, input_shape=base_model.output_shape, name="rmac_"+layer)

out = rmac_layer(base_out)
#out = Dense(1024)(out) # fc to desired dimensionality
model = model = Model(base_model.input, out)

# display architecture
print(model.summary())



# load a sample image
i = load_img('adorable-al-reves-animal-atigrado-248280.jpg', target_size=(224, 224))
x = img_to_array(i)
x = x[None, ...]
x = preprocess_input(x)

# obtain RMAC descriptor for the image
y = model.predict(x)
print("\nOut:")
print("Shape:  ", y.shape)
print("Values: ", y)
print("Min:    ", np.min(y))
print("Max:    ", np.max(y))

