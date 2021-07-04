import tensorflow as tf
from keras.layers import *
from keras.models import *

model = Sequential()
model.add(Dense(256, input_shape=(2,)))
model.add(Dense(1, activation='softmax'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.save("dummymodel")
