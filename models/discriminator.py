from tensorflow import keras
from helpers import conv2d, deconv2d

def create(input_shape=(512, 512, 3), summary=True, name="discriminator"):
    inputs = keras.layers.Input(shape=input_shape)
    
    conv1 = conv2d(inputs, 16, "conv1", (3,3), bn=False)
    max_pool = keras.layers.MaxPool2D(2, name="max_pool1")(conv1)
    conv2 = conv2d(max_pool, 32, "conv2", (3,3), bn=False)
    max_pool = keras.layers.MaxPool2D(2, name="max_pool2")(conv2)
    conv3 = conv2d(max_pool, 64, "conv3", (3,3), bn=False)
    max_pool = keras.layers.MaxPool2D(2, name="max_pool3")(conv3)
    conv4 = conv2d(max_pool, 128, "conv4", (3,3), bn=False)
    max_pool = keras.layers.MaxPool2D(2, name="max_pool4")(conv4)
    flatten = keras.layers.Flatten(name="flatten")(max_pool)
    dense1 = keras.layers.Dense(128, activation="relu", name="dense1")(flatten)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="outputs")(dense1)
    
    model = keras.models.Model(inputs, outputs, name=name)
    None if not summary else model.summary()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model