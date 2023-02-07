# %%
import keras
import tensorflow as tf
import keras.layers as layers

model = keras.Sequential()

model.add(layers.Conv2D(
    filters=96, kernel_size=(11, 11),
    strides=(4, 4), input_shape=(227, 227, 3),
    activation='relu'
))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

model.add(layers.Conv2D(
    filters=256, kernel_size=(5, 5),
    strides=(1, 1), activation='relu',
    padding='same'
))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

model.add(layers.Conv2D(
    filters=384, kernel_size=(5, 5),
    strides=(1, 1), activation='relu'
))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(
    filters=384, kernel_size=(3, 3),
    strides=(1, 1), activation='relu',
    padding='same'
))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(
    filters=256, kernel_size=(3, 3),
    strides=(1, 1), activation='relu',
    padding='same'
))
model.add(layers.BatchNormalization())

model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(1000, activation='softmax'))

# %%
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.optimizers.SGD(lr=0.001),
    metrics=['accuracy']
)
model.summary()

# %%
import visualkeras
import matplotlib.pyplot as plt

plt.imshow(visualkeras.layered_view(model))
plt.show()