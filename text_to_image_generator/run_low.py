# %%
import tensorflow as tf
import numpy as np

# Define the generator model
def generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128 * 7 * 7, input_dim=100))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Reshape((7, 7, 128)))
    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(64, (5, 5), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Conv2D(1, (7, 7), padding='same', activation='tanh'))
    return model

# Define the discriminator model
def discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

# Define the combined generator and discriminator model
def gan(generator, discriminator):
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Compile the discriminator
discriminator = discriminator_model()
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])

# Compile the combined generator and discriminator
generator = generator_model()
discriminator.trainable = False
gan = gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4))

#Load the MNIST dataset
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()


#Rescale the images to [-1, 1]with a

X_train = (x_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=-1)

#Train the GAN

# %%
num_epochs = 200
batch_size = 128
for epoch in range(num_epochs):
    for i in range(X_train.shape[0] // batch_size):
    # Get a batch of real images
        real_images = X_train[i * batch_size:(i + 1) * batch_size]
        # Generate a batch of fake images
        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_images = generator.predict(noise)

        # Train the discriminator
        X = np.concatenate([real_images, fake_images])
        y = np.ones((2 * batch_size, 1))
        y[batch_size:] = 0
        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(X, y)

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        y = np.ones((batch_size, 1))
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, y)

    print(
        f"Epoch: {epoch + 1}/{num_epochs}, Discriminator Loss: {d_loss[0]:.4f}, Discriminator Accuracy: {d_loss[1] * 100:.2f}%, Generator Loss: {g_loss:.4f}")

# %%
import matplotlib.pyplot as plt

# %%
# plt.imshow(fake_images[5])
plt.imshow(real_images[60])
plt.show()

#%%
gan.save_weights('gan_weights')

# %%
gan.save('gan_model.h5')

# %%
noise1 = np.random.normal(0, 1, (64, 100))
generated_image = generator.predict(noise)


# %%
plt.imshow(generated_image)
plt.show()

# %%
generator.save_weights('generator_weight')
discriminator.save_weights('generator_weight')
generator.save('generator.h5')
discriminator.save('discriminator.h5')
