# %%
import tensorflow as tf

# %%
# Define the generator network
def generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128 * 7 * 7, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7, 7, 128)))
    assert model.output_shape == (None, 7, 7, 128)  # Note: None is the batch size

    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

# %%
# Define the discriminator network
def discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    return model


# %%
# Define the combined generator and discriminator model
def gan(generator, discriminator):
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


# %%
# Compile the discriminator
discriminator = discriminator_model()
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])

# %%
# Compile the combined generator and discriminator
generator = generator_model()
discriminator.trainable = False
gan_model = gan(generator, discriminator)
gan_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4))

# %%
# Train the GAN
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_train = (x_train - 127.5) / 127.5

# %%
for epoch in range(100):
    noise = tf.random.normal([x_train.shape[0], 100])
    generated_images = generator(noise, training=True)
    print(f'running {epoch}')
    real_labels = tf.ones((x_train.shape[0], 1))
    fake_labels = tf.zeros((x_train.shape[0], 1))

    d_loss_real = discriminator.train_on_batch(x_train, real_labels)
    d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
    d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])

    noise = tf.random.normal([x_train.shape[0], 100])
    g_loss = gan_model.train_on_batch(noise, real_labels)
    # if epoch % 10 == 0:
    print("Epoch {}: D Loss = {}, G Loss = {}".format(epoch, d_loss, g_loss))

