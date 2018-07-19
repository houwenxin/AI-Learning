"""
DCGAN: Deep Convolutional Generative Adversial Network
"""

import tensorflow as tf

# Hyperparameters:
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
BETA_1 = 0.5

# Discriminator Model
def discriminator_model():
	model = tf.keras.models.Sequential()
	# Convolutional Layer 1
	model.add(tf.keras.layers.Conv2D(
		64, # Depth
		(5, 5), # Filter Size
		padding='same', # Keep Size
		input_shape=(64, 64, 3) # Input Size
		))
	# Activation Layer and Pool Layer 1
	model.add(tf.keras.layers.Activation("tanh"))
	model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

	# Convolutional Layer 2
        model.add(tf.keras.layers.Conv2D(
                128, # Depth
                (5, 5), # Filter Size
                ))
	 # Activation Layer and Pool Layer 2
        model.add(tf.keras.layers.Activation("tanh"))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
	 # Convolutional Layer 2
        model.add(tf.keras.layers.Conv2D(
                128, # Depth
                (5, 5), # Filter Size
                ))
	 # Activation Layer and Pool Layer 2
        model.add(tf.keras.layers.Activation("tanh"))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
	# Flatten Layer and Dense Layer with 1024 neuron
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(1024))
	model.add(tf.keras.layers.Activation("tanh"))
	
	model.add(tf.keras.layers.Dense(1))
        model.add(tf.keras.layers.Activation("sigmoid"))
	
	return model

# Generator Model: generate pictures from random numbers
def generator_model():
	model = tf.keras.models.Sequential()
	# Input Size: [100], #neuron: [1024]
	model.add(tf.keras.layers.Dense(input_dim=100, units=1024))
	model.add(tf.keras.layers.Activation("tanh"))
	model.add(tf.keras.layers.Dense(128 * 8 * 8))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Activation("tanh"))
	model.add(tf.keras.layers.Reshape((8, 8, 128), input_shape=(128 * 128 * 8, ))) # 8 x 8
	model.add(tf.keras.layers.UpSampling2D(size=(2, 2))) # 16 x 16
	model.add(tf.keras.layers.Conv2D(128, (5, 5), padding="same"))
	model.add(tf.keras.layers.Activation("tanh"))
	model.add(tf.keras.layers.UpSampling2D(size=(2, 2))) # 32 x 32
	model.add(tf.keras.layers.Conv2D(128, (5, 5), padding="same"))
	model.add(tf.keras.layers.Activation("tanh"))
	model.add(tf.keras.layers.UpSampling2D(size=(2, 2))) # 64 x 64
	model.add(tf.keras.layers.Conv2D(3, (5, 5), padding="same"))
	model.add(tf.keras.layers.Activation("tanh"))
	
	return model

# Build a Sequential object, including a generator and a discriminator
# Input -> generator -> discriminator -> Output
def generator_containing_discriminator(generator, discriminator):
	model = tf.keras.models.Sequential()
	model.add(generator)
	# discriminator is untrainable at first.
	discriminator.trainable = False
	model.add(discriminator)
	return model
