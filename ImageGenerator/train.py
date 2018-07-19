"""
Train DCGAN
"""
import glob
import numpy as np
from scipy import misc
import tensorflow as tf

from network import *

def train():
	# Load Image Data
	data = []
	for image in glob.glob("images/*"):
		image_data = misc.imreader(image)
		data.append(image_data)
	input_data = np.array(data)

	# Standardize data from range [0, 255] to [-1, 1]
	input_data = (input_data.astype(np.float32) - 127.5) / 127.5
	
	# Build generator and discriminator
	generator = generator_model()
	discriminator = discriminator_model()
	
	# Build network
	d_on_g = generator_containing_discriminator(generator, discriminator)
	
	# Adam Optimizer
	g_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)
	d_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)

	generator.compile(loss="binary_crossentropy", optimizer=g_optimizer)
	d_on_g.compile(loss="binary_crossentropy", optimizer=g_optimizer)
	discriminator.trainable = True
	discriminator.compile(loss="binary_crossentropy", optimizer=d_optimizer)

	
	for epoch in range(EPOCHS):
		for index in range(int(input_data.shape[0] / BATCH_SIZE)):
			input_batch = input_data[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]
			# Noise
			random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
			generated_images = g.predict(random_data, verbose=0)
			
			input_batch = np.concatenate((input_batch, generated_images))
			output_batch = [1] * BATCH_SIZE + [0] * BATCH_SIZE
			
			# Train the discriminator
			d_loss = d.train_on_batch(input_batch, output_batch)

			# When training generator, let discriminator untrainable
			discriminator.trainable = False

			random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))	
			generator.loss = d_on_g.train_on_batch(random_data, [1] * BATCH_SIZE)
			
			discriminator.trainable = True

			print("Step %d Generator Loss: %f Discriminator Loss: %f" %(index, g_loss, d_loss))
		# Save model
		if epoch % 10 == 9:
			generator.save_weights("generator_weight", True)
			discriminator.save_weights("discriminator_weight", True)


if __name__ == "__main__":
	train()
