"""
Generate pictures with trained DCGAN generator.
"""
import tensorflow as tf
# PIL: Python Image Library
from PIL import Image
import numpy as np

from network import *

def generate():
	generator = generator_model()
	
	generator.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1))
	# Load Weight
	generator.load_weights("generator_weight")

	# Give Noise
	random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
	
	images = generator.predict(random_data, verbose=1)

	for i in range(BATCH_SIZE):
		# Reverse Standardize to range [0, 255]
		image = images[i] * 127.5 + 127.5
		Image.fromarray(image.astype(np.uint8)).save("./generated_images/image-%s.png" %i)

if __name__ == "__main__":
	generate()
