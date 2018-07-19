'''
MNIST is a gray-scale handwriting database.
Dataset Size: 55000 * 28 * 28
Recognition with tensoflow CNN.
'''
# Remove the warning
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import numpy as np
import tensorflow as tf

# Download MNIST database.
from tensorflow.examples.tutorials.mnist import input_data

# In one-hot encoding.
mnist = input_data.read_data_sets('mnist_data', one_hot=True)

# Define placeholders for x and y.
input_x = tf.placeholder(tf.float32, [None, 28 * 28]) / 255.	# Normalize input_x.
output_y = tf.placeholder(tf.int32, [None, 10])
# Need to reshpae input_x for tf.conv2d.
input_images = tf.reshape(input_x, [-1, 28, 28, 1])

# Test set size: 3000
test_x = mnist.test.images[:3000]
test_y = mnist.test.labels[:3000]


# BUild convolutional neural network.
# 1st ConvLayer. Output Size: 28*28*32
conv1 = tf.layers.conv2d(
	inputs=input_images,	# Input Size: 28*28*1
	filters=32,		# Depth: 32
	kernel_size=[5, 5],
	strides=1,
	padding='same',	# Keep Size: 28*28. 
	activation=tf.nn.relu	# Activation Function
	)
# 1st Pool Layer. Output Size: 14*14*32
pool1 = tf.layers.max_pooling2d(
	inputs=conv1,		#Input Size: 28*28*32
	pool_size=[2, 2],
	strides=2
	)
# 2nd ConvLayer. Output Size: 14*14*64
conv2 = tf.layers.conv2d(
	inputs=pool1,
	filters=64,
	kernel_size=[5, 5],
	strides=1,
	padding='same',
	activation=tf.nn.relu
	)
# 2nd Pool Layer. Output Size: 7*7*64
pool2 = tf.layers.max_pooling2d(
	inputs=conv2,
	pool_size=[2, 2],
	strides=2
	)
# Flat.
flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
# Full Connection Layer.
dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
# Dropout: rate = 0.5
dropout = tf.layers.dropout(inputs=dense, rate=0.5)
# Output Layer.
logits = tf.layers.dense(inputs=dropout, units=10)
# Calculate Error (Cross Entropy)
loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logits)
# Adam Optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
# Calculate Accuracy.
# tf.metrics.accuracy(): return (accuracy, update_op), Create 2 local variables.
accuracy = tf.metrics.accuracy(
	labels=tf.argmax(output_y, axis=1),
	predictions=tf.argmax(logits, axis=1))[1]
# Create Session.
sess = tf.Session()
# Initialize global and local Variables
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)

for i in range(20000):
	batch = mnist.train.next_batch(50) # Get next batch, size:50
	train_loss, train_op_ = sess.run([loss, train_op], {input_x: batch[0], output_y: batch[1]})
	if i % 100 == 0:
		test_accuracy = sess.run(accuracy, {input_x: test_x, output_y: test_y})
		print("Step:%d, Train Loss=%.4f, [Test accuracy=%.2f]") \
			% (i, train_loss, test_accuracy)
sess.close()
