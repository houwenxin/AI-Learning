'''
Linear Regression with Gradient Descent
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Create data
points_num = 100
vectors = []

# Generate 100 points with numpy.random
# (x, y) is subordinate to linear equation: y = 0.1 * x + 0.2
# Weight: 0.1, Bias: 0.2
for i in xrange(points_num):
	# Normal Distribution with noise
	x1 = np.random.normal(0.0, 0.66)
	y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0, 0.04)
	vectors.append([x1, y1])

# Data x and y
x_data = [v[0] for v in vectors]
y_data = [v[1] for v in vectors]

# Figure 1: visualize the data (red * points)
plt.plot(x_data, y_data, 'r*', label='Original data')
plt.title("Linear Regression with Gradient Descent")
plt.legend()
plt.show()

# Build a LR model
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # Initialize Weight
b = tf.Variable(tf.zeros([1])) # Initialize Bias
y_predict = W * x_data + b # Output y

# Loss Function (Cost Function)
# Calculate (y - y_data) ^ 2 / points_num
loss = tf.reduce_mean(tf.square(y_predict - y_data))

# Use Gradient Descent Optimizer to minimize loss, learning rate = 0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# train for 20 epochs
	for step in xrange(20):
		sess.run(train)
		# print parameters
		print("Step=%d, Loss=%f, Weight=%f, Bias=%f") \
			% (step, sess.run(loss), sess.run(W), sess.run(b))
	# Figure 2: plot points and best-fitted line
	plt.plot(x_data, y_data, 'r*', label="Original data")
	plt.title("Linear Regression with Gradient Descent")
	plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label="Best Fitted Line")
	plt.legend()
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()
		

