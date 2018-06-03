# -*- coding: UTF-8 -*-

# Remove Warning
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import tensorflow as tf

# Create Graph
# Using a linear equation y = W * x + b
W = tf.Variable(2.0, dtype = tf.float32, name = "Weight")
b = tf.Variable(1.0, dtype = tf.float32, name = "Bias")
x = tf.placeholder(dtype = tf.float32, name = "Input")

with tf.name_scope("Output"):
	y = W * x + b

# Define saving path
path = "./tensorboardlog"

# Initialize all the Variables, pay attention to the 's'
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	writer = tf.summary.FileWriter(path, sess.graph)
	writer.close()
	result = sess.run(y, {x: 3.0})
	print("y = %s" % result)

