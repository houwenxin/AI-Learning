# -*- coding: UTF-8 -*-

# Remove the warning
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import tensorflow as tf

const1 = tf.constant([[2, 2]])
const2 = tf.constant([[4],
		     [4]])

# Create a Matrix Multiple Operation
multiple = tf.matmul(const1, const2)

print(multiple)

with tf.Session() as sess:
	# Run the 'multiple' operation
	result = sess.run(multiple)
	print("Result of \'multiple\' is %s" %result)

if const1.graph is tf.get_default_graph():
	print("Const1's graph is the default graph")
