import tensorflow as tf

hw = tf.constant("Hello World !")

with tf.Session() as sess:
	print(sess.run(hw))
