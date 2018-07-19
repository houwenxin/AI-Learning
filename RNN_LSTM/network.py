# -*- coding: UTF-8 -*-

'''
RNN-LSTM Recurrent Network
'''

import tensorflow as tf

# Build Model
class Model(object):
	def __init__(self, input, is_training, hidden_size, vocab_size, num_layers, dropout=0.5, init_scale=0.05):
		self.is_training = is_training
		self.input_obj = input
		self.batch_size = input.batch_size
		self.num_steps = input.num_steps
		self.hidden_size = hidden_size
		
		# Designate CPU to calculate.
		with tf.device("/cpu:0"):
			embedding = tf.Variable(tf.random_uniform([vocab_size, self.hidden_size], -init_scale, init_scale))
			# embedding_lookup return word vector.
			inputs = tf.nn.embedding_lookup(embedding, self.input_obj.input_data)
		if is_training and dropout < 1:
			inputs = tf.nn.dropout(inputs, dropout)
		
		# The second dimension is 2 because of 2 input for time t: C(t-1) and h(t-1)
		self.init_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, self.hidden_size])
		# State for every layer.
		state_per_layer_list = tf.unstack(self.init_state, axis=0)		

		# Initial State.
		rnn_tuple_state = tuple(
			[tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1]) for idx in range(num_layers)]
		)
		# Build a LSTM layer with hidden_size number of neuron (default 650).
		cell = tf.contrib.rnn.LSTMCell(hidden_size)
		if is_training and dropout < 1:
			cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
		
		if num_layers > 1:
			cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)
		# dynamic_rnn return: 
		# output: [20, #time(35), #neuron(650)]
		# state: [h(t) and C(t)]
		output, self.state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=rnn_tuple_state)
		# Flatten.
		output = tf.reshape(output, [-1, hidden_size])
		# Weight default: [650, 10000]
		softmax_w = tf.Variable(tf.random_uniform([hidden_size, vocab_size], -init_scale, init_scale))
		# Bias : vocab_size * 1 (10000 * 1)
		softmax_b = tf.Variable(tf.random_uniform([vocab_size], -init_scale, init_scale))
		
		# logits: Logistic Regression for Classification.
		logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
		# Reshape logits to 3-D Tensor, for calculating sequence loss
		# Default: [20, 35, 10000]
		logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])
		# Cross-Entropy
		loss = tf.contrib.seq2seq.sequence_loss(
			logits, # [20, 35, 10000]
			self.input_obj.targets, # [20, 35]
			tf.ones([self.batch_size, self.num_steps], dtype=tf.float32), # Weights
			average_across_timesteps=False,
			average_across_batch=True)

		# Update loss
		self.cost = tf.reduce_sum(loss)

		# Calculate Possibility
		self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, vocab_size]))
		# Predict
		self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
		# True Predictions
		correct_prediction = tf.equal(self.predict, tf.reshape(self.input_obj.targets, [-1]))
		# Calculate Accuracy
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		# Quit if is not training
		if not is_training:
			return
		# Code below is used for training
		# Set learning rate.
		self.learning_rate = tf.Variable(0.0, trainable=False)
		# tvars: trainable variables
		tvars = tf.trainable_variables()
		# Gradient Clipping to prevent gradient explosion
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)

		optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
		# optimizer.minimize() = 1.tf.gradients().(+ gradient clipping here) 2.apply_gradients()
		self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.train.get_or_create_global_step())
		# Update learning rate
		self.new_lr = tf.placeholder(tf.float32, shape=[])
		self.lr_update = tf.assign(self.learning_rate, self.new_lr)

	def assign_lr(self, session, lr_value):
		session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

			


		


