# -*- coding: UTF-8 -*-

'''
RNN-LSTM Neural Network.
Training Method.
'''

from utils import *
from network import *

def train(train_data, vocab_size, num_layers, num_epochs, batch_size, model_save_name, learning_rate=1.0, max_lr_epoch=10, lr_decay=0.93, print_iter=50):
	# Load data (from utils.py)
	training_input = Input(batch_size=batch_size, num_steps=35, data=train_data)
	
	# Create Model (from network.py)
	m = Model(training_input, is_training=True, hidden_size=650, vocab_size=vocab_size, num_layers=num_layers)
	# Initialize global variables.
	init_op = tf.global_variables_initializer()
	# Initialize learning rate decay
	orig_decay = lr_decay

	with tf.Session() as sess:
		sess.run(init_op)
		
		# Coordinator, coordinate multithread training.
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		# Save model.
		saver = tf.train.Saver(max_to_keep=5)

		for epoch in range(num_epochs):
			# Learning rate start to decay after max_lr_epoch.
			new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0.0)
			m.assign_lr(sess, learning_rate * new_lr_decay)
			
			# Get current state. h(t-1), C(t-1)
			current_state = np.zeros((num_layers, 2, batch_size, m.hidden_size))
			# Get current time.
			curr_time = datetime.datetime.now()

			for step in range(training_input.epoch_size):
				if step % print_iter != 0:
					cost, _, current_state = sess.run([m.cost, m.train_op, m.state], feed_dict={m.init_state: current_state})
				else:
					seconds = (float((datetime.datetime.now() - curr_time).seconds) / print_iter)
					curr_time = datetime.datetime.now()
					cost, _, current_state, acc = sess.run([m.cost, m.train_op, m.state, m.accuracy], feed_dict={m.init_state: current_state})
					print("Epoch {}, Step {}, Cost: {:.3f}, Accuracy: {:.3f}, Seconds per step: {:.3f}".format(epoch, step, cost, acc, seconds))
		
		
			saver.save(sess, save_path + '/' + model_save_name, global_step=epoch)
		# Save the final model
		saver.save(sess, save_path + '/' + model_save_name + '-final')	
		# Close threads
		coord.request_stop()
		coord.join(threads)

if __name__ == "__main__":
	if args.data_path:
		data_path = args.data_path

	train_data, valid_data, test_data, vocab_size, id_to_word = load_data(data_path)
	train(train_data, vocab_size, num_layers=2, num_epochs=70, batch_size=20, model_save_name='train-checkpoint')
